from __future__ import print_function
import functools
import vgg, pdb, time
import tensorflow as tf, numpy as np, os
import transform
from utils import get_img

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'
DEVICES = 'CUDA_VISIBLE_DEVICES'

# np arr, np arr
def optimize(content_targets, style_target, content_weight, style_weight,
             tv_weight, vgg_path, epochs=2, print_iterations=1000,
             batch_size=4, save_path='saver/fns.ckpt', slow=False,
             learning_rate=1e-3, debug=False,

             # more cli params
             data_format='NHWC',
             num_base_channels=32
             ):

    #print ("optimize().data_format:{}".format(data_format))
    #print ("optimize().num_base_channels:{}".format(num_base_channels))

    if slow:
        batch_size = 1
    mod = len(content_targets) % batch_size
    if mod > 0:
        print("Train set has been trimmed slightly..")
        content_targets = content_targets[:-mod] 

    style_features = {}

    batch_shape = (batch_size,256,256,3)
    #batch_shape = (batch_size,128,128,3) #mcky,smaller size for MX150
    style_shape = (1,) + style_target.shape
    #print(style_shape)

    # precompute style features
    print("precompute style features") #mcky
    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_image_pre = vgg.preprocess(style_image)
        net = vgg.net(vgg_path, style_image_pre)
        style_pre = np.array([style_target])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={style_image:style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    with tf.Graph().as_default(), tf.Session() as sess:
        X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
        X_pre = vgg.preprocess(X_content)

        # precompute content features
        print("precompute content features") #mcky
        content_features = {}
        content_net = vgg.net(vgg_path, X_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

        if slow:
            preds = tf.Variable(
                tf.random_normal(X_content.get_shape()) * 0.256
            )
            preds_pre = preds
        else:
            if data_format == 'NHWC':
                #NHWC path
                preds = transform.net(X_content/255.0, data_format=data_format, num_base_channels=num_base_channels)
            else:
                #NCHW path
                
                # use NCHW transformer net.  bug vgg net needs NHWC.  so transposes needed for input and output.
                #nhwc --> nchw --> transform.net --> nhwc
                x_content = X_content/255.0
                X_content_nchw = tf.transpose(x_content,[0,3,1,2])
                preds_nchw = transform.net(X_content_nchw, data_format=data_format, num_base_channels=num_base_channels)
                preds = tf.transpose(preds_nchw,[0,2,3,1])

                print ("preds.shape:{}".format(preds.shape))

            preds_pre = vgg.preprocess(preds)
            print ("preds_pre.shape:{}".format(preds_pre.shape))

        net = vgg.net(vgg_path, preds_pre) # <-- mcky, this is to feed the output of ITN to VGG, along with content data set.

        content_size = _tensor_size(content_features[CONTENT_LAYER])*batch_size
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
        content_loss = content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
        )

        print("style_losses") #mcky
        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0,2,1])
            grams = tf.matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)

        style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size

        # total variation denoising
        print("total variation denoising") #mcky
        
        tv_y_size = _tensor_size(preds[:,1:,:,:])
        tv_x_size = _tensor_size(preds[:,:,1:,:])
        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size
        
        '''
        #mcky, tv for preds in nchw format
        tv_y_size = _tensor_size(preds[:,:,1:,:])
        tv_x_size = _tensor_size(preds[:,:,:,1:])
        y_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[1]-1,:])
        x_tv = tf.nn.l2_loss(preds[:,:,:,1:] - preds[:,:,:,:batch_shape[2]-1])
        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size
        '''

        loss = content_loss + style_loss + tv_loss

        # overall loss
        #print("overall loss") #mcky
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())
        
        #mcky, Variables are printed here
        #for v in tf.global_variables():
        #    print (v)

        import random
        uid = random.randint(1, 100)
        print("----------------------------------------------------------------------------------------------------") #mcky
        print("data_format:{}".format(data_format))
        print("num_base_channels:{}".format(num_base_channels))
        print("EPOCHS:{}".format(epochs)) #mcky
        print("UID: %s" % uid)
        for epoch in range(epochs):
            num_examples = len(content_targets)
            print("num_examples:{}".format(num_examples)) #mcky
            iterations = 0
            while iterations * batch_size < num_examples:
                start_time = time.time()
                curr = iterations * batch_size
                step = curr + batch_size
                X_batch = np.zeros(batch_shape, dtype=np.float32)
                for j, img_p in enumerate(content_targets[curr:step]):
                   X_batch[j] = get_img(img_p, (256,256,3)).astype(np.float32)
                   #X_batch[j] = get_img(img_p, (128,128,3)).astype(np.float32) #mcky, smaller size for MX150

                iterations += 1
                assert X_batch.shape[0] == batch_size

                feed_dict = {
                   X_content:X_batch
                }

                train_step.run(feed_dict=feed_dict)
                end_time = time.time()
                delta_time = end_time - start_time
                if debug:
                    print("UID: %s, batch time: %s" % (uid, delta_time))
                is_print_iter = int(iterations) % print_iterations == 0
                if slow:
                    is_print_iter = epoch % print_iterations == 0
                is_last = epoch == epochs - 1 and iterations * batch_size >= num_examples
                
                should_print = is_print_iter or is_last
                #should_print = True#mcky, is_print_iter or is_last
                if should_print:
                    to_get = [style_loss, content_loss, tv_loss, loss, preds]
                    test_feed_dict = {
                       X_content:X_batch
                    }

                    tup = sess.run(to_get, feed_dict = test_feed_dict)
                    _style_loss,_content_loss,_tv_loss,_loss,_preds = tup
                    losses = (_style_loss, _content_loss, _tv_loss, _loss)
                    if slow:
                       _preds = vgg.unprocess(_preds)
                    else:
                       saver = tf.train.Saver()
                       res = saver.save(sess, save_path)
                    yield(_preds, losses, iterations, epoch)

def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)
