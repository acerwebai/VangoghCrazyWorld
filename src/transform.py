import tensorflow as tf, pdb

WEIGHTS_INIT_STDEV = .1

KERNEL_SIZE = 9 # default is 9

def net(image, data_format='NHWC', num_base_channels=32):
    conv1 = _conv_layer(image, num_base_channels, KERNEL_SIZE, 1, data_format=data_format)
    conv2 = _conv_layer(conv1, num_base_channels * 2, 3, 2, data_format=data_format)
    conv3 = _conv_layer(conv2, num_base_channels * 4, 3, 2, data_format=data_format)

    resid1 = _residual_block(conv3,  num_base_channels * 4, 3, data_format=data_format)
    resid2 = _residual_block(resid1, num_base_channels * 4, 3, data_format=data_format)
    resid3 = _residual_block(resid2, num_base_channels * 4, 3, data_format=data_format)
    resid4 = _residual_block(resid3, num_base_channels * 4, 3, data_format=data_format)
    resid5 = _residual_block(resid4, num_base_channels * 4, 3, data_format=data_format)
    
    
    # layers used in the original source

    conv_t1 = _conv_tranpose_layer(resid5, num_base_channels * 2, 3, 2, data_format=data_format)
    conv_t2 = _conv_tranpose_layer(conv_t1, num_base_channels , 3, 2, data_format=data_format)
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False, data_format=data_format)
    preds = tf.nn.tanh(conv_t3) * tf.constant([150.]) + tf.constant([255./2])

    
    
    # use upsample convolution to reduce checkerboard effects
    #     ref: http://distill.pub/2016/deconv-checkerboard/
    #up2_1 = _upsample2 (resid5, num_base_channels * 2, kernel_size=3, stride=1, data_format=data_format)
    #up2_2 = _upsample2 (up2_1,  num_base_channels,     kernel_size=3, stride=1, data_format=data_format)
    #preds = _conv_layer(up2_2,  3, KERNEL_SIZE, 1, instanceNorm=False, relu=False, data_format=data_format)

    #mcky 2018/12/25, this is to workaround a bug in tf2onnx (currently v0.3.2 in pip and v0.4.0 in master)
    #        The bug does not export the last node correctly (the output node does not have 'shape' attribute)
    #        The output node name is 'add_37' (observed from tensorflow 'summarize_graph' tool
    #        Adding an Identity node to receive the output from 'add_37'as last node would workaround this tf2onnx bug.
    #        Specify 'dummy_output' in tf2onnx as it's the desired output node. (Step 2 from above)
    #
    #        this should be removed during training
    #
    #preds = tf.identity(preds, "dummy_output")
    #print("tf.identity: {}".format(preds))
    
    return  preds

def _conv_layer(net, num_channels, filter_size, strides, instanceNorm=True, relu=True, data_format='NHWC'):
    if data_format == 'NHWC':
        weights_init = _conv_init_vars(net, num_channels, filter_size, data_format=data_format)
        strides_shape = [1, strides, strides, 1]
    
        # clc - improve border garbage
        net=tf.pad(net,tf.constant([[0, 0], [filter_size // 2, filter_size // 2],
               [filter_size // 2, filter_size // 2], [0, 0]]),"REFLECT")
        net = tf.nn.conv2d(net, weights_init, strides_shape, padding='VALID',data_format=data_format)
    
        #net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME',
        #    data_format=data_format
        #    )

        if instanceNorm:
            net = _instance_norm (net)
    else:
        weights_init = _conv_init_vars(net, num_channels, filter_size, data_format=data_format)
        strides_shape = [1, 1, strides, strides]

        # clc - improve border garbage
        net=tf.pad(net,tf.constant([[0, 0], [0, 0], [filter_size // 2, filter_size // 2],
               [filter_size // 2, filter_size // 2]]),"REFLECT")
        net = tf.nn.conv2d(net, weights_init, strides_shape, padding='VALID',data_format=data_format)
  

        #net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME',
        #    data_format=data_format
        #    )

        if instanceNorm:
            net = _instance_norm_nchw(net)

    if relu:
        net = tf.nn.relu(net)

    return net

# upsample_by_2() - upsample the input to twice the size in h & w using only reshape and concat.
#                   supports both NCHW and NHWC data format.
#
# tensorflow resize methods only works for NHWC.  calling upsample_by_2() for NCHW data format
#
def upsample_by_2 (x, c, h, w):
    bb  = tf.reshape(x,[-1,1])
    cc  = tf.concat([bb,bb],1)
    cc1 = tf.reshape(cc,[-1,w*2])
    cc2 = tf.concat([cc1,cc1],1)
    
    out = tf.reshape(cc2,[-1,c,h*2,w*2])

    return out

def _upsample2(net, out_channels, kernel_size, stride, data_format='NHWC'):
    if data_format == 'NHWC':
        c = net.shape[3]
        h = net.shape[1]
        w = net.shape[2]

        net = tf.image.resize_nearest_neighbor (net,[w*2,h*2])

    else:
        c = net.shape[1]
        h = net.shape[2]
        w = net.shape[3]
        
        net = upsample_by_2(net, c, h, w)
        
    net = _conv_layer(net, out_channels, kernel_size, stride, data_format=data_format)

    return net

def _conv_tranpose_layer(net, num_channels, filter_size, strides, data_format='NHWC'):
    weights_init = _conv_init_vars(net, num_channels, filter_size, transpose=True, data_format=data_format)

    if data_format == 'NHWC':
        batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
        new_rows, new_cols = int(rows * strides), int(cols * strides)
        new_shape = [batch_size, new_rows, new_cols, num_channels]
        tf_shape = tf.stack(new_shape)
        strides_shape = [1,strides,strides,1]
    
        net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME',
                                        data_format=data_format)
        net = _instance_norm(net)
    else:
        batch_size, in_channels, rows, cols = [i.value for i in net.get_shape()]
        new_rows, new_cols = int(rows * strides), int(cols * strides)
        new_shape = [batch_size, num_channels, new_rows, new_cols]
        tf_shape = tf.stack(new_shape)
        strides_shape = [1,1,strides,strides]
    
        net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME',
                                        data_format=data_format)
        net = _instance_norm_nchw(net)

    return tf.nn.relu(net)

def _residual_block(net, num_channels = 128, filter_size=3, data_format='NHWC'):
    tmp = _conv_layer(net, num_channels, filter_size, 1, data_format=data_format)
    return net + _conv_layer(tmp, num_channels, filter_size, 1, relu=False, data_format=data_format)

# InstanceNorm for NHWC
def _instance_norm(net, train=True):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)

    #epsilon = 1e-9 # 1e-3 originally.  set to 1e-9 to avoid a conversion issue for Intel OpenVINO model optimizer
    #normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    
    # change constants to tensors to avoid tflite 'rank 0'
    c_eps = tf.constant([1e-9])
    c_pow = tf.constant([.5])
    normalized = (net-mu)/(sigma_sq + c_eps)**(c_pow)

    var_shape = [channels]
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    net = scale * normalized + shift

    return net

# InstanceNorm for NCHW
def _instance_norm_nchw(net, train=True):
    batch, channels, rows, cols = [i.value for i in net.get_shape()]
    mu, sigma_sq = tf.nn.moments(net, [2,3], keep_dims=True)

    #epsilon = 1e-9 # 1e-3 originally.  set to 1e-9 to avoid a conversion issue for Intel OpenVINO model optimizer
    #normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    # change constants to tensors to avoid tflite 'rank 0'
    c_eps = tf.constant([1e-9])
    c_pow = tf.constant([.5])
    normalized = (net-mu)/(sigma_sq + c_eps)**(c_pow)

    var_shape = [1,channels,1,1]
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    net = scale * normalized + shift

    return net

def _conv_init_vars(net, out_channels, filter_size, transpose=False, data_format='NHWC'):
    
    if data_format == 'NHWC':
        _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    else:
        _, in_channels, rows, cols = [i.value for i in net.get_shape()]

    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)

    return weights_init
