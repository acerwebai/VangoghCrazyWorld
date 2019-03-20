from __future__ import print_function
import sys
sys.path.insert(0, 'src')
import transform, numpy as np, vgg, pdb, os
import scipy.misc
import tensorflow as tf
from utils import save_img, get_img, exists, list_files
from argparse import ArgumentParser
from collections import defaultdict
import time
import json
import subprocess
import numpy
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer

#BATCH_SIZE = 4
BATCH_SIZE = 4 #mcky
DEVICE = '/gpu:0'

# example for eval:
#    python evaluate.py --data-format NHWC --num-base-channels 16 --checkpoint ckpts/udnie-nhwc_nf16_b01_1e2 --in-path examples/content/chicago.jpg --out-path chicago_undie.jpg
#
#  to execute and generate .onnx model:
#  1. eval and generate 'graph.pbtxt' and 'saver' files in 'tf-models' folder.  the folder name in 'tf-models' folder is checkpoint folder name.
#       python evaluate.py --data-format NHWC --num-base-channels 16 --checkpoint ckpts/udnie-nhwc_nf16_b01_1e2 --in-path examples/content/chicago.jpg --out-path chicago_undie.jpg
#
#  2. freeze_graph to create .pb file
#       python -m tensorflow.python.tools.freeze_graph --input_graph=tf-models/udnie-nhwc_nf16_b01_1e2/graph.pbtxt --input_checkpoint=tf-models/udnie-nhwc_nf16_b01_1e2/saver --output_graph=udnie-nhwc_nf16_b01_1e2.pb --output_node_names="output"
#
#  3. tf2onnx to convert .pb file to .onnx file in tf2onnx_models folder (todo: validate again with 'data-format' set to NCHW)
#       python -m tf2onnx.convert --input muse_frozen.pb --inputs img_placeholder --outputs add_37:0 --opset 8 --output tf2onnx_models/la_muse.onnx
#

def ffwd_video(path_in, path_out, checkpoint_dir, device_t='/gpu:0', batch_size=4,
                data_format='NHWC', num_base_channels=32 # more cli params
                ):
    video_clip = VideoFileClip(path_in, audio=False)
    video_writer = ffmpeg_writer.FFMPEG_VideoWriter(path_out, video_clip.size, video_clip.fps, codec="libx264",
                                                    preset="medium", bitrate="2000k",
                                                    audiofile=path_in, threads=None,
                                                    ffmpeg_params=None)

    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), \
            tf.Session(config=soft_config) as sess:
        batch_shape = (batch_size, video_clip.size[1], video_clip.size[0], 3)
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

        #preds = transform.net(img_placeholder)

        if data_format == 'NHWC':
            #NHWC path
            preds = transform.net(img_placeholder, data_format=data_format, num_base_channels=num_base_channels)
        else:
            #NCHW path
            img_placeholder_nchw = tf.transpose(img_placeholder, [0,3,1,2])
            preds_nchw = transform.net(img_placeholder_nchw, data_format=data_format, num_base_channels=num_base_channels)
            preds = tf.transpose(preds_nchw,[0,2,3,1])

        # add output node
        preds = tf.identity(preds, "output")
        #print("tf.identity: {}".format(preds))

        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        X = np.zeros(batch_shape, dtype=np.float32)

        def style_and_write(count):
            for i in range(count, batch_size):
                X[i] = X[count - 1]  # Use last frame to fill X
            _preds = sess.run(preds, feed_dict={img_placeholder: X})
            for i in range(0, count):
                video_writer.write_frame(np.clip(_preds[i], 0, 255).astype(np.uint8))

        frame_count = 0  # The frame count that written to X
        for frame in video_clip.iter_frames():
            X[frame_count] = frame
            frame_count += 1
            if frame_count == batch_size:
                style_and_write(frame_count)
                frame_count = 0

        if frame_count != 0:
            style_and_write(frame_count)

        video_writer.close()


# get img_shape
def ffwd(data_in, paths_out, checkpoint_dir, device_t='/gpu:0', batch_size=4,
        data_format='NHWC', num_base_channels=32 # more cli params
        ):
    assert len(paths_out) > 0
    is_paths = type(data_in[0]) == str
    if is_paths:
        assert len(data_in) == len(paths_out)
        img_shape = get_img(data_in[0]).shape
        #img_shape = get_img(data_in[0],(256,256,3)).shape #mcky, make the input size the same as training size
        #img_shape = get_img(data_in[0],(128,128,3)).shape #mcky, MX150
    else:
        assert data_in.size[0] == len(paths_out)
        img_shape = X[0].shape

    g = tf.Graph()
    batch_size = min(len(paths_out), batch_size)
    curr_num = 0
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), \
            tf.Session(config=soft_config) as sess:
        batch_shape = (batch_size,) + img_shape
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')
        
        #mcky
        print ("---------- ffwd() - transform.net() ----------")

        if data_format == 'NHWC':
            #NHWC path
            preds = transform.net(img_placeholder, data_format=data_format, num_base_channels=num_base_channels)
        else:
            #NCHW path
            img_placeholder_nchw = tf.transpose(img_placeholder, [0,3,1,2])
            preds_nchw = transform.net(img_placeholder_nchw, data_format=data_format, num_base_channels=num_base_channels)
            preds = tf.transpose(preds_nchw,[0,2,3,1])
        
        # add output node
        preds = tf.identity(preds, "output")
        #print("tf.identity: {}".format(preds))
        
        #mcky, Variables are printed here
        #for v in tf.global_variables():
        #    print (v)
        
        print ("---------- tf.train.Saver() ----------")
        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path) #mcky, restore variable error if image input size is not the same as size in training.
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        print ("---------- num_iters ----------")
        num_iters = int(len(paths_out)/batch_size)
        for i in range(num_iters):
            pos = i * batch_size
            curr_batch_out = paths_out[pos:pos+batch_size]
            if is_paths:
                curr_batch_in = data_in[pos:pos+batch_size]
                X = np.zeros(batch_shape, dtype=np.float32)
                for j, path_in in enumerate(curr_batch_in):
                    img = get_img(path_in)

                    #print ("get_img():{}".format(img))#mcky
                    #img = get_img(path_in,(256,256,3)) #mcky, make the input size the same as training size
                    #img = get_img(path_in,(128,128,3)) #mcky, MX150
                    assert img.shape == img_shape, \
                        'Images have different dimensions. ' +  \
                        'Resize images or use --allow-different-dimensions.'
                    X[j] = img
            else:
                X = data_in[pos:pos+batch_size]

            _preds = sess.run(preds, feed_dict={img_placeholder:X})
            for j, path_out in enumerate(curr_batch_out):
                #print ("_preds[j]:{}".format(_preds[j]))#mcky
                save_img(path_out, _preds[j])
                
                # mcky, create a folder under 'tf-models'
                tf_models_dir = "tf-models"
                if not os.path.isdir(tf_models_dir):
                    os.mkdir (tf_models_dir)

                tf_model_dirname = os.path.basename(checkpoint_dir)
                tf_model_dirpath = os.path.join(tf_models_dir, tf_model_dirname)
                if not os.path.isdir(tf_model_dirpath):
                    os.mkdir (tf_model_dirpath)

                # mcky, save to checkpoint files
                print ("save saver...")
                tf_model_fpath = os.path.join(tf_model_dirpath, "saver")
                saver.save(sess, tf_model_fpath)
            
                # mcky, save graph to .pbtxt files
                print ("save graph...")
                graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)  # 'add_shapes' adds '_output_shapes' attribute to every node.  needed for tf2onnx
                tf.train.write_graph(graph_def, tf_model_dirpath, "graph.pbtxt")
                
        remaining_in = data_in[num_iters*batch_size:]
        remaining_out = paths_out[num_iters*batch_size:]
    if len(remaining_in) > 0:
        ffwd(remaining_in, remaining_out, checkpoint_dir, 
            device_t=device_t, batch_size=1,
            data_format=data_format, num_base_channels=num_base_channels)

def ffwd_to_img(in_path, out_path, checkpoint_dir, device='/cpu:0',
                data_format='NHWC', num_base_channels=32 # more cli params
                ):
    paths_in, paths_out = [in_path], [out_path]
    ffwd(paths_in, paths_out, checkpoint_dir, batch_size=1, device_t=device,
        data_format=data_format, num_base_channels=num_base_channels)

def ffwd_different_dimensions(in_path, out_path, checkpoint_dir, 
            device_t=DEVICE, batch_size=4,
            data_format='NHWC', num_base_channels=32 # more cli params
            ):
    in_path_of_shape = defaultdict(list)
    out_path_of_shape = defaultdict(list)
    for i in range(len(in_path)):
        in_image = in_path[i]
        out_image = out_path[i]
        shape = "%dx%dx%d" % get_img(in_image).shape
        in_path_of_shape[shape].append(in_image)
        out_path_of_shape[shape].append(out_image)
    for shape in in_path_of_shape:
        print('Processing images of shape %s' % shape)
        ffwd(in_path_of_shape[shape], out_path_of_shape[shape], 
            checkpoint_dir, device_t, batch_size,
            data_format=data_format, num_base_channels=num_base_channels)

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint_dir',
                        help='dir or .ckpt file to load checkpoint from',
                        metavar='CHECKPOINT', required=True)

    parser.add_argument('--in-path', type=str,
                        dest='in_path',help='dir or file to transform',
                        metavar='IN_PATH', required=True)

    help_out = 'destination (dir or file) of transformed file or files'
    parser.add_argument('--out-path', type=str,
                        dest='out_path', help=help_out, metavar='OUT_PATH',
                        required=True)

    parser.add_argument('--device', type=str,
                        dest='device',help='device to perform compute on',
                        metavar='DEVICE', default=DEVICE)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size',help='batch size for feedforwarding',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--allow-different-dimensions', action='store_true',
                        dest='allow_different_dimensions', 
                        help='allow different image dimensions')

    # more cli params
    parser.add_argument('--data-format',
                        dest='data_format', 
                        type=str,
                        default='NHWC',
                        help='data format, NHWC or NCHW.  default is NHWC')

    parser.add_argument('--num-base-channels',
                        dest='num_base_channels', 
                        type=int,
                        default=32,
                        help='number of base channels in 1st layer.  default is 32')

    return parser

def check_opts(opts):
    exists(opts.checkpoint_dir, 'Checkpoint not found!')
    exists(opts.in_path, 'In path not found!')
    if os.path.isdir(opts.out_path):
        exists(opts.out_path, 'out dir not found!')
        assert opts.batch_size > 0

def main():
    parser = build_parser()
    opts = parser.parse_args()
    check_opts(opts)

    if not os.path.isdir(opts.in_path):
        if os.path.exists(opts.out_path) and os.path.isdir(opts.out_path):
            out_path = \
                    os.path.join(opts.out_path,os.path.basename(opts.in_path))
        else:
            out_path = opts.out_path

        ffwd_to_img(opts.in_path, out_path, opts.checkpoint_dir,
                    device=opts.device,
                    data_format=opts.data_format, num_base_channels=opts.num_base_channels # more cli params
                    )
    else:
        files = list_files(opts.in_path)
        full_in = [os.path.join(opts.in_path,x) for x in files]
        full_out = [os.path.join(opts.out_path,x) for x in files]
        if opts.allow_different_dimensions:
            ffwd_different_dimensions(full_in, full_out, opts.checkpoint_dir, 
                    device_t=opts.device, batch_size=opts.batch_size,
                    data_format=opts.data_format, num_base_channels=opts.num_base_channels # more cli params
                    )
        else :
            ffwd(full_in, full_out, opts.checkpoint_dir, device_t=opts.device,
                    batch_size=opts.batch_size,
                    data_format=opts.data_format, num_base_channels=opts.num_base_channels # more cli params
                    )

if __name__ == '__main__':
    main()
