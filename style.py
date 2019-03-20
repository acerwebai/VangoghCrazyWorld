from __future__ import print_function
import sys, os, pdb
sys.path.insert(0, 'src')
import numpy as np, scipy.misc 
from optimize import optimize
from argparse import ArgumentParser
from utils import save_img, get_img, exists, list_files
import evaluate

#mcky
'''
python style.py --checkpoint-dir ckpts --style examples/style/la_muse.jpg --train-path data/train2014 --test examples/content/chicago.jpg --test-dir test --vgg-path data/imagenet-vgg-verydeep-19.mat --checkpoint-iterations 100 --style-weight 1e10

Windows:
python style.py --checkpoint-dir ckpts --style examples\style\la_muse.jpg --train-path data\train2014 --test examples\content\chicago.jpg --test-dir test --vgg-path data\imagenet-vgg-verydeep-19.mat --checkpoint-iterations 100 --style-weight 1e10
'''

CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2

LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_ITERATIONS = 2000
VGG_PATH = 'data/imagenet-vgg-verydeep-19.mat'
TRAIN_PATH = 'data/train2014'
BATCH_SIZE = 4
DEVICE = '/gpu:0'
FRAC_GPU = 1

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str,
                        dest='checkpoint_dir', help='dir to save checkpoint in',
                        metavar='CHECKPOINT_DIR', required=True)

    parser.add_argument('--style', type=str,
                        dest='style', help='style image path',
                        metavar='STYLE', required=True)

    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH', default=TRAIN_PATH)

    parser.add_argument('--test', type=str,
                        dest='test', help='test image path',
                        metavar='TEST', default=False)

    parser.add_argument('--test-dir', type=str,
                        dest='test_dir', help='test image save dir',
                        metavar='TEST_DIR', default=False)

    parser.add_argument('--slow', dest='slow', action='store_true',
                        help='gatys\' approach (for debugging, not supported)',
                        default=False)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency',
                        metavar='CHECKPOINT_ITERATIONS',
                        default=CHECKPOINT_ITERATIONS)

    parser.add_argument('--vgg-path', type=str,
                        dest='vgg_path',
                        help='path to VGG19 network (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)

    parser.add_argument('--content-weight', type=float,
                        dest='content_weight',
                        help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    
    parser.add_argument('--style-weight', type=float,
                        dest='style_weight',
                        help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)

    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)
    
    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)


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
    exists(opts.checkpoint_dir, "checkpoint dir not found!")
    exists(opts.style, "style path not found!")
    exists(opts.train_path, "train path not found!")
    if opts.test or opts.test_dir:
        exists(opts.test, "test img not found!")
        exists(opts.test_dir, "test directory not found!")
    exists(opts.vgg_path, "vgg network data not found!")
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.checkpoint_iterations > 0
    assert os.path.exists(opts.vgg_path)
    assert opts.content_weight >= 0
    assert opts.style_weight >= 0
    assert opts.tv_weight >= 0
    assert opts.learning_rate >= 0

    # more cli params
    #print ("data-format:{}".format(opts.data_format))
    #print ("num-base-channels:{}".format(opts.num_base_channels))

def _get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir,x) for x in files]

    
def main():
    parser = build_parser()
    options = parser.parse_args()
    check_opts(options)

    style_target = get_img(options.style)


    if not options.slow:
        content_targets = _get_files(options.train_path)
    elif options.test:
        content_targets = [options.test]

    # a simpler folder name format
    #   [style image name]_[data format]_[num base channels]_[content weight]_[style weight]_[learning rate]
    style_image_str = os.path.basename(options.style)
    style_image_str = os.path.splitext(style_image_str)[0]
    content_weight_str = np.format_float_scientific(options.content_weight).replace('.','').replace('+','')
    style_weight_str = np.format_float_scientific(options.style_weight).replace('.','').replace('+','')

    folder_name = style_image_str + "-" + options.data_format + "_nbc" + str(options.num_base_channels
                                    ) + "_bs" + str(options.batch_size) + "_" + content_weight_str + "_" + style_weight_str + "_" + str(options.learning_rate)

    # create checkpoint and test image folders
    ckpt_dir_path = os.path.join(options.checkpoint_dir, folder_name)
    test_dir_path = os.path.join(options.test_dir, folder_name)

    print ("ckpt_dir_path:{}".format(ckpt_dir_path))
    print ("test_dir_path:{}".format(test_dir_path))

    if not os.path.isdir(ckpt_dir_path):
        os.mkdir (ckpt_dir_path)
    if not os.path.isdir(test_dir_path):
        os.mkdir (test_dir_path)
    
    options.checkpoint_dir = ckpt_dir_path
    options.test_dir = test_dir_path

    kwargs = {
        "slow":options.slow,
        "epochs":options.epochs,
        "print_iterations":options.checkpoint_iterations,
        "batch_size":options.batch_size,
        "save_path":os.path.join(options.checkpoint_dir,'fns.ckpt'),
        #"save_path":os.path.join(options.checkpoint_dir,folder_name,'fns.ckpt'),
        "learning_rate":options.learning_rate,

        # more cli params
        "data_format":options.data_format,
        "num_base_channels":options.num_base_channels
    }

    if options.slow:
        if options.epochs < 10:
            kwargs['epochs'] = 1000
        if options.learning_rate < 1:
            kwargs['learning_rate'] = 1e1

    args = [
        content_targets,
        style_target,
        options.content_weight,
        options.style_weight,
        options.tv_weight,
        options.vgg_path,

        # more cli params
        #options.data_format,
        #options.num_base_channels
    ]

    #print ("kwargs:{}".format(kwargs))
    #print ("args:{}".format(*args))

    for preds, losses, i, epoch in optimize(*args, **kwargs):
        style_loss, content_loss, tv_loss, loss = losses

        print('Epoch %d, Iteration: %d, Loss: %s' % (epoch, i, loss))
        to_print = (style_loss, content_loss, tv_loss)
        print('style: %s, content:%s, tv: %s' % to_print)
        if options.test:
            assert options.test_dir != False
            preds_path = '%s/%s_%s.png' % (options.test_dir,epoch,i)
            if not options.slow:
                ckpt_dir = os.path.dirname(options.checkpoint_dir)
                evaluate.ffwd_to_img(options.test, preds_path,
                                     options.checkpoint_dir,
                                     device='/gpu:0', #mcky, force to use GPU, or would have 'Conv2DCustomBackpropInputOp only supports NHWC.' error.
                                     data_format=options.data_format, num_base_channels=options.num_base_channels
                                     )
            else:
                save_img(preds_path, img)
    
    ckpt_dir = options.checkpoint_dir
    cmd_text = 'python evaluate.py --checkpoint %s ...' % ckpt_dir
    print("Training complete. For evaluation:\n    `%s`" % cmd_text)

if __name__ == '__main__':
    main()
