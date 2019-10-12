
'''
@author: edison huang 
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import logging
import numpy as np
import functools
import argparse

import mxnet as mx
from mxnet import ndarray as nd
import mxnet.optimizer as optimizer

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import flops_counter

sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
import verification

from config import config, default, generate_config
from image_iter import FaceImageIter
from face_symbol_factory import get_symbol_embedding, get_symbol_arcface

def parse_args():
    parser = argparse.ArgumentParser(description='Train parall face network')
    # general
    parser.add_argument('--dataset', default=default.dataset, help='dataset config')
    parser.add_argument('--network', default=default.network, help='network config')
    parser.add_argument('--loss', default=default.loss, help='loss config')
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset, args.loss)
    parser.add_argument('--models-root', default=default.models_root, help='root directory to save model.')
    parser.add_argument('--pretrained', default=default.pretrained, help='pretrained model to load')
    parser.add_argument('--pretrained-epoch', type=int, default=default.pretrained_epoch, help='pretrained epoch to load')
    parser.add_argument('--ckpt', type=int, default=default.ckpt, help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
    parser.add_argument('--verbose', type=int, default=default.verbose, help='do verification testing and model saving every verbose batches')
    parser.add_argument('--lr', type=float, default=default.lr, help='start learning rate')
    parser.add_argument('--lr-steps', type=str, default=default.lr_steps, help='steps of lr changing')
    parser.add_argument('--wd', type=float, default=default.wd, help='weight decay')
    parser.add_argument('--mom', type=float, default=default.mom, help='momentum')
    parser.add_argument('--frequent', type=int, default=default.frequent, help='')
    parser.add_argument('--per-batch-size', type=int, default=default.per_batch_size, help='batch size in each context')
    parser.add_argument('--kvstore', type=str, default=default.kvstore, help='kvstore setting')
    parser.add_argument('--worker-id', type=int, default=0, help='worker id for dist training, starts from 0')
    parser.add_argument('--extra-model-name', type=str, default='', help='extra model name')
    parser.add_argument('--gpu-num4cls', type=int, default=0, help='gpu used by only fc weight')

  
    args = parser.parse_args()
    return args

def get_data_iter(config, batch_size):
    data_dir = config.dataset_path
    path_imgrec = None
    path_imglist = None
    image_size = config.image_shape[0:2]

    assert len(image_size)==2
    assert image_size[0]==image_size[1]
    print('image_size', image_size)
    print('num_classes', config.num_classes)

    path_imgrec = os.path.join(data_dir, "train.rec")
    data_shape = (config.image_shape[2], image_size[0], image_size[1])

    val_dataiter = None

    train_dataiter = FaceImageIter(
        batch_size           = batch_size,
        data_shape           = data_shape,
        path_imgrec          = path_imgrec,
        shuffle              = True,
        rand_mirror          = config.data_rand_mirror,
        mean                 = None,
        cutoff               = config.data_cutoff,
        color_jittering      = config.data_color,
        images_filter        = config.data_images_filter,
    )
    train_dataiter = mx.io.PrefetchingIter(train_dataiter)
    return train_dataiter, val_dataiter

def train_net(args):
    ## =================== parse context ==========================
    ctx_list = []
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    if len(cvd)>0:
        for i in range(len(cvd.split(','))):
            ctx_list.append(mx.gpu(i))
    if len(ctx_list)==0:
        ctx_list = [mx.cpu()]
        print('use cpu')
    else:
        print('gpu num:', len(ctx_list))


    ## ==================== get model save prefix and log ============
    if len(args.extra_model_name)==0:
        prefix = os.path.join(args.models_root, '%s-%s-%s'%(args.network, args.loss, args.dataset), 'model')
    else:
        prefix = os.path.join(args.models_root, '%s-%s-%s-%s'%(args.network, args.loss, args.dataset, args.extra_model_name), 'model')

    prefix_dir = os.path.dirname(prefix)
    print('prefix', prefix)
    if not os.path.exists(prefix_dir):
      os.makedirs(prefix_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    filehandler = logging.FileHandler("{}.log".format(prefix))
    streamhandler = logging.StreamHandler()
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    ## ================ parse batch size and class info ======================
    if args.gpu_num4cls > 0: # model weight split num
        global_num_ctx4cls = args.gpu_num4cls
        args.backbone_ctx_num = len(ctx_list) - args.gpu_num4cls
        assert args.backbone_ctx_num > 1
        ctx_list4backbone = ctx_list[:args.backbone_ctx_num]
        ctx_list4cls = ctx_list[-1*args.gpu_num4cls:]
    else: 
        args.backbone_ctx_num = len(ctx_list)
        global_num_ctx4cls = config.num_workers * args.backbone_ctx_num
        ctx_list4backbone = ctx_list
        ctx_list4cls = ctx_list

    if args.per_batch_size==0:
        args.per_batch_size = 128

    args.batch_size = args.per_batch_size*args.backbone_ctx_num

    
    if config.num_classes % global_num_ctx4cls == 0:
        args.ctx_num_classes = config.num_classes//global_num_ctx4cls
    else:
        args.ctx_num_classes = config.num_classes//global_num_ctx4cls+1

    args.local_class_start = 0

    logger.info("Train model with argument: {}".format(args, config))

    train_dataiter, val_dataiter = get_data_iter(config, args.batch_size)

    ## =============== get train info ============================
    image_size = config.image_shape[0:2]
    config.per_batch_size = args.per_batch_size

    if len(args.pretrained) == 0: # train from scratch 
        esym = get_symbol_embedding(config)
        asym = functools.partial(get_symbol_arcface, config=config)
    else: # load train model to continue
        assert False

    if config.count_flops:
        all_layers = esym.get_internals()
        _sym = all_layers['fc1_output']
        FLOPs = flops_counter.count_flops(_sym, data=(1,3,image_size[0],image_size[1]))
        _str = flops_counter.flops_str(FLOPs)
        print('Network FLOPs: %s'%_str)
        logging.info("Network FLOPs : %s" % _str)

    if config.num_workers == 1:
        from parall_loss_module import ParallLossModule
        default_ctx = ctx_list4cls[-1]
        model_parall_loss = ParallLossModule(
                context=default_ctx, batch_size=args.batch_size, 
                ctx_num_class=args.ctx_num_classes, 
                local_class_start=args.local_class_start, logger=logger)

        for i, ctx in enumerate(ctx_list4cls):
            args._ctxid = i
            part_w_sym = asym(args)
            part_weight_module = mx.mod.Module(part_w_sym, context=ctx)

            model_parall_loss.add(part_weight_module, ctx)

    else: # distribute parall loop
        assert False

    model = mx.mod.SequentialModule()
    embedding_feat_mod = mx.mod.Module(esym, 
                            context=ctx_list4backbone, label_names=None)
    model.add(embedding_feat_mod)
    model.add(model_parall_loss, auto_wiring=True, take_labels=True)


    ## ============ get optimizer =====================================
    if config.net_name=='fresnet' or config.net_name=='fmobilefacenet':
        initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    else:
        initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)

    opt = optimizer.SGD(learning_rate=args.lr, momentum=args.mom, wd=args.wd, rescale_grad=1.0/args.batch_size)
    _cb = mx.callback.Speedometer(args.batch_size, args.frequent)

    ver_list = []
    ver_name_list = []
    for name in config.val_targets:
        path = os.path.join(config.dataset_path, name+".bin")
        if os.path.exists(path):
            data_set = verification.load_bin(path, image_size)
            ver_list.append(data_set)
            ver_name_list.append(name)
            print('ver', name)

    def ver_test(nbatch, feat_model):
        results = []
        for i in range(len(ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], feat_model, args.batch_size, 10, None, None)
            print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
            #print('[%s][%d]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc1, std1))
            print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
            results.append(acc2)
        return results

    highest_acc = [0.0, 0.0]  #lfw and target
    

    global_step = [0]
    save_step = [0]
    lr_steps = [int(x) for x in args.lr_steps.split(',')]
    print('lr_steps', lr_steps)

    ## =============== batch end callback definition ===================================
    def _batch_callback(param):
        #global global_step
        global_step[0]+=1
        mbatch = global_step[0]
        for step in lr_steps:
            if mbatch==step:
                opt.lr *= 0.1
                print('lr change to', opt.lr)
                logger.info('lr change to', opt.lr)
                break

        _cb(param)
        if mbatch%1000==0:
            print('lr-batch-epoch:',opt.lr,param.nbatch,param.epoch)
            logger.info('lr-batch-epoch: {}'.format(opt.lr,param.nbatch,param.epoch))

        if mbatch>=0 and mbatch%args.verbose==0:
            acc_list = ver_test(mbatch, embedding_feat_mod)
            save_step[0]+=1
            msave = save_step[0]
            do_save = False
            is_highest = False
            if len(acc_list)>0:
                #lfw_score = acc_list[0]
                #if lfw_score>highest_acc[0]:
                #  highest_acc[0] = lfw_score
                #  if lfw_score>=0.998:
                #    do_save = True
                score = sum(acc_list)
                if acc_list[-1]>=highest_acc[-1]:
                    if acc_list[-1]>highest_acc[-1]:
                        is_highest = True
                    else:
                        if score>=highest_acc[0]:
                            is_highest = True
                            highest_acc[0] = score
                    highest_acc[-1] = acc_list[-1]
                    
            if is_highest:
                do_save = True
            if args.ckpt==0:
                do_save = False
            elif args.ckpt==2:
                do_save = True
            elif args.ckpt==3:
                msave = 1

            if do_save:
                print('saving', msave)
                logger.info('saving {}'.format(msave))

                arg, aux = embedding_feat_mod.get_params()
                all_layers = embedding_feat_mod.symbol.get_internals()
                _sym = all_layers['fc1_output']
                mx.model.save_checkpoint(prefix, msave, _sym, arg, aux)
            print('[%d]Accuracy-Highest: %1.5f'%(mbatch, highest_acc[-1]))
            logger.info('[%d]Accuracy-Highest: %1.5f'%(mbatch, highest_acc[-1]))

        if config.max_steps>0 and mbatch>config.max_steps:
            sys.exit(0)

    model.fit(train_dataiter,
        begin_epoch        = 0,
        num_epoch          = 999999,
        eval_data          = val_dataiter,
        kvstore            = args.kvstore,
        optimizer          = opt,
        initializer        = initializer,
        arg_params         = None,
        aux_params         = None,
        allow_missing      = True,
        batch_end_callback = _batch_callback)

def main():
    args = parse_args()
    train_net(args)

if __name__ == '__main__':
    main()

