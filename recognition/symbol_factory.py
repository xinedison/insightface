import sys
import os
import mxnet as mx
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbol'))
import fresnet
import fmobilefacenet
import fmobilenet
import fmnasnet
import fdensenet

def get_symbol_embedding(config):
    embedding = eval(config.net_name).get_symbol()
    all_label = mx.symbol.Variable('softmax_label')
    #embedding = mx.symbol.BlockGrad(embedding)
    all_label = mx.symbol.BlockGrad(all_label)
    out_list = [embedding, all_label]
    out = mx.symbol.Group(out_list)
    return out

def get_symbol_arcface(args, config):
    embedding = mx.symbol.Variable('data')
    all_label = mx.symbol.Variable('softmax_label')
    gt_label = all_label
    is_softmax = True
    #print('call get_sym_arcface with', args, config)
    _weight = mx.symbol.Variable("fc7_%d_weight" % args._ctxid, shape=(args.ctx_num_classes, config.emb_size), 
        lr_mult=config.fc7_lr_mult, wd_mult=config.fc7_wd_mult)
  
    if config.loss_name=='softmax': #softmax
        fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, no_bias = True, num_hidden=args.ctx_num_classes, name='fc7_%d'%args._ctxid)
        assert False
  
    elif config.loss_name=='margin_softmax':
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n_%d'%args._ctxid)
        fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=args.ctx_num_classes, name='fc7_%d'%args._ctxid)
        if config.loss_m1 != 1.0 or config.loss_m2 != 0.0 or config.loss_m3 != 0.0:
            gt_one_hot = mx.sym.one_hot(gt_label, depth = args.ctx_num_classes, on_value = 1.0, off_value = 0.0)
            if config.loss_m1==1.0 and config.loss_m2==0.0:
                _one_hot = gt_one_hot*args.margin_b
                fc7 = fc7-_one_hot
            else:
                fc7_onehot = fc7 * gt_one_hot
                cos_t = fc7_onehot
                t = mx.sym.arccos(cos_t)

                if config.loss_m1 != 1.0:
                    t = t*config.loss_m1

                if config.loss_m2 != 0.0:
                    t = t+config.loss_m2

                margin_cos = mx.sym.cos(t)
                if config.loss_m3 != 0.0:
                    margin_cos = margin_cos - config.loss_m3
                margin_fc7 = margin_cos
                margin_fc7_onehot = margin_fc7 * gt_one_hot
                diff = margin_fc7_onehot - fc7_onehot
                fc7 = fc7+diff
        fc7 = fc7*config.loss_s
  
    out_list = []
    out_list.append(fc7)
    if config.loss_name=='softmax': #softmax
        assert False
        out_list.append(gt_label)
    out = mx.symbol.Group(out_list)
    return out


