import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import logging
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import torch.nn as nn


def test_single(step,
                dataset_test,
                name,
                n_share,
                G,
                C,
                output_file,
                open_class=None,
                open=False,
                entropy=False,
                thr=None,
                num_class=15):
    G.eval()
    C.eval()
    ## Known Score Calculation.
    correct = 0
    correct_close = 0
    size = 0
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    class_list = [i for i in range(n_share)]
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t = data[0].cuda(), data[1].cuda()
            feat = G(img_t)
            out_t = C(feat)
            if batch_idx == 0:
                #open_class = int(out_t.size(1))
                class_list.append(num_class)
            pred = out_t[:, :num_class].data.max(1)[1]
            correct_close += pred.eq(label_t.data).cpu().sum()
            
            pred = out_t.data.max(1)[1]
            correct += pred.eq(label_t.data).cpu().sum()
            pred = pred.cpu().numpy()
            k = label_t.data.size()[0]
            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_num[i] += float(len(t_ind[0]))
            size += k
            pred_unk = nn.Softmax(dim=-1)(out_t)[:, -1]
            out_t_ = nn.Softmax(dim=-1)(out_t[:, :-1])
            entr = -torch.sum(out_t_ * torch.log(out_t_), 1).data.cpu().numpy()
            if open:
                label_t = label_t.data.cpu().numpy()
                if batch_idx == 0:
                    label_all = label_t
                    pred_open = pred_unk.data.cpu().numpy()
                    pred_all = out_t_.data.cpu().numpy()
                    pred_ent = entr
                else:
                    pred_open = np.r_[pred_open, pred_unk.data.cpu().numpy()]
                    pred_ent = np.r_[pred_ent, entr]
                    pred_all = np.r_[pred_all, out_t_.data.cpu().numpy()]
                    label_all = np.r_[label_all, label_t]
    
    if True:
        roc = 0.0
        roc_ent = 0.0
        best_th = 0.
        best_acc = 0.
        roc_softmax = 0.0
    #logger = logging.getLogger(__name__)
    logger = logging.getLogger(output_file)
    #logging.basicConfig(filename=name, format="%(message)s")
    logging.basicConfig(filename=output_file, format="%(message)s")
    logger.setLevel(logging.INFO)
    per_class_acc = per_class_correct / per_class_num
    acc_all = 100. * float(correct) / float(size)
    close_count = float(per_class_num[:len(class_list) - 1].sum())
    acc_close_all = 100. * float(correct_close) / close_count
    known_acc = per_class_acc[:len(class_list) - 1].mean()
    unknown = per_class_acc[-1]
    h_score = 2 * known_acc * unknown / (known_acc + unknown)
    output = [
        "step %s" % step,
        "closed perclass",
        list(per_class_acc),
        "acc per class %s" % (float(per_class_acc.mean())),
        "acc %s" % float(acc_all),
        "acc close all %s" % float(acc_close_all),
        "known acc %s" % float(known_acc),
        "unknown acc %s" % float(unknown),
        "h score %s" % float(h_score),
        "roc %s" % float(roc),
        "roc ent %s" % float(roc_ent),
        "roc softmax %s" % float(roc_softmax),
        "best hscore %s" % float(best_acc),
        #"best thr %s" % float(best_th)
    ]

    #out_file = open(osp.join(args.output_dir, 'logfile.txt'), 'w')
    #out_file.write(output + '\n')
    #out_file.flush()
    logger.info(output)
    #print(output)
    return acc_all, known_acc, unknown, h_score


def test_osdg31(step,
                dataset_test,
                name,
                n_share,
                G,
                C,
                output_file,
                open_class=None,
                open=False,
                entropy=False,
                thr=None):
    G.eval()
    C.eval()
    ## Known Score Calculation.
    correct = 0
    correct_close = 0
    size = 0
    per_class_num = np.zeros((10 + 1))
    per_class_correct = np.zeros((10 + 1)).astype(np.float32)
    class_list = [i for i in range(10)]
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t = data[0].cuda(), data[1].cuda()
            feat = G(img_t)
            out_t = C(feat)
            if batch_idx == 0:
                #open_class = int(out_t.size(1))
                class_list.append(10)
            pred = out_t[:, :10].data.max(1)[1]
            correct_close += pred.eq(label_t.data).cpu().sum()
            pred = out_t.data.max(1)[1]
            correct += pred.eq(label_t.data).cpu().sum()
            pred = pred.cpu().numpy()
            k = label_t.data.size()[0]
            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_num[i] += float(len(t_ind[0]))
            size += k

    if True:
        roc = 0.0
        roc_ent = 0.0
        best_th = 0.
        best_acc = 0.
        roc_softmax = 0.0
    #logger = logging.getLogger(__name__)
    logger = logging.getLogger(output_file)
    #logging.basicConfig(filename=name, format="%(message)s")
    logging.basicConfig(filename=output_file, format="%(message)s")
    logger.setLevel(logging.INFO)
    per_class_acc = per_class_correct / per_class_num
    acc_all = 100. * float(correct) / float(size)
    close_count = float(per_class_num[:len(class_list) - 1].sum())
    acc_close_all = 100. * float(correct_close) / close_count
    known_acc = per_class_acc[:len(class_list) - 1].mean()
    unknown = per_class_acc[-1]
    h_score = 2 * known_acc * unknown / (known_acc + unknown)
    output = [
        "step %s" % step,
        "closed perclass",
        list(per_class_acc),
        "acc per class %s" % (float(per_class_acc.mean())),
        "acc %s" % float(acc_all),
        "acc close all %s" % float(acc_close_all),
        "known acc %s" % float(known_acc),
        "unknown acc %s" % float(unknown),
        "h score %s" % float(h_score),
        #"roc %s" % float(roc),
        #"roc ent %s" % float(roc_ent),
        #"roc softmax %s" % float(roc_softmax),
        "best hscore %s" % float(best_acc),
        #"best thr %s" % float(best_th)
    ]

    #out_file = open(osp.join(args.output_dir, 'logfile.txt'), 'w')
    #out_file.write(output + '\n')
    #out_file.flush()
    '''with open(output_file,'a+') as f:
        f.write(output + '\n')
        f.flush()'''
    logger.info(output)
    print(output)
    return acc_all, known_acc, unknown, h_score