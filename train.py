import logging
import shutil
import configparser
import sys, os
from time import time
from datetime import datetime

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

from utils import data_preprocess
from model import *
from mydataset import ListDataset
from utils.common import progress_bar, AverageMeter

def train_epoch(train_loader, model, criterion, optimizer, epoch, use_cuda):
    losses = AverageMeter()
    model.train(True)
    torch.set_grad_enabled(True)
    for i, data in enumerate(train_loader):
        xi, xv, y = data[0], data[1], data[2]
        if use_cuda:
            xi, xv, y = xi.cuda(), xv.cuda(), y.cuda()
        optimizer.zero_grad()
        outputs = model(xi, xv)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), y.shape[0])
        
        progress_bar(i, len(train_loader), 'batch {}, train loss {:.5f}'.format(i, losses.avg))
    logging.info('Epoch: [{0}]\t Loss {loss.avg:.4f}\t'.format(
                  epoch, loss=losses))

def eval_model(val_loader, model, criterion, eval_metric, epoch, use_cuda):
    losses = AverageMeter()
    y_pred = []
    y_label = []
    model.train(False)
    torch.set_grad_enabled(False)
    for i, data in enumerate(val_loader):
        xi, xv, y = data[0], data[1], data[2]
        if use_cuda:
            xi, xv, y = xi.cuda(), xv.cuda(), y.cuda()
        outputs = model(xi, xv)
        loss = criterion(outputs, y)
        pred = torch.sigmoid(outputs).cpu()
        y_pred.extend(pred.data.numpy())
        y_label.extend(y.data.numpy())
        losses.update(loss.item(), y.shape[0])
    total_metric = eval_metric(y_label, y_pred)
    return losses.avg, total_metric

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def train_model():
    configFilePath = sys.argv[1]
    print(configFilePath)
    config = configparser.ConfigParser()
    config.read(configFilePath)    
    #save dir
    save_path = config.get('sys', 'save_dir')
    save_path=os.path.join(save_path, datetime.now().strftime('%Y%m%d_%H'))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        shutil.copy(configFilePath, save_path)
    
    # log setting
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    log_path='%s/model.log'%(save_path)
    ch = logging.FileHandler(log_path)
    ch.setLevel(logging.INFO)
    #logging.basicConfig(filename='%s.log'%(args.model_path), level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)
    # settings
    use_cuda = config.getboolean('sys', 'use_cuda')

    #data loader
    category_emb = config.get('data', 'category_emb')
    train_list = config.get('data', 'train_list')
    val_list = config.get('data', 'val_list')
    field_size = config.getint('data', 'field_size')
    workers = config.getint('data', 'workers')   

    batch_size = config.getint('optimizer', 'batch_size')
    train_dataset = ListDataset(train_list, category_emb, field_size)
    val_dataset = ListDataset(val_list, category_emb, field_size)    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    #model and optimizer
    arch = config.get('model', 'arch')
    if arch=='afm':
        embedding_size = config.getint('model', 'embedding_size')
        use_fm = config.getboolean('model','use_fm')
        use_ffm = config.getboolean('model','use_ffm')
        attention_layers_activation = config.get('model','attention_layers_activation')
        is_batch_norm = config.getboolean('model','is_batch_norm')
        dropout_attention = config.getfloat('model','dropout_attention')
        compression = config.getint('model', 'compression')
        net = AFM(field_size, train_dataset.feature_size, 
                embedding_size=embedding_size, 
                use_fm=use_fm, 
                use_ffm=use_ffm, 
                is_batch_norm=is_batch_norm,
                dropout_attention=dropout_attention,
                attention_layers_activation=attention_layers_activation,
                compression=compression)
    elif arch=='nfm':
        embedding_size = config.getint('model', 'embedding_size')
        use_fm = config.getboolean('model','use_fm')
        use_ffm = config.getboolean('model','use_ffm')
        interation_type = config.getboolean('model','interation_type')
        dropout_shallow = config.getfloat('model','dropout_shallow')
        net = NFM(field_size, train_dataset.feature_size, 
                embedding_size=embedding_size, 
                use_fm=use_fm, 
                use_ffm=use_ffm, 
                dropout_shallow=dropout_shallow,
                interation_type=interation_type)
    elif arch=='deepfm':
        embedding_size = config.getint('model', 'embedding_size')
        use_fm = config.getboolean('model','use_fm')
        use_ffm = config.getboolean('model','use_ffm')
        net = DeepFM(field_size, train_dataset.feature_size, 
                embedding_size=embedding_size, 
                use_fm=use_fm, 
                use_ffm=use_ffm, 
                is_shallow_dropout=False)

    if use_cuda:
        net = net.cuda()
    
    #optimizer
    optimizer_type = config.get('optimizer', 'optimizer_type')
    learning_rate = config.getfloat('optimizer', 'lr')
    weight_decay = config.getfloat('optimizer', 'weight_decay')
    momentum = config.getfloat('optimizer', 'momentum')    
    Parameters = filter(lambda p: p.requires_grad, net.parameters())
    if optimizer_type =='sgd':
        optimizer = torch.optim.SGD(Parameters, lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(Parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'rmsp':
        optimizer = torch.optim.RMSprop(Parameters, lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_type == 'adag':
        optimizer = torch.optim.Adagrad(Parameters, lr=learning_rate, weight_decay=weight_decay)

    decay_epochs = config.getint('optimizer', 'decay_epochs')
    lr_adjustor = CosineAnnealingLR(optimizer, decay_epochs, eta_min=0.00005, last_epoch=-1)

    #criterion
    criterion = F.binary_cross_entropy_with_logits

    #train interation
    n_epochs = config.getint('optimizer', 'epochs')
    eval_metric = config.get('optimizer', 'eval_metric')
    if eval_metric == 'roc_auc_score':
        metric = roc_auc_score

    for epoch in range(n_epochs):
        epoch_begin_time = time()
        train_epoch(train_loader, net, criterion, optimizer, epoch, use_cuda)
        if save_path:
            checkpoint_path=os.path.join(save_path, '%03d.ckpt' % epoch)
            save_checkpoint({
                'epoch': epoch,
                'arch': arch,
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, checkpoint_path)
        lr_adjustor.step()
        with torch.no_grad():
            valid_loss, valid_eval = eval_model(val_loader, net, criterion, metric, epoch, use_cuda)        
        logging.info('Epoch: [{0}]\t valid_Loss {loss:.4f}\t valid_metric {valid_metric:.4f}'.format(
                  epoch, loss=valid_loss, valid_metric=valid_eval))
        print('[%d] loss: %.6f metric: %.6f time: %.1f s' %
                    (epoch + 1, valid_loss, valid_eval, time()-epoch_begin_time))      

if __name__ == '__main__':
    train_model()
    print('finishing training')                    