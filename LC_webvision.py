from __future__ import print_function
import time
from sklearn.decomposition import PCA
from InceptionResNetV2 import *
import torchvision.models as models

from torch.cuda.amp import autocast, GradScaler
from SCE import SCELoss
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
from cg import one_iter_true_wm
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
import dataloader_wm_strong as dataloader
import mkl
mkl.get_max_threads()

parser = argparse.ArgumentParser(description='PyTorch WebVision Training')
parser.add_argument('--batch_size', default=32,
                    type=int, help='train batchsize')
parser.add_argument('--warm_up', default=1, type=int, help='warm epochs')
parser.add_argument('--lr', '--learning_rate', default=0.01,
                    type=float, help='initial learning rate')
parser.add_argument('--num_class', default=50, type=int)
parser.add_argument('--epochs', default=50000000, type=int, help='epochs')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--save_name', type=str, default='test')
parser.add_argument('--mode', type=str, default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--data_path', default='../data/webvision/',
                    type=str, help='path to dataset')
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--resume_epoch', default=0, type=int, help='epochs')

parser.add_argument('--knn', default=50, type=int, help=' ')
parser.add_argument('--T', default=1, type=float,
                    help='temperature for sharping pseudo-labels')
parser.add_argument('--pca', default=128, type=int, help='PCA dimension')

args = parser.parse_args()

cudnn.benchmark=True
torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

class prettyfloat(float):
    def __repr__(self):
        return "%0.2f" % self

def save_model(save_name, save_path, net1, net2, optimizer1, optimizer2):
    save_filename = os.path.join(save_path, save_name)
    torch.save({'net1': net1.state_dict(),
        'net2': net2.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'optimizer2': optimizer2.state_dict(),
                }, save_filename)
    print(f"model saved: {save_filename}")

def entropy(p):
    return - torch.sum(p * torch.log(p), axis=-1)

class SoftCELoss(object):
    def __call__(self, outputs, targets):
        probs = torch.softmax(outputs, dim=1)
        Lx = - \
            torch.mean(torch.sum(F.log_softmax(
                outputs, dim=1) * targets, dim=1))
        return Lx

def load_model(load_path, net1, net2, optimizer1, optimizer2):
    checkpoint = torch.load(load_path)
    for key in checkpoint.keys():
        if 'net1' in key:
            net1.load_state_dict(checkpoint[key])
        elif 'net2' in key:
            net2.load_state_dict(checkpoint[key])
        elif key == 'optimizer1':
            optimizer1.load_state_dict(checkpoint[key])
        elif key == 'optimizer2':
            optimizer2.load_state_dict(checkpoint[key])
        print(f"Check Point Loading: {key} is LOADED")

def train(epoch, net1, net2, optimizer, labeled_trainloader, mode='PL', num_iter=400):
    net1.train()
    net2.eval()
    hard_label = False

    gmm = GaussianMixture(n_components=2, max_iter=10,
                          tol=1e-2, reg_covar=5e-4, warm_start=True)

    scaler = GradScaler()

    labeled_train_iter = iter(labeled_trainloader)
    I = 1

    for batch_idx in range(num_iter):
        try:
            inputs, labels_x, F = next(labeled_train_iter)
            inputs_w, inputs_s = inputs
        except StopIteration:
            if I == 1:
                stats_log.write(f'\n')
                stats_log.flush()
                return
            else:
                labeled_train_iter = iter(labeled_trainloader)
                inputs, labels_x, F = next(labeled_train_iter)
                inputs_w, inputs_s = inputs
                I = I-1

        F = F.cuda()

        batch_size = inputs_w.size(0)

        one_hot_x = torch.zeros(batch_size, args.num_class).scatter_(
            1, labels_x.view(-1, 1), 1)
        one_hot_x = one_hot_x.cuda()

        inputs_w, inputs_s, labels_x = inputs_w.cuda(), inputs_s.cuda(), labels_x.cuda()

        with torch.no_grad():

            outputs_1_w = net1(inputs_w)
            outputs_2_w = net2(inputs_w)

        with autocast():
            outputs_s = net1(inputs_s)

            if hard_label:
                pass

            else:

                w_x = F.view(-1, 1)

                probs = torch.softmax(outputs_1_w, dim=-1) + \
                                      torch.softmax(outputs_2_w, dim=-1)
                probs = probs/2

                probs_T = probs**args.T
                probs_T = probs_T / probs_T.sum(dim=1, keepdim=True)
                probs_T = probs_T.detach()
                probs_T_x, probs_T_u = probs_T[:batch_size], probs_T[batch_size:]
                targets_x = one_hot_x*w_x+(1-w_x)*probs_T_x

                Lx = softCE(outputs_s, targets_x)

            prior = torch.ones(args.num_class)/args.num_class
            prior = prior.cuda()

            pred_mean = torch.softmax(outputs_s, dim=1).mean(0)
            penalty = torch.sum(prior*torch.log(prior/pred_mean))

            loss = Lx+penalty

        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def warmup(epoch, net, eval_models, ema, optimizer, dataloader, updataema=False):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)
        penalty = conf_penalty(outputs)
        L = loss + penalty
        L.backward()
        optimizer.step()

@torch.no_grad()
def test(epoch, net1, net2, test_loader):
    acc_meter.reset()
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)
            acc_meter.add(outputs, targets)
    accs = acc_meter.value()
    print(accs)
    acc = accs[0]
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))
    stats_log.write('Epoch:%d   Accuracy:%.2f\n' % (epoch, acc))
    stats_log.flush()
    return accs

@ torch.no_grad()
def eval_train(model, all_loss, _, gmm):
    model.eval()
    features=[]
    L=[]
    num_iter=(len(eval_loader.dataset)//eval_loader.batch_size)+1
    losses=torch.zeros(len(eval_loader.dataset))
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets=inputs.cuda(), targets.cuda()
            L.append(targets)

            f, outputs=model(inputs, feat=True)
            features.append(f)

            loss=CE(outputs, targets)
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]
            sys.stdout.write('\r')
            sys.stdout.write(
                '| Evaluating loss Iter[%3d/%3d]\t' % (batch_idx, num_iter))
            sys.stdout.flush()

    features=torch.cat(features, dim=0).detach().cpu().numpy()
    L=torch.cat(L, dim=0).detach().cpu().numpy()
    L=np.eye(args.num_class)[L]

    pca=PCA(n_components=args.pca)
    F=one_iter_true_wm(pca.fit_transform(features), L,
                       gpuid=args.gpuid, k=args.knn)
    return F, None

class NegEntropy(object):
    def __call__(self, outputs):
        probs=torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model=InceptionResNetV2(num_classes=args.num_class)
    model=model.cuda()
    return model

stats_log=open('./checkpoint/%s_WV' % (args.save_name)+'_stats.txt', 'w')
loader=dataloader.webvision_dataloader(
    batch_size=args.batch_size, num_class=args.num_class, num_workers=5, root=args.data_path, log=stats_log)
print('| Building net')
acc_meter=torchnet.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)

net1=create_model()
net2=create_model()
test_loader=loader.run('test')
accs=test(-1, net1, net2, test_loader)

optimizer1=optim.SGD(net1.parameters(), lr=args.lr,
                     momentum=0.9, weight_decay=5e-4)
optimizer2=optim.SGD(net2.parameters(), lr=args.lr,
                     momentum=0.9, weight_decay=5e-4)

CE=nn.CrossEntropyLoss(reduction='none')
CEloss=nn.CrossEntropyLoss()

softCE=SoftCELoss()
conf_penalty=NegEntropy()
best_acc=0
all_loss_1=[]
all_loss_2=[]

if len(args.resume) != 0:
    load_model(args.resume, net1, net2, optimizer1, optimizer2)
    test_loader=loader.run('test')
    acc=test(-1, net1, net2, test_loader)
    exit()

for epoch in range(args.resume_epoch, args.warm_up):
    test_loader=loader.run('test')
    for param_group in optimizer1.param_groups:
        param_group['lr']=0.01
    for param_group in optimizer2.param_groups:
        param_group['lr']=0.01
    warmup_trainloader=loader.run('warmup')
    print('Warmup Net1')
    warmup(epoch, net1, None, None, optimizer1, warmup_trainloader)
    print('Warmup Net2')
    warmup(epoch, net2, None, None, optimizer2, warmup_trainloader)

save_model(f"{args.save_name}_WV_latest",
           "./checkpoint/", net1, net2, optimizer1, optimizer2)

gmm4loss=GaussianMixture(n_components=2, max_iter=10,
                         tol=1e-2, reg_covar=1e-3, warm_start=False)

for epoch in range(args.resume_epoch, args.num_epochs+1):
    start_time=time.time()
    lr=args.lr
    if args.num_epochs-epoch < 100:
        lr /= 10
    for param_group in optimizer1.param_groups:
        param_group['lr']=lr
    for param_group in optimizer2.param_groups:
        param_group['lr']=lr

    eval_loader=loader.run('eval_train')

    imagenet_valloader=loader.run('imagenet')

    prob1, _=eval_train(net1, all_loss_1, None, gmm4loss)

    eval_loader=loader.run('eval_train')
    prob2, _=eval_train(net2, all_loss_2, None, gmm4loss)

    pred=np.ones(65944)

    print('Train Net')
    labeled_trainloader=loader.run('train', pred, prob2)
    train(epoch, net1, net2, optimizer1,
          labeled_trainloader, num_iter=args.epochs)

    print('\nTrain Net2')
    labeled_trainloader=loader.run('train', pred, prob1)
    train(epoch, net2, net1, optimizer2,
          labeled_trainloader, num_iter=args.epochs)
    imagenet_acc=test(epoch, net1, net2, imagenet_valloader)
    acc=imagenet_acc[0]
    if acc > best_acc:
        best_acc=acc
        save_model(f"{args.save_name}_WV_best", "./checkpoint/",
                   net1, net2, optimizer1, optimizer2)
    save_model(f"{args.save_name}_WV_latest", "./checkpoint/",
               net1, net2, optimizer1, optimizer2)
