from __future__ import print_function
import time
import copy
from torch.cuda.amp import autocast, GradScaler
from cg import one_iter_true
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
import dataloader_cifar_lc as dataloader
import mkl
mkl.get_max_threads()

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64,
                    type=int, help='train batchsize')
parser.add_argument('--warm_up', default=15, type=int, help='warm epochs')
parser.add_argument('--lr', '--learning_rate', default=0.02,
                    type=float, help='learning rate')
parser.add_argument('--num_epochs', default=400, type=int, help='number of trainig epochs')
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--seed', default=123)
parser.add_argument('--save_name', type=str, default='test')
parser.add_argument('--data_path', default='./cifar-10',
                    type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--resume', default='', type=str)

parser.add_argument('--T', default=1, type=float,
                    help='temperature for sharping pseudo-labels')
parser.add_argument('--knn', default=50, type=int,
                    help='knn number for constructing graph')
parser.add_argument('--pca', default=64, type=int, help='PCA dimension')

parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--noise_mode',  default='sym', help='choose symmetrical or asymmetrical noise')


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
		Lx = - torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * targets, dim=1))
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


def train(epoch, net1, net2, ema_model, optimizer, labeled_trainloader):
	net1.train()
	net2.eval()
	hard_label = False
	scaler = GradScaler()

	labeled_train_iter = iter(labeled_trainloader)
	I = 2

	while True:

		try:
			inputs, labels_x, F = next(labeled_train_iter)
			inputs_w, inputs_s = inputs
		except:
			if I == 1:
				stats_log.write(f'\n')
				stats_log.flush()
				return
			else:
				labeled_train_iter = iter(labeled_trainloader)
				inputs, labels_x, F = next(labeled_train_iter)
				inputs_w, inputs_s = inputs
				I = I-1

		F = F.cuda(args.gpuid)

		batch_size = inputs_w.size(0)

		one_hot_x = torch.zeros(batch_size, args.num_class).scatter_(
		    1, labels_x.view(-1, 1), 1)
		one_hot_x = one_hot_x.cuda(args.gpuid)

		inputs_w, inputs_s, labels_x = inputs_w.cuda(
		    args.gpuid), inputs_s.cuda(args.gpuid), labels_x.cuda(args.gpuid)

		with torch.no_grad():
			outputs_1_w = net1(inputs_w)
			outputs_2_w = net2(inputs_w)

		with autocast():

			outputs_s = net1(inputs_s)

			if hard_label:
				pass
			else:

				w_x = F.gather(1, labels_x.view(-1, 1))
				w_x = w_x.view(-1, 1)
				targets_x = (torch.softmax(outputs_1_w, dim=1) +
				             torch.softmax(outputs_2_w, dim=1))/2

				targets_x = targets_x**args.T
				targets_x = targets_x/targets_x.sum(dim=1, keepdim=True)
				targets_x = targets_x.detach()

				targets_x = one_hot_x*w_x+(1-w_x)*targets_x

				Lx = softCE(outputs_s, targets_x)

			prior = torch.ones(args.num_class)/args.num_class
			prior = prior.cuda(args.gpuid)

			pred_mean = torch.softmax(outputs_s, dim=1).mean(0)
			penalty = torch.sum(prior*torch.log(prior/pred_mean))

			loss = Lx+penalty

		optimizer.zero_grad()
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()


def warmup(epoch, net, ema_model, ema, optimizer, dataloader):
	net.train()
	num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
	for batch_idx, (inputs, labels, path) in enumerate(dataloader):
		inputs, labels = inputs.cuda(args.gpuid), labels.cuda(args.gpuid)
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = CEloss(outputs, labels)
		if args.noise_mode == 'asym':
			penalty = conf_penalty(outputs)
			L = loss + penalty
		elif args.noise_mode == 'sym':
			L = loss
		L.backward()
		optimizer.step()


@torch.no_grad()
def test(epoch, net1, net2):
	net1.eval()
	net2.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets, _) in enumerate(test_loader):
			inputs, targets = inputs.cuda(args.gpuid), targets.cuda(args.gpuid)
			outputs1 = net1(inputs)
			outputs2 = net2(inputs)
			outputs = outputs1+outputs2
			_, predicted = torch.max(outputs, 1)

			total += targets.size(0)
			correct += predicted.eq(targets).cpu().sum().item()
	acc = 100.*correct/total
	print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))
	stats_log.write('Epoch:%d	Accuracy:%.2f\n' % (epoch, acc))
	stats_log.flush()
	return acc


@ torch.no_grad()
def eval_train(epoch, model1, model2):
	model1.eval()

	model2.eval()

	features1=[]
	predictions1=[]

	features2=[]
	predictions2=[]

	with torch.no_grad():

		for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
			inputs, targets=inputs.cuda(args.gpuid), targets.cuda(args.gpuid)

			f1, outputs1=model1(inputs, feat=True)
			f2, outputs2=model2(inputs, feat=True)

			features1.append(f1)
			predictions1.append(torch.softmax(outputs1, dim=1))

			features2.append(f2)
			predictions2.append(torch.softmax(outputs2, dim=1))

	features1=torch.cat(features1, dim=0).detach().cpu().numpy()
	predictions1=torch.cat(predictions1, dim=0).detach().cpu().numpy()

	features2=torch.cat(features2, dim=0).detach().cpu().numpy()
	predictions2=torch.cat(predictions2, dim=0).detach().cpu().numpy()

	N=features1.shape[0]
	F1=one_iter_true(features1, L1, gpuid=args.gpuid,
	                 k=args.knn, classes=args.num_class)
	F2=one_iter_true(features2, L2, gpuid=args.gpuid,
	                 k=args.knn, classes=args.num_class)



	return F1, F2

@ torch.no_grad()
def get_pred(model1, model2):
	model1.eval()
	model2.eval()

	predictions1=[]

	predictions2=[]

	with torch.no_grad():
		for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
			inputs, targets=inputs.cuda(args.gpuid), targets.cuda(args.gpuid)
			outputs1=model1(inputs)
			outputs2=model2(inputs)

			predictions1.append(torch.softmax(outputs1, dim=1))

			predictions2.append(torch.softmax(outputs2, dim=1))

	predictions1=torch.cat(predictions1, dim=0).detach().cpu().numpy()

	predictions2=torch.cat(predictions2, dim=0).detach().cpu().numpy()

	return predictions1, predictions2, (predictions1+predictions2)/2


class NegEntropy(object):
	def __call__(self, outputs):
		probs=torch.softmax(outputs, dim=-1)
		return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
	model=ResNet18(num_classes=args.num_class)
	model=model.cuda(args.gpuid)
	return model

with open(f'{args.data_path}{args.r}_{args.noise_mode}.json') as f:
	 all_labels=np.asarray(eval(f.readlines()[0]))

	 one_hot_targets=np.eye(args.num_class)[all_labels]
	 L1=copy.deepcopy(one_hot_targets)
	 L2=copy.deepcopy(one_hot_targets)
	 del one_hot_targets

stats_log=open('./checkpoint/%s_%s_%.1f_%s' % (args.save_name,
               args.dataset, args.r, args.noise_mode)+'_stats.txt', 'w')

loader=dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size, num_workers=5,
	root_dir=args.data_path, log=stats_log, noise_file='%s/%.1f_%s.json' % (args.data_path, args.r, args.noise_mode))
print('| Building net')

net1=create_model()
net2=create_model()

ema_model1=None
ema_model2=None
test_loader=loader.run('test')
acc=test(-1, net1, net2)

optimizer1=optim.SGD(net1.parameters(), lr=args.lr,
                     momentum=0.9, weight_decay=5e-4)
optimizer2=optim.SGD(net2.parameters(), lr=args.lr,
                     momentum=0.9, weight_decay=5e-4)

CE=nn.CrossEntropyLoss(reduction='none')
CEloss=nn.CrossEntropyLoss()
softCE=SoftCELoss()
if args.noise_mode == 'asym':
	conf_penalty=NegEntropy()
best_acc=0



P1=[]
P2=[]
W1=[]
W2=[]





if len(args.resume) != 0:
	load_model(args.resume, net1, net2, optimizer1, optimizer2)
	test_loader=loader.run('test')











	print('unpruned acc')
	acc=test(-1, net1, net2)



	eval_loader=loader.run('eval_train')
	print('unpruned acc')
	F1, F2=eval_train(0, net1, net2)

	w_x=torch.from_numpy(F1).gather(1, torch.from_numpy(all_labels).view(1, -1))
	w_x=w_x.view(-1, 1)
	print(w_x.mean())








	exit()
	p_epoch=150


else:
	p_epoch=0

for epoch in range(p_epoch, args.warm_up):
	test_loader=loader.run('test')
	for param_group in optimizer1.param_groups:
		param_group['lr']=0.02
	for param_group in optimizer2.param_groups:
		param_group['lr']=0.02
	warmup_trainloader=loader.run('warmup')
	print('Warmup Net1')
	warmup(epoch, net1, ema_model1, None, optimizer1, warmup_trainloader)

	print('Warmup Net2')
	warmup(epoch, net2, ema_model2, None, optimizer2, warmup_trainloader)

	acc=test(epoch, net1, net2)







save_model(f"{args.save_name}_{args.dataset}_{args.r}_{args.noise_mode}_latest",
           "./checkpoint/", net1, net2, optimizer1, optimizer2)






for epoch in range(p_epoch, args.num_epochs+1):

	lr=args.lr
	if args.num_epochs-epoch < 100:
		lr /= 10
	for param_group in optimizer1.param_groups:
		param_group['lr']=lr
	for param_group in optimizer2.param_groups:
		param_group['lr']=lr

	test_loader=loader.run('test')
	eval_loader=loader.run('eval_train')
	F1, F2=eval_train(epoch, net1, net2)
	print('Train Net')
	labeled_trainloader=loader.run('train', F1)
	train(epoch, net2, net1, ema_model2, optimizer2, labeled_trainloader)
	print('\nTrain Net2')
	labeled_trainloader=loader.run('train', F2)
	train(epoch, net1, net2, ema_model1, optimizer1, labeled_trainloader)
	acc=test(epoch, net1, net2)
	if acc > best_acc:
		best_acc=acc
		save_model(f"{args.save_name}_{args.dataset}_{args.r}_{args.noise_mode}_best",
		           "./checkpoint/", net1, net2, optimizer1, optimizer2)

	save_model(f"{args.save_name}_{args.dataset}_{args.r}_{args.noise_mode}_latest",
	           "./checkpoint/", net1, net2, optimizer1, optimizer2)
