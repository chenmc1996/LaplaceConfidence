from torch.utils.data import Dataset, DataLoader
import copy
from randaugment import RandAugment
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
import os

class imagenet_dataset(Dataset):
    def __init__(self, root, transform, num_class):
        self.root = root+'imagenet/val/'
        self.transform = transform
        self.val_data = []
        for c in range(num_class):
            imgs = os.listdir(self.root+str(c))
            for img in imgs:
                self.val_data.append([c,os.path.join(self.root,str(c),img)])                
                
    def __getitem__(self, index):
        data = self.val_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')   
        img = self.transform(image) 
        return img, target
    
    def __len__(self):
        return len(self.val_data)

class TransformTwice:
    def __init__(self, transform1,transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform1(inp)
        out2 = self.transform2(inp)
        return out1, out2

class TransformThrice:
    def __init__(self, transform1,transform2,transform3):
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3

    def __call__(self, inp):
        out1 = self.transform1(inp)
        out2 = self.transform2(inp)
        out3 = self.transform3(inp)
        return out1, out2,out3

class webvision_dataset(Dataset): 
    def __init__(self, root, transform, mode, num_class=50, pred=[], probability=[], log=''): 
        self.root = root
        self.transform = transform
        self.mode = mode  
        self.log=log
     
        if self.mode=='test':
            with open(self.root+'info/val_filelist.txt') as f:
                lines=f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    self.val_imgs.append(img)
                    self.val_labels[img]=target                             
        else:    
            with open(self.root+'info/train_filelist_google.txt') as f:
                lines=f.readlines()    
            train_imgs = []
            self.train_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    train_imgs.append(img)
                    self.train_labels[img]=target            
            if self.mode == 'all':
                self.train_imgs = train_imgs
            else:                   
                if self.mode == "labeled":
                    #pred_idx = pred.nonzero()[0]
                    #self.train_imgs = [train_imgs[i] for i in pred_idx]                
                    #self.probability = [probability[i] for i in pred_idx]            
                    self.train_imgs = train_imgs
                    self.probability = probability 
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))            
                    log.write('Numer of labeled samples:%d \n'%(pred.sum()))
                    log.flush()                          
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                               
                    self.train_imgs = [train_imgs[i] for i in pred_idx]                           
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))            
                    
    def __getitem__(self, index):
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path] 
            prob = self.probability[index]
            image = Image.open(self.root+img_path).convert('RGB')    
            img1 = self.transform(image) 
            #img2 = self.transform(image) 
            return img1, target, prob
        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(self.root+img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2  
        elif self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(self.root+img_path).convert('RGB')   
            img = self.transform(image)
            return img, target, index        
        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]     
            image = Image.open(self.root+'val_images_256/'+img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)    
        
class webvision_dataloader():  
    #def __init__(self, root, batch_size, num_batches, num_workers):    
    def __init__(self, batch_size, num_class, num_workers, root, log):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = root
        self.num_class=num_class
        self.log=log
                   
        self.transform_train = transforms.Compose([
                transforms.Resize(320),
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ]) 
        self.transform_test = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])  
        self.transform_imagenet = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])         
        self.strong_transform = copy.deepcopy(self.transform_train)
        self.strong_transform.transforms.insert(0, RandAugment(3,5))

    def run(self,mode,pred=[],prob=[]):        
        if mode=='warmup':
            all_dataset = webvision_dataset(root=self.root, transform=self.transform_train, mode="all", num_class=self.num_class)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)                 
            return trainloader
        elif mode=='train':
            labeled_dataset = webvision_dataset(root=self.root, transform=TransformTwice(self.transform_train,self.strong_transform), mode="labeled",num_class=self.num_class,pred=pred,probability=prob,log=self.log)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size*5,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)        
            return labeled_trainloader
        elif mode=='test':
            test_dataset = webvision_dataset(root=self.root, transform=self.transform_test, mode='test', num_class=self.num_class)      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size*5,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)               
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = webvision_dataset(root=self.root, transform=self.transform_test, mode='all', num_class=self.num_class)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size*5,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)               
            return eval_loader     
        
        elif mode=='imagenet':
            imagenet_val = imagenet_dataset(root=self.root, transform=self.transform_imagenet, num_class=self.num_class)      
            imagenet_loader = DataLoader(
                dataset=imagenet_val, 
                batch_size=self.batch_size*5,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)               
            return imagenet_loader     
