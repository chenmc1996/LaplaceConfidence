import torch.utils.data as data
from scipy.spatial import distance
from scipy.special import softmax
import copy
from PIL import Image
import os
import os.path
import sys
import numpy as np
import time
import faiss
from faiss import normalize_L2
import scipy
import torch.nn.functional as F
import torch
import scipy.stats
import pickle

def one_iter_true_c1m( Z,_Y, k = 100, max_iter = 30, l2 = True, index="ip",n_labels=None,alpha = 0.99,gpuid=0,classes=14):
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{gpuid}"

    Z=np.ascontiguousarray(Z)

    if l2:
        normalize_L2(Z)


    #labeled_idx = np.asarray(labeled_idx)
    #unlabeled_idx = np.asarray(unlabeled_idx)

    # kNN search for the graph
    d = Z.shape[1]
    if index == "ip":
        index = faiss.IndexFlatIP(d)   # build the index
    if index == "l2":
        index = faiss.IndexFlatL2(d)   # build the index


    index.add(Z)
    N = Z.shape[0]
    D, I = index.search(Z, k + 1)

    # Create the graph
    D = D[:,1:]
    I = I[:,1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx,(k,1)).T
    W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
    W = W + W.T.multiply(W.T > W) - W.multiply(W.T > W)

    # Normalize the graph
    W = W - scipy.sparse.diags(W.diagonal())
    S = W.sum(axis = 1)
    S[S==0] = 1
    D = np.array(1./ np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D

    # Initiliaze the y vector for each class
    F = np.zeros((N,classes))
    A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
    #Y = np.zeros((N,len(classes)))
    #Y[labeled_idx,labels[labeled_idx]] = 1

    for i in range(classes):
        f, _ = scipy.sparse.linalg.cg(A, _Y[:,i], tol=1e-3, maxiter=max_iter)
        F[:,i] = f
    F[F < 0] = 0

    F = softmax(F,1)
    return F

def one_iter_true_wm( Z,_Y, k = 50, max_iter = 30, l2 = True, index="ip",n_labels=None,alpha = 0.99,gpuid=0):
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{gpuid}"

    Z=np.ascontiguousarray(Z)

    if l2:
        normalize_L2(Z)  
    classes=50
    
    #labeled_idx = np.asarray(labeled_idx)        
    #unlabeled_idx = np.asarray(unlabeled_idx)
    
    # kNN search for the graph
    d = Z.shape[1]       
    if index == "ip":
        index = faiss.IndexFlatIP(d)   # build the index
    if index == "l2":            
        index = faiss.IndexFlatL2(d)   # build the index

    index.add(Z) 
    N = Z.shape[0]
    D, I = index.search(Z, k + 1)

    # Create the graph
    D = D[:,1:] 
    I = I[:,1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx,(k,1)).T
    W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))       
    W = W + W.T.multiply(W.T > W) - W.multiply(W.T > W)
   
    # Normalize the graph
    W = W - scipy.sparse.diags(W.diagonal())
    S = W.sum(axis = 1)
    S[S==0] = 1
    D = np.array(1./ np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D        

    # Initiliaze the y vector for each class
    F = np.zeros((N,classes))
    A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn        
    #Y = np.zeros((N,len(classes)))
    #Y[labeled_idx,labels[labeled_idx]] = 1
    
    for i in range(classes):
        f, _ = scipy.sparse.linalg.cg(A, _Y[:,i], tol=1e-3, maxiter=max_iter)
        F[:,i] = f
    F[F < 0] = 0 

    F = softmax(F,1)
    return F

def add(moving_window,new_element,quee):
    if len(moving_window)==quee:
        del moving_window[0]
        moving_window.append(new_element)
    else:
        moving_window.append(new_element)
    return moving_window

def one_iter_true_epli( Z,_Y,part=None, k = 50, max_iter = 30, l2 = True, index="ip",n_labels=None,alpha = 0.99,gpuid=0,classes=10,dropedge=1):
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{gpuid}"
    Z=np.ascontiguousarray(Z)

    with open(f'cifar{classes}_label') as f:
        all_labels=np.asarray(eval(f.readlines()[0]))

    labels=all_labels
    num_samples=all_labels.shape[0]
    p_labels = np.argmax(_Y[:num_samples],1)
    correct_idx = (p_labels == labels[:num_samples])
    acc = correct_idx.mean()   
    print("Oringinal Label Accuracy {:.2f}".format(100*acc) + "%")    

    if l2:
        normalize_L2(Z)  

    d = Z.shape[1]       
    if index == "ip":
        index = faiss.IndexFlatIP(d)   # build the index
    if index == "l2":            
        index = faiss.IndexFlatL2(d)   # build the index

    index.add(Z) 
    N = Z.shape[0]
    D, I = index.search(Z, k + 1)

    D = D[:,1:] 
    I = I[:,1:]

    avg_d=np.mean(D)

    for i in range(N):
        if D[i,5]<avg_d:
            D[i,5:]=0
        else:
            D[i,:]=D[i,:]*(D[i,:]>avg_d)

    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx,(k,1)).T
    W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))       
    W = W + W.T.multiply(W.T > W) - W.multiply(W.T > W)
    
    W = W - scipy.sparse.diags(W.diagonal())

    S = W.sum(axis = 1)
    S[S==0] = 1
    D = np.array(1./ np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D        

    # Initiliaze the y vector for each class
    F = np.zeros((N,classes))
    A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn        
    
    for i in range(classes):
        f, _ = scipy.sparse.linalg.cg(A, _Y[:,i], tol=1e-3, maxiter=max_iter)
        F[:,i] = f
    F[F < 0] = 0 
    
    F = softmax(F,1)

    #print(F.shape,num_samples)
    #p_labels = np.argmax(F[:num_samples]+F[num_samples:],1)
    p_labels = np.argmax(F[:num_samples],1)
    #print(p_labels.shape,labels[:num_samples].shape)
    correct_idx = (p_labels == labels[:num_samples])
    acc = correct_idx.mean()
    print("Pseudo Label Accuracy {:.2f}".format(100*acc) + "%")    
    return F

def one_iter_true( Z,_Y,weight=None,history_W=None,quee=0,part=None, k = 50, max_iter = 30, l2 = True, index="ip",n_labels=None,alpha = 0.99,gpuid=0,classes=10,dropedge=1):
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{gpuid}"
    Z=np.ascontiguousarray(Z)
    #print(Z.shape)
    #exit()

    with open(f'cifar{classes}_label') as f:
        all_labels=np.asarray(eval(f.readlines()[0]))

    labels=all_labels
    num_samples=all_labels.shape[0]
    ### Extract and test pseduo labels for accuracy
    #probs_iter = F.normalize(torch.tensor(_Y),1).numpy() 
    #probs_iter[labeled_idx] = np.zeros(len(classes)) 
    #probs_iter[labeled_idx,labels[labeled_idx]] = 1                
    p_labels = np.argmax(_Y[:num_samples],1)
    correct_idx = (p_labels == labels[:num_samples])
    acc = correct_idx.mean()   
    print("Oringinal Label Accuracy {:.2f}".format(100*acc) + "%")    


    if l2:
        normalize_L2(Z)  

    if weight is not None:
        Z=Z*weight

    
    
    #labeled_idx = np.asarray(labeled_idx)        
    #unlabeled_idx = np.asarray(unlabeled_idx)
    
    # kNN search for the graph
    d = Z.shape[1]       
    if index == "ip":
        index = faiss.IndexFlatIP(d)   # build the index
    if index == "l2":            
        index = faiss.IndexFlatL2(d)   # build the index

    index.add(Z) 
    N = Z.shape[0]
    D, I = index.search(Z, k + 1)

    # Create the graph
    D = D[:,1:] 
    I = I[:,1:]

    if part is not None:
        for i in range(N):
            #inds=np.random.permutation(k)[int(k*part[i])]
            k_i=int(k*part[i])
            D[i,-k_i:]=0
    #print(D.shape)
    #print(D[0])
    #exit()


    if dropedge!=1:
        inds=np.random.permutation(k)[:int(k*dropedge)]
        D,I=D[:,inds],I[:,inds]
        k=int(k*dropedge)

    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx,(k,1)).T
    W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))       
    W = W + W.T.multiply(W.T > W) - W.multiply(W.T > W)
    
    if history_W is not None:
        add(history_W,W,quee)
        W=sum(history_W)/len(history_W)
   
    # Normalize the graph
    W = W - scipy.sparse.diags(W.diagonal())

    S = W.sum(axis = 1)
    S[S==0] = 1
    D = np.array(1./ np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D        

    # Initiliaze the y vector for each class
    F = np.zeros((N,classes))
    A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn        
    #Y = np.zeros((N,len(classes)))
    #Y[labeled_idx,labels[labeled_idx]] = 1
    
    for i in range(classes):
        f, _ = scipy.sparse.linalg.cg(A, _Y[:,i], tol=1e-3, maxiter=max_iter)
        F[:,i] = f
    F[F < 0] = 0 
    
    F = softmax(F,1)

    #print(F.shape,num_samples)
    #p_labels = np.argmax(F[:num_samples]+F[num_samples:],1)
    p_labels = np.argmax(F[:num_samples],1)
    #print(p_labels.shape,labels[:num_samples].shape)
    correct_idx = (p_labels == labels[:num_samples])
    acc = correct_idx.mean()
    print("Pseudo Label Accuracy {:.2f}".format(100*acc) + "%")    
    return F

def one_iter_true_birdge( Z,_Y,history_W=[], k = 50, max_iter = 30, l2 = True, index="ip",n_labels=None,alpha = 0.99,gpuid=0,classes=10,dropedge=1):
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{gpuid}"
    Z=np.ascontiguousarray(Z)

    with open(f'cifar{classes}_label') as f:
        all_labels=np.asarray(eval(f.readlines()[0]))

    labels=all_labels
    num_samples=all_labels.shape[0]
    ### Extract and test pseduo labels for accuracy
    #probs_iter = F.normalize(torch.tensor(_Y),1).numpy() 
    #probs_iter[labeled_idx] = np.zeros(len(classes)) 
    #probs_iter[labeled_idx,labels[labeled_idx]] = 1                
    p_labels = np.argmax(_Y[:num_samples],1)
    correct_idx = (p_labels == labels[:num_samples])
    acc = correct_idx.mean()   
    print("Oringinal Label Accuracy {:.2f}".format(100*acc) + "%")    


    if l2:
        normalize_L2(Z)  
    
    
    #labeled_idx = np.asarray(labeled_idx)        
    #unlabeled_idx = np.asarray(unlabeled_idx)
    
    # kNN search for the graph
    d = Z.shape[1]       
    if index == "ip":
        index = faiss.IndexFlatIP(d)   # build the index
    if index == "l2":            
        index = faiss.IndexFlatL2(d)   # build the index

    index.add(Z) 
    N = Z.shape[0]
    D, I = index.search(Z, k + 1)

    

    # Create the graph
    D = D[:,1:] 
    I = I[:,1:]

    if dropedge!=1:
        inds=np.random.permutation(k)[:int(k*dropedge)]
        D,I=D[:,inds],I[:,inds]
        k=int(k*dropedge)

    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx,(k,1)).T
    W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))       

    #bridge=[]
    #for i in range(num_samples):
    #    bridge.append(1-distance.cosine(Z[i],Z[i+num_samples]))
    #bridge=np.array(bridge)

    #W[np.arange(num_samples),np.arange(num_samples,num_samples*2)] = bridge
    W[np.arange(num_samples),np.arange(num_samples,num_samples*2)] = 1

    W = W + W.T.multiply(W.T > W) - W.multiply(W.T > W)
    
    #add(history_W,W)
    #W=sum(history_W)/len(history_W)
   
    # Normalize the graph
    W = W - scipy.sparse.diags(W.diagonal())

    S = W.sum(axis = 1)
    S[S==0] = 1
    D = np.array(1./ np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D        

    # Initiliaze the y vector for each class
    F = np.zeros((N,classes))
    A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn        
    #Y = np.zeros((N,len(classes)))
    #Y[labeled_idx,labels[labeled_idx]] = 1
    
    for i in range(classes):
        f, _ = scipy.sparse.linalg.cg(A, _Y[:,i], tol=1e-3, maxiter=max_iter)
        F[:,i] = f
    F[F < 0] = 0 

    
    F = softmax(F,1)

    p_labels = np.argmax(F[:num_samples]+F[num_samples:],1)
    #print(p_labels.shape,labels[:num_samples].shape)
    correct_idx = (p_labels == labels[:num_samples])
    acc = correct_idx.mean()
    print("Pseudo Label Accuracy {:.2f}".format(100*acc) + "%")    

    return F
    
def one_iter_true_only_birdge(Z,_Y,history_W=[], k = 50, max_iter = 30, l2 = True, index="ip",n_labels=None,alpha = 0.99,gpuid=0,classes=10,dropedge=1):
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{gpuid}"
    Z=np.ascontiguousarray(Z)

    with open(f'cifar{classes}_label') as f:
        all_labels=np.asarray(eval(f.readlines()[0]))

    labels=all_labels
    num_samples=all_labels.shape[0]
    ### Extract and test pseduo labels for accuracy
    #probs_iter = F.normalize(torch.tensor(_Y),1).numpy() 
    #probs_iter[labeled_idx] = np.zeros(len(classes)) 
    #probs_iter[labeled_idx,labels[labeled_idx]] = 1                
    p_labels = np.argmax(_Y[:num_samples],1)
    correct_idx = (p_labels == labels[:num_samples])
    acc = correct_idx.mean()   
    print("Oringinal Label Accuracy {:.2f}".format(100*acc) + "%")    


    if l2:
        normalize_L2(Z)  
    
    
    #labeled_idx = np.asarray(labeled_idx)        
    #unlabeled_idx = np.asarray(unlabeled_idx)
    
    # kNN search for the graph
    d = Z.shape[1]       
    if index == "ip":
        index = faiss.IndexFlatIP(d)   # build the index
    if index == "l2":            
        index = faiss.IndexFlatL2(d)   # build the index

    index.add(Z) 
    N = Z.shape[0]
    D, I = index.search(Z, k + 1)

    

    # Create the graph
    D = D[:,1:] 
    I = I[:,1:]

    if dropedge!=1:
        inds=np.random.permutation(k)[:int(k*dropedge)]
        D,I=D[:,inds],I[:,inds]
        k=int(k*dropedge)

    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx,(k,1)).T
    W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))       

    #bridge=[]
    #for i in range(num_samples):
    #    bridge.append(1-distance.cosine(Z[i],Z[i+num_samples]))
    #bridge=np.array(bridge)

    W[np.arange(num_samples),np.arange(num_samples,num_samples*2)] = 1.0

    W = W + W.T.multiply(W.T > W) - W.multiply(W.T > W)
    
    #add(history_W,W)
    #W=sum(history_W)/len(history_W)
   
    # Normalize the graph
    W = W - scipy.sparse.diags(W.diagonal())

    S = W.sum(axis = 1)
    S[S==0] = 1
    D = np.array(1./ np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D        

    # Initiliaze the y vector for each class
    F = np.zeros((N,classes))
    A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn        
    #Y = np.zeros((N,len(classes)))
    #Y[labeled_idx,labels[labeled_idx]] = 1
    
    for i in range(classes):
        f, _ = scipy.sparse.linalg.cg(A, _Y[:,i], tol=1e-3, maxiter=max_iter)
        F[:,i] = f
    F[F < 0] = 0 

    
    F = softmax(F,1)

    p_labels = np.argmax(F[:num_samples]+F[num_samples:],1)
    #print(p_labels.shape,labels[:num_samples].shape)
    correct_idx = (p_labels == labels[:num_samples])
    acc = correct_idx.mean()
    print("Pseudo Label Accuracy {:.2f}".format(100*acc) + "%")    

    return F


def one_iter_true_seperate_birdge( Z,_Y,history_W=[], k = 50, max_iter = 30, l2 = True, index="ip",n_labels=None,alpha = 0.99,gpuid=0,classes=10,dropedge=1):
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{gpuid}"
    Z=np.ascontiguousarray(Z)

    with open(f'cifar{classes}_label') as f:
        all_labels=np.asarray(eval(f.readlines()[0]))

    labels=all_labels
    num_samples=all_labels.shape[0]
    ### Extract and test pseduo labels for accuracy
    #probs_iter = F.normalize(torch.tensor(_Y),1).numpy()
    #probs_iter[labeled_idx] = np.zeros(len(classes))
    #probs_iter[labeled_idx,labels[labeled_idx]] = 1
    p_labels = np.argmax(_Y[:num_samples],1)
    correct_idx = (p_labels == labels[:num_samples])
    acc = correct_idx.mean()
    print("Oringinal Label Accuracy {:.2f}".format(100*acc) + "%")


    Z1,Z2=Z[:num_samples],Z[num_samples:]
    W=[]

    for Z,start in zip([Z1,Z2],[0,num_samples]):

        if l2:
            normalize_L2(Z)

        # kNN search for the graph
        d = Z.shape[1]
        index = faiss.IndexFlatIP(d)   # build the index

        index.add(Z)
        N = Z.shape[0]
        D, I = index.search(Z, k + 1)

        # Create the graph
        D = D[:,1:]
        #print(I.max(),start)
        I = I[:,1:]+start

        #if dropedge!=1:
        #    inds=np.random.permutation(k)[:int(k*dropedge)]
        #    D,I=D[:,inds],I[:,inds]
        #    k=int(k*dropedge)

        row_idx = np.arange(N)+start
        row_idx_rep = np.tile(row_idx,(k,1)).T
        #print(row_idx.max())
        #print(row_idx_rep.max())
        #print(I.max())
        W.append(scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(2*N,2*N)))

    W=sum(W)
    #return W

    #bridge=[]
    #for i in range(num_samples):
    #    bridge.append(1-distance.cosine(Z[i],Z[i+num_samples]))
    #bridge=np.array(bridge)

    W[np.arange(num_samples),np.arange(num_samples,num_samples*2)] = 1.0

    W = W + W.T.multiply(W.T > W) - W.multiply(W.T > W)

    #add(history_W,W)
    #W=sum(history_W)/len(history_W)

    # Normalize the graph
    W = W - scipy.sparse.diags(W.diagonal())

    S = W.sum(axis = 1)
    S[S==0] = 1
    D = np.array(1./ np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D

    # Initiliaze the y vector for each class
    F = np.zeros((2*num_samples,classes))
    A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
    #Y = np.zeros((N,len(classes)))
    #Y[labeled_idx,labels[labeled_idx]] = 1

    for i in range(classes):
        f, _ = scipy.sparse.linalg.cg(A, _Y[:,i], tol=1e-3, maxiter=max_iter)
        F[:,i] = f
    F[F < 0] = 0


    F = softmax(F,1)

    p_labels = np.argmax(F[:num_samples]+F[num_samples:],1)
    #print(p_labels.shape,labels[:num_samples].shape)
    correct_idx = (p_labels == labels[:num_samples])
    acc = correct_idx.mean()
    print("Pseudo Label Accuracy {:.2f}".format(100*acc) + "%")

    return F

if __name__ == "__main__":

    
    Z=np.random.rand(3,5).astype(np.float32)
    d = Z.shape[1]       
    normalize_L2(Z)
    index = faiss.IndexFlatIP(d)   # build the index
    ngus = faiss.get_num_gpus()
    index = faiss.index_cpu_to_all_gpus(index,ngpu=ngus)
    index.add(Z) 
    N = Z.shape[0]
    D, I = index.search(Z, 2)
    #exit()


    with open('cifar10_label') as f, open('0.8_sym.json') as f2:
        all_labels=np.asarray(eval(f.readlines()[0]))
        train_noise=np.asarray(eval(f2.readlines()[0]))
        
    with open('NJ','rb') as f:
        prediction=np.load(f)
        presentation=np.load(f)
        print(prediction.shape)
        print(presentation.shape)
        print(prediction.dtype)
    all_labels = np.asarray(all_labels)
    #presentation=copy.deepcopy(presentation)
    #prediction=copy.deepcopy(prediction)
    presentation1=np.ascontiguousarray(presentation[:,:512])
    presentation2=np.ascontiguousarray(presentation[:,512:])
    prediction1=np.ascontiguousarray(prediction[:,:10]) 
    prediction2=np.ascontiguousarray(prediction[:,10:]) 

    #one_iter_true(presentation,all_labels,prediction)
    F1=one_iter_true(presentation1,all_labels,prediction2,alpha=0.3)
    F2=one_iter_true(presentation2,all_labels,prediction1,alpha=0.3)

    p_labels = np.argmax(prediction1+prediction2,1)
    correct_idx = (p_labels == all_labels)
    acc = correct_idx.mean()   
    print("final Label Accuracy {:.2f}".format(100*acc) + "%")    

    F1 = softmax(F1,1)
    F2 = softmax(F2,1)
    p_labels = np.argmax(F1+F2,1)
    correct_idx = (p_labels == all_labels)
    acc = correct_idx.mean()   
    print("final Label Accuracy {:.2f}".format(100*acc) + "%")    
    #one_iter_true(presentation,all_labels,prediction,alpha=0.2)
    #one_iter_true(presentation,all_labels,prediction,alpha=0.1)
    #one_iter_true(presentation,all_labels,prediction,alpha=0.05)
