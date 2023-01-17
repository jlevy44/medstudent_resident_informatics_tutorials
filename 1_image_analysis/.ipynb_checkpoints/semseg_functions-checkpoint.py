import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import segmentation_models_pytorch as smp
from PIL import Image
import tqdm
import pandas as pd
# import pytorch_lightning as pl
import os
import copy
from skimage.io import imread
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader

def load_imgs_labels(train_dir="./train",val_dir="./val"):
    train_imgs=np.stack(list(map(imread,sorted(glob.glob(os.path.join(train_dir,"imgs","*.png"))))))
    X_train=torch.FloatTensor(train_imgs).permute((0,3,1,2))/255
    
    val_imgs=np.stack(list(map(imread,sorted(glob.glob(os.path.join(val_dir,"imgs","*.png"))))))
    X_val=torch.FloatTensor(val_imgs).permute((0,3,1,2))/255
    
    train_lbls=np.stack(list(map(lambda x: imread(x)[...,0].astype(int),sorted(glob.glob(os.path.join(train_dir,"labels","*.png"))))))
    Y_train=torch.LongTensor(train_lbls)
    
    val_lbls=np.stack(list(map(lambda x: imread(x)[...,0].astype(int),sorted(glob.glob(os.path.join(val_dir,"labels","*.png"))))))
    Y_val=torch.LongTensor(val_lbls)
    
    return X_train,Y_train,X_val,Y_val


def train_model(X_train,Y_train,X_val,Y_val,save=True,n_epochs=10,path_dir = "./seg_models", device="cpu"):
    train_data=TensorDataset(X_train,Y_train)
    val_data=TensorDataset(X_val,Y_val)

    train_dataloader=DataLoader(train_data,batch_size=8,shuffle=True)
    train_dataloader_ordered=DataLoader(train_data,batch_size=8,shuffle=False)

    val_dataloader=DataLoader(val_data,batch_size=8,shuffle=False)
    
    model=smp.Unet(classes=3,in_channels=3, encoder_weights=None).to(device)
    optimizer=torch.optim.Adam(model.parameters())
    class_weight=compute_class_weight(class_weight='balanced', classes=[0,1,2], y=Y_train.numpy().flatten())
    class_weight=torch.FloatTensor(class_weight).to(device)

    loss_fn=torch.nn.CrossEntropyLoss(weight=class_weight)
    if not os.path.exists(path_dir): os.makedirs(path_dir,exist_ok=True)
    min_loss=np.inf
    for epoch in range(1,n_epochs+1):
        # training set 
        model.train(True)
        for i,(x,y_true) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                x=x.to(device)
                y_true=y_true.to(device)
            optimizer.zero_grad()

            y_pred=model(x)
            loss=loss_fn(y_pred,y_true)

            loss.backward()
            optimizer.step()

            print(f"Training: Epoch {epoch}, Batch {i}, Loss: {round(loss.item(),3)}")

        # validation set
        model.train(False)
        with torch.no_grad():
            val_loss = []
            for i,(x,y_true) in enumerate(val_dataloader):
                if torch.cuda.is_available():
                    x=x.to(device)
                    y_true=y_true.to(device)
                y_pred=model(x)
                loss=loss_fn(y_pred,y_true)
                val_loss.append(loss.item())

            val_loss=np.mean(val_loss)
            print(f"Val: Epoch {epoch}, Loss: {round(val_loss,3)}")
            if val_loss < min_loss:
                min_loss=val_loss
                best_model=copy.deepcopy(model.state_dict())
                if save:
                    with open(path_dir + f'/{epoch}.{i}_model.pkl', "w") as f:
                        torch.save(model.state_dict(), path_dir + f'/{epoch}.{i}_model.pkl')

    model.load_state_dict(best_model)
    return model

def make_predictions(X_val,model=None,save=True,path_dir = "./seg_models"):
    val_data=TensorDataset(X_val)
    val_dataloader=DataLoader(val_data,batch_size=8,shuffle=False)
    predictions=[]
    # load most recent saved model
    if model is None and save:
        model=smp.Unet(classes=3,in_channels=3,encoder_weights=None)
        model_list=sorted(glob.glob(path_dir + '/*_model.pkl'), key=os.path.getmtime)
        model.load_state_dict(torch.load(model_list[-1]))
    model.train(False)
    with torch.no_grad():
        for i,(x,) in enumerate(val_dataloader):
            if torch.cuda.is_available():
                x=x.to(device)
            y_pred=torch.softmax(model(x),1).detach().cpu().numpy()
            predictions.append(y_pred)
    predictions=np.concatenate(predictions,axis=0)
    return predictions