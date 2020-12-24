import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import pickle as pl
import pandas as pd


class KDDCupData:
    def __init__(self, data_dir, mode):
        """Loading the data for train and test."""
        #data = np.load(data_dir, allow_pickle=True)
        #data = pd.read_csv(data_dir, header=None, index_col=None, skiprows=0, sep=',')
        data = pd.read_csv(data_dir, index_col=None)
#        data = pd.read_csv(data_dir, header=None, index_col=None)
#        features = data.values
#        label = pd.read_csv("/home/ubuntu/DAGMM/SMD/label/"+ data_dir[-15:], header=None, index_col=None)
#        labels = label.values
        features = data.iloc[:, [0,1]].values
        labels = data.iloc[:, 2].values
	#labels = data["kdd"][:,-1]
        #features = data["kdd"][:,:-1]
        #In this case, "atack" has been treated as normal data as is mentioned in the paper
#        normal_data = features[labels==0] 
#        normal_labels = labels[labels==0]

#        n_train = int(normal_data.shape[0]*0.5)
        #print(n_train)
#        ixs = np.arange(normal_data.shape[0])
        #print(ixs.shape())
#        np.random.shuffle(ixs)
#        normal_data_test = normal_data[ixs[n_train:]]
#        normal_labels_test = normal_labels[ixs[n_train:]]

        if mode == 'train':
            self.x = features #[ixs[:n_train]]
            self.y = labels # [ixs[:n_train]]
            #print(self.x)
        elif mode == 'test':
            #anomalous_data = features[labels==1]
            #anomalous_labels = labels[labels==1]
            self.x = features #np.concatenate((anomalous_data, normal_data_test), axis=0)
            self.y = labels #np.concatenate((anomalous_labels, normal_labels_test), axis=0)

    def __len__(self):
        """Number of images in the object dataset."""
        return self.x.shape[0]

    def __getitem__(self, index):
        """Return a sample from the dataset."""
        return np.float32(self.x[index]), np.float32(self.y[index])



def get_KDDCup99(args, data_dir='./ECG/chfdb_chf13.csv'):
    """Returning train and test dataloaders."""
    train = KDDCupData(data_dir, 'train')
    dataloader_train = DataLoader(train, batch_size=args.batch_size, 
                              shuffle=True, num_workers=0)

    test = KDDCupData(data_dir, 'train')
    dataloader_test = DataLoader(test, batch_size=args.batch_size, 
                              shuffle=True, num_workers=0)
    return dataloader_train, dataloader_test
