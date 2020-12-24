# code based on https://github.com/danieltan07

import numpy as np
import argparse 
import torch
import time

from train import TrainerDAGMM
from test import eval
from preprocess import get_KDDCup99
from os import listdir
from os.path import isfile, join

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="number of epochs")
    parser.add_argument("--patience", type=int, default=50, 
                        help="Patience for Early Stopping")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--lr_milestones', type=list, default=[50],
                        help='Milestones at which the scheduler multiply the lr by 0.1')
    parser.add_argument("--batch_size", type=int, default=1024, 
                        help="Batch size")
    parser.add_argument('--latent_dim', type=int, default=1,
                        help='Dimension of the latent variable z')
    parser.add_argument('--n_gmm', type=int, default=4,
                        help='Number of Gaussian components ')
    parser.add_argument('--lambda_energy', type=float, default=0.1,
                        help='Parameter labda1 for the relative importance of sampling energy.')
    parser.add_argument('--lambda_cov', type=int, default=0.005,
                        help='Parameter lambda2 for penalizing small values on'
                             'the diagonal of the covariance matrix')
    #parsing arguments.
    args = parser.parse_args() 

    #check if cuda is available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get train and test dataloaders.
    start_time = time.time()
    filesPath = "./Donut/"
    onlyfiles = [f for f in listdir(filesPath) if isfile(join(filesPath, f))]
    for nameFile in onlyfiles:
        print(nameFile)
        file_time = time.time()
        dir_address = filesPath + nameFile
        data = get_KDDCup99(args,data_dir=dir_address)

        DAGMM = TrainerDAGMM(args, data, device)
        DAGMM.train()
        eval(DAGMM.model, data, device, 4) # data[1]: test dataloader
        print("TimeFile:"+str(time.time()-file_time))
    end_time = time.time()
    print("Time:"+str(end_time-start_time))
