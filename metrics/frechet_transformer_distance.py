import numpy as np
import pandas as pd
import scipy
from scipy import linalg
from GAT_GAN.modules import TSTransformerEncoder
from torch.utils.data import TensorDataset, DataLoader
import torch
import os
from tqdm import tqdm
import json
#import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FTD_score():
    def data_reshape(self,data,max_seq_len):
        total_samples = data.shape[0]
        num_samples = int(total_samples / max_seq_len)
        total_samples = num_samples * max_seq_len
        temp_data = data[:total_samples].reshape((num_samples, max_seq_len, data.shape[1]))
        return temp_data

    def Transformer_embeddings(self,data,dataset,seq_length,algo,datatype):
        #path : path to real/fake preprocessed_data
        data = self.data_reshape(data,seq_length)

        # Basic Parameters
        samples, seq , feat_dim = data.shape
        max_len = seq_length
        batch_size = 32
        X = data[:,:-1,:] #To predict the last time stamp in the sequence
        y = data[:, -1, :].reshape(data.shape[0],1,data.shape[-1])
        train = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)

        #Transformer parameters
        d_model = feat_dim*2
        n_heads = feat_dim #number of multi-head attention heads
        num_layers = 3 #num of transformer encoder layers
        dim_feedforward = 256

        #Loss function
        criterion = torch.nn.L1Loss()
        lr = 0.0001
        num_epochs = 100

        embedder = TSTransformerEncoder(feat_dim,max_len-1, d_model,n_heads,num_layers,dim_feedforward)

        optim = torch.optim.Adam(embedder.parameters(), lr=lr, betas=(0.9, 0.999))

        embedder.to(device).float()
        #embedder.train()
        losses = []
        for epoch in range(1, num_epochs + 1):
            losses_temp = []
            embeddings_temp = torch.empty((0, max_len-1 , feat_dim)).to(device)
            for i, (X, Y) in enumerate((train_loader)):

                X = X.float().to(device)
                Y = Y.float().to(device)

                y_logit, embeddings = embedder(X)
                loss = criterion(y_logit.reshape(y_logit.shape[0],y_logit.shape[2],y_logit.shape[1]), Y)
                losses_temp.append(loss.item())
                loss.backward()
                optim.step()

                embeddings_temp = torch.cat((embeddings_temp,embeddings),dim=0)
            losses.append(np.mean(losses_temp))

        embeddings_path = 'Embeddings_path/' + dataset + '/' + algo + '/' + str(seq) + '/'
        os.makedirs(embeddings_path, exist_ok=True)
        torch.save(embeddings_temp[1:,:,:], embeddings_path + '/' + datatype + '_embeddings.pt')


    def calculate_ftd(self,generated_embeddings,real_embeddings):
        # calculate mean and covariance statistics
        generated_embeddings = generated_embeddings.reshape(generated_embeddings.shape[0] * generated_embeddings.shape[1],
                                                            generated_embeddings.shape[2])
        real_embeddings = real_embeddings.reshape(real_embeddings.shape[0] * real_embeddings.shape[1],
                                                            real_embeddings.shape[2])

        mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
        mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings,  rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        ftd = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return ftd


'''if __name__ == '__main__':
    root_path = 'A:/User/envs/gat_gan/GAT_GAN/preprocessed_data/'
    for dataset in ['Traffic']:
        for algo in ['GMMN','RCGAN','SigCWGAN','TimeGAN']:
            for seq in [16,64,128,256]:
                ftd = []
                ftd_results = dict()
                for i in tqdm(range(10), desc='Computing FTD scores for '+dataset+' using '+algo):
                    for datatype in ['actual_data','generated_data']:
                        preprocessed_data = pd.read_csv(root_path + dataset + '/' + str(seq) + '/seed=0' + '/' + algo+'/'+datatype+'.csv').values
                        FTD_score().Transformer_embeddings(preprocessed_data,dataset,seq,algo,datatype)
                    real_embeddings = torch.load('Embeddings_path/' + dataset + '/' + algo + '/' + str(seq) + '/'+'actual_data_embeddings.pt').cpu().detach().numpy()
                    fake_embeddings = torch.load('Embeddings_path/' + dataset + '/' + algo + '/' + str(seq) + '/'+'generated_data_embeddings.pt').cpu().detach().numpy()
                    ftd_score = FTD_score().calculate_ftd(fake_embeddings,real_embeddings)
                    ftd.append(ftd_score)
                ftd_results['ftd_mean'] = np.mean(ftd)
                ftd_results['ftd_std'] = np.std(ftd)

                with open('Embeddings_path/' + dataset + '/' + algo + '/' + str(seq) + '/'+ 'fid_results.txt', "w") as f:
                    json_str = json.dumps(ftd_results)
                    f.write(json_str)'''