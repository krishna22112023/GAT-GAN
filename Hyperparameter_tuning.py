


# hyperparameter optimization libraries
from skopt import gp_minimize  ##bayesian optimization using gaussian processes
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
from skopt.plots import plot_objective

import os
import matplotlib.pyplot as plt
import json

import numpy as np
import pandas as pd
from discriminator import Discriminator
from decoder import Decoder
from generator import Generator
from train import train_validate
from utils import get_data
from metrics.frechet_transformer_distance import FTD_score
from args import get_parser
import torch
from tqdm import tqdm

device = torch.device('cuda:0')

import warnings
warnings.filterwarnings("ignore")


class hyperparameter:
    def __init__(self):
        parser = get_parser()
        args = parser.parse_args()
        self.hyperparameter_path =  './Hyperparameter_results'
        os.makedirs(self.hyperparameter_path, exist_ok=True)
        self.no_calls = args.no_calls
        self.seq_length = args.seq_length
        self.dataset = args.dataset
        self.best_disc_score = np.inf
        self.default_parameters = [args.init_lr_gen,args.init_lr_disc,args.n_blocks_gen,args.n_blocks_disc,args.epochs]
        input_path = f'data/'
        data = get_data(self.dataset + '_' + str(self.seq_length), input_path, args.normalize, self.seq_length)
        train_len = int(0.8 * len(data))
        train_data = data[:train_len]
        val_data = data[train_len:]

        self.n_features = np.shape(train_data)[2]
        train_data = torch.tensor(np.array(train_data)).float()
        self.train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1], train_data.shape[2])
        val_data = torch.tensor(np.array(val_data)).float()
        val_data = val_data.reshape(val_data.shape[0], 1, val_data.shape[1], val_data.shape[2])
        self.kernel_size = args.kernel_size
        self.feat_gat_embed_dim = args.feat_gat_embed_dim
        self.time_gat_embed_dim = args.time_gat_embed_dim
        self.dropout = args.dropout
        self.alpha = args.alpha
        self.bs = args.bs
        self.shuffle_dataset = args.shuffle_dataset
        self.parameter_bounds = self.parameter_bounds()
    def parameter_bounds(self):
        dim_learning_rate_disc = Real(low=0.0001, high=0.01, prior='log-uniform', name='lr_disc')
        dim_learning_rate_gen = Real(low=0.0001, high=0.01, prior='log-uniform', name='lr_gen')
        dim_n_blocks_gen = Integer(low=1, high=4, name='n_blocks_gen')
        dim_n_blocks_disc = Integer(low=1, high=4, name='n_blocks_disc')
        dim_epochs = Integer(low=30, high=100, name='epochs')

        dimensions = [dim_learning_rate_disc,
                      dim_learning_rate_gen,
                      dim_n_blocks_gen,
                      dim_n_blocks_disc,
                      dim_epochs]
        return dimensions
    def fitness(self,lr_disc,lr_gen,n_blocks_gen,n_blocks_disc,epochs):
        encoder = Generator(self.n_features,
                            self.seq_length,
                            fcn_dim=self.seq_length,
                            n_layers=n_blocks_gen,
                            kernel_size=self.kernel_size,
                            feat_gat_embed_dim=self.feat_gat_embed_dim,
                            time_gat_embed_dim=self.time_gat_embed_dim,
                            dropout=self.dropout,
                            alpha=self.alpha
                            )
        # encoder = nn.DataParallel(encoder)
        encoder = encoder.to(device).float()

        decoder = Decoder(self.n_features,
                          self.seq_length,
                          fcn_dim=self.seq_length,
                          n_layers=n_blocks_gen,
                          kernel_size=self.kernel_size,
                          feat_gat_embed_dim=self.feat_gat_embed_dim,
                          time_gat_embed_dim=self.time_gat_embed_dim,
                          dropout=self.dropout,
                          alpha=self.alpha
                          )
        decoder = decoder.to(device).float()

        disciminator = Discriminator(self.n_features,
                                     self.seq_length,
                                     kernel_size=self.kernel_size,
                                     fcn_dim=self.seq_length,
                                     n_layers=n_blocks_disc,
                                     feat_gat_embed_dim=self.feat_gat_embed_dim,
                                     time_gat_embed_dim=self.time_gat_embed_dim,
                                     dropout=self.dropout,
                                     alpha=self.alpha
                                     )
        disciminator = disciminator.to(device).float()

        # encode/decode optimizers
        optim_encoder = torch.optim.Adam(encoder.parameters(), lr=lr_gen)
        optim_decoder = torch.optim.Adam(decoder.parameters(), lr=lr_gen)
        optim_discriminator = torch.optim.Adam(disciminator.parameters(), lr=lr_disc)
        optim_encoder_reg = torch.optim.Adam(encoder.parameters(), lr=lr_gen)

        # gradually reduce the lr
        schedulerDisc = torch.optim.lr_scheduler.ExponentialLR(optim_discriminator, gamma=0.99)
        schedulerD = torch.optim.lr_scheduler.ExponentialLR(optim_decoder, gamma=0.99)
        schedulerE = torch.optim.lr_scheduler.ExponentialLR(optim_encoder, gamma=0.99)
        EPS = 1e-15
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.bs, shuffle=self.shuffle_dataset)

        for epoch in tqdm(range(1,epochs), desc='Training GAT-GAN'):
            l1, l2, l3, gen_data, real_data = train_validate(self.train_data,device,encoder, decoder,EPS, disciminator, train_loader,
                                                             optim_encoder, optim_decoder, optim_discriminator,
                                                             optim_encoder_reg, True)
        gen_data = gen_data.reshape(gen_data.shape[0]*gen_data.shape[1],gen_data.shape[2])
        real_data = real_data.reshape(real_data.shape[0] * real_data.shape[1], real_data.shape[2])

        gen_data_embeddings = FTD_score().Transformer_embeddings(gen_data, self.dataset, self.seq_length, 'gat_gan','generated_data')
        real_data_embeddings = FTD_score().Transformer_embeddings(real_data, self.dataset, self.seq_length, 'gat_gan','actual_data')

        real_embeddings = torch.load('Embeddings_path/' + self.dataset + '/' + 'gat_gan' + '/' + str(
            self.seq_length) + '/' + 'actual_data_embeddings.pt').cpu().detach().numpy()
        fake_embeddings = torch.load('Embeddings_path/' + dataset + '/' + 'gat_gan' + '/' + str(
            self.seq_length) + '/' + 'generated_data_embeddings.pt').cpu().detach().numpy()
        ftd_score = FTD_score().calculate_ftd(fake_embeddings, real_embeddings)
        print('FTD score:',ftd_score)
        return ftd_score

    def bayesian_gaussian_process(self):
        print('Tuning GAT-GAN model for '+str(self.no_calls)+' rounds using Bayesian')
        @use_named_args(dimensions=self.parameter_bounds)
        def fitness_wrapper(*args, **kwargs):
            return self.fitness(*args, **kwargs)

        search_result_model = gp_minimize(
            func=fitness_wrapper,
            dimensions=self.parameter_bounds,
            acq_func='EI',  # Expected Improvement.
            n_calls=self.no_calls,
            x0=self.default_parameters,
            random_state=42)

        new_path = self.hyperparameter_path + '/' + self.dataset + '/' + str(self.seq_length) + '/'
        os.makedirs(new_path, exist_ok=True)
        plot_convergence(search_result_model)
        #plotting convergence
        plt.savefig(new_path + 'Model_convergence.png', dpi=400)
        # plotting hyperparameter minimization
        dim_names = ['init_lr_gen', 'init_lr_disc', 'n_blocks_gen', 'n_blocks_disc', 'epochs']
        plot_objective(result=search_result_model, dimensions=dim_names)
        plt.savefig(new_path + 'Hyperparameter_minimization.png', dpi=400)
        print('Hyperparameter tuning DONE ... ')

        results_dict = {
            "x": search_result_model.x,
            "fun": search_result_model.fun
        }
        with open(new_path+"results.txt", 'w') as f:
            for key, value in results_dict.items():
                f.write('%s:%s\n' % (key, value))


