"""Graph-attention Generative Adversarial Networks (GAT-GAN) Codebase.

Reference: Srikrishna Iyer, Teng Teck Hou,
"Graph-attention Generative Adversarial Networks," ArXiv.

Last updated Date: 26th July 2023
Code author: Srikrishna Iyer (srikrish001@e.ntu.edu.sg)"""

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
from torch import autograd
from utils import *
from discriminator import Discriminator
from decoder import Decoder
from generator import Generator
from args import get_parser
from utils import visualization
from metrics.predictive_metrics import predictive_score_metrics
from metrics.frechet_transformer_distance import FTD_score
from Hyperparameter_tuning import hyperparameter
from train import train_validate
import json
import os
parser = get_parser()
args = parser.parse_args()
#If use_train_params is enabled, tuned hyperparameters are used.
if args.use_train_params:
    #Calling bayesian optimization for hyperparameter tuning
    #hyperparameters supported : n_layers_gen, n_layers_disc, n_epochs,init_lr_gen, init_lr_disc
    hyperparameter().bayesian_gaussian_process() #comment this line to skip hyperparameter tuning
    with open('./Hyperparameter_results/'+args.dataset+'/'+str(args.seq_length)+'/results.txt') as f:
        lines = f.readlines()
        tuned_hyperparameters = lines[0].strip('\n').strip('x:').strip('[').strip(']').strip(' ').split(',')
        n_layers_gen = int(tuned_hyperparameters[2])
        n_layers_disc = int(tuned_hyperparameters[3])
        n_epochs = int(tuned_hyperparameters[4])
        init_lr_gen = float(tuned_hyperparameters[0])
        init_lr_disc = float(tuned_hyperparameters[1])
else:
    n_layers_gen = args.n_blocks_gen
    n_layers_disc = args.n_blocks_disc
    n_epochs = args.epochs
    init_lr_gen = args.init_lr_gen
    init_lr_disc = args.init_lr_disc
kernel_size = args.kernel_size
feat_gat_embed_dim = args.feat_gat_embed_dim
time_gat_embed_dim = args.time_gat_embed_dim
alpha = args.alpha
dropout = args.dropout
dataset = args.dataset
seq_length = args.seq_length
normalize = args.normalize
batch_size = args.bs
shuffle_dataset = args.shuffle_dataset
use_cuda = args.use_cuda
metric_iteration = args.metric_iteration
print('Hyperparameters : discriminator lr:'+str(init_lr_disc)+' generator lr:'+str(init_lr_gen)+' n_blocks_gen:'+str(n_layers_gen)+' n_blocks_disc:'+str(n_layers_disc)+' epochs:'+str(n_epochs))

if use_cuda:
    device = torch.device('cuda:0')
    print('Using GPU:',device)
else:
    device = torch.device("cpu")
    print('Using CPU')

input_path = f'data/'
output_path = f'output/'+dataset+'/'
model_path = f'models/'+dataset+'/'+str(seq)+'/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if dataset in ['traffic', 'motor','ecg']:
    data = get_data(dataset+'_'+str(seq_length), input_path, normalize,seq_length)
else:
    raise Exception(f'Dataset "{dataset}" not available.')


train_data = data[:int(len(data)*0.8),:,:]
#train_data = preprocessed_data
n_features = np.shape(train_data)[2]
train_data = torch.tensor(np.array(train_data)).float()
train_data = train_data.reshape(train_data.shape[0],1,train_data.shape[1],train_data.shape[2])
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=shuffle_dataset,drop_last=False)

val_data = data[int(len(data) * 0.8):, :, :]
# train_data = preprocessed_data
val_data = torch.tensor(np.array(val_data)).float()
val_data = val_data.reshape(val_data.shape[0], 1, val_data.shape[1], val_data.shape[2])
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=shuffle_dataset,
                                           drop_last=False)

ae_criterion = nn.MSELoss()
EPS = 1e-15

encoder = Generator(n_features,
    seq_length,
    fcn_dim = seq_length,
    n_layers = n_layers_gen,
    kernel_size=kernel_size,
    feat_gat_embed_dim=feat_gat_embed_dim,
    time_gat_embed_dim=time_gat_embed_dim,
    dropout=dropout,
    alpha=alpha
    )
#encoder = nn.DataParallel(encoder)
encoder = encoder.to(device).float()

decoder = Decoder(n_features,
    seq_length,
    fcn_dim = seq_length,
    n_layers = n_layers_gen,
    kernel_size = kernel_size,
    feat_gat_embed_dim=feat_gat_embed_dim,
    time_gat_embed_dim=time_gat_embed_dim,
    dropout=dropout,
    alpha=alpha
    )
#decoder = nn.DataParallel(decoder)
decoder = decoder.to(device).float()

disciminator = Discriminator(n_features,
                        seq_length,
                        kernel_size = kernel_size,
                        fcn_dim=seq_length,
                        n_layers=n_layers_disc,
                        feat_gat_embed_dim=feat_gat_embed_dim,
                        time_gat_embed_dim=time_gat_embed_dim,
                        dropout=dropout,
                        alpha=alpha
                        )
#disciminator = nn.DataParallel(disciminator)
disciminator = disciminator.to(device).float()

#encode/decode optimizers
optim_encoder = torch.optim.Adam(encoder.parameters(), lr=init_lr_gen)
optim_decoder = torch.optim.Adam(decoder.parameters(), lr=init_lr_gen)
optim_discriminator = torch.optim.Adam(disciminator.parameters(), lr=init_lr_disc)
optim_encoder_reg = torch.optim.Adam(encoder.parameters(), lr=init_lr_gen)

#gradually reduce the lr
schedulerDisc = torch.optim.lr_scheduler.ExponentialLR(optim_discriminator, gamma=0.99)
schedulerD = torch.optim.lr_scheduler.ExponentialLR(optim_decoder, gamma=0.99)
schedulerE = torch.optim.lr_scheduler.ExponentialLR(optim_encoder, gamma=0.99)
train_loss_ae = []
train_loss_disc = []
train_loss_gen = []
val_loss_ae = []
val_loss_disc = []
val_loss_gen = []
for epoch in tqdm(range(1,n_epochs),desc='Training GAT-GAN for '+dataset+' for sequence length:'+str(seq_length)):
    l1, l2 ,l3, gen_data, real_data, encoder_model, decoder_model, discrminator_model = train_validate(train_data,device,encoder, decoder, EPS, disciminator, train_loader, optim_encoder, optim_decoder, optim_discriminator,optim_encoder_reg, True)
    l1_val, l2_val ,l3_val, gen_data_val, real_data_val, _, _, _ = train_validate(train_data,device,encoder, decoder, EPS, disciminator, val_loader, optim_encoder, optim_decoder, optim_discriminator,optim_encoder_reg, False)

    #print('\n Epoch [%d], recon_loss: %.4f, discriminator_loss :%.4f , generator_loss:%.4f'% (epoch, l1,l2, l3))
    if epoch == (n_epochs-1):
        torch.save(encoder_model, model_path + 'generator.pt')
        torch.save(decoder_model, model_path + 'decoder.pt')
        torch.save(discrminator_model, model_path + 'discriminator.pt')
    train_loss_ae.append(l1)
    train_loss_disc.append(l2)
    train_loss_gen.append(l3)
    val_loss_ae.append(l1)
    val_loss_disc.append(l2)
    val_loss_gen.append(l3)

if args.save_metrics :
    visualization(real_data, gen_data, 'pca', epoch, output_path)
    visualization(real_data, gen_data, 'tsne', epoch, output_path)

    '''#saving loss functions
    # Create the subplot layout
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    axs[0].plot(np.arange(0,len(train_loss_ae)), train_loss_ae, label='train',color='blue')
    axs[0].plot(np.arange(0,len(val_loss_ae)), val_loss_ae, label='val',color='red')
    axs[0].set_title('Reconstruction loss')
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(np.arange(0, len(train_loss_gen)), train_loss_gen, label='train',color='blue')
    axs[1].plot(np.arange(0, len(val_loss_gen)), val_loss_gen, label='val',color='red')
    axs[1].set_title('Generator loss')
    axs[1].grid()
    axs[0].legend()

    axs[2].plot(np.arange(0, len(train_loss_disc)), train_loss_disc, label='train',color='blue')
    axs[2].plot(np.arange(0, len(val_loss_disc)), val_loss_disc, label='val',color='red')
    axs[2].set_title('Discriminator loss')
    axs[2].grid()
    axs[0].legend()

    plt.savefig(output_path + 'Epoch_' + str(epoch) + '/seq_' + str(seq_length) +'/loss_functions.jpg')
    plt.close(fig)'''
    metric_results = dict()
    real_data_reshaped = real_data.reshape(real_data.shape[0]*real_data.shape[1],real_data.shape[2])
    gen_data_reshaped = gen_data.reshape(gen_data.shape[0] * gen_data.shape[1], gen_data.shape[2])

    pd.DataFrame(real_data_reshaped).to_csv(
        output_path + 'Epoch_' + str(epoch) + '/seq_' + str(seq_length) + '/real_data.csv', index=False)
    pd.DataFrame(gen_data_reshaped).to_csv(
        output_path + 'Epoch_' + str(epoch) + '/seq_' + str(seq_length) + '/gen_data.csv', index=False)

    # 1. Predictive score
    predictive_score = list()
    for tt in tqdm(range(metric_iteration),desc='Computing predictive metrics'):
        temp_pred = predictive_score_metrics(real_data, gen_data)
        predictive_score.append(temp_pred)
    metric_results['predictive_mean'] = np.mean(predictive_score)
    metric_results['predictive_std'] = np.std(predictive_score)
    metric_results['predictive_scores'] = predictive_score

    # 2. Frechet transformer distance
    ftd = []
    for i in tqdm(range(10), desc='Computing FTD scores for ' + dataset):
        FTD_score().Transformer_embeddings(real_data.reshape(real_data.shape[0]*real_data.shape[1],real_data.shape[2]), dataset, seq, 'gat_gan', 'actual_data')
        FTD_score().Transformer_embeddings(gen_data.reshape(gen_data.shape[0]*gen_data.shape[1],gen_data.shape[2]), dataset, seq, 'gat_gan', 'generated_data')
        real_embeddings = torch.load('Embeddings_path/' + dataset + '/' + 'gat_gan' + '/' + str(
            seq) + '/' + 'actual_data_embeddings.pt').cpu().detach().numpy()
        fake_embeddings = torch.load('Embeddings_path/' + dataset + '/' + 'gat_gan' + '/' + str(
            seq) + '/' + 'generated_data_embeddings.pt').cpu().detach().numpy()
        ftd_score = FTD_score().calculate_ftd(fake_embeddings, real_embeddings)
        ftd.append(ftd_score)
    metric_results['ftd_mean'] = np.mean(ftd)
    metric_results['ftd_std'] = np.std(ftd)
    metric_results['ftd_scores'] = ftd

    # saving predictive and FTD scores
    os.makedirs(output_path + 'Epoch_' + str(epoch) + '/seq_' + str(seq_length), exist_ok=True)
    with open(output_path + 'Epoch_' + str(epoch) + '/seq_' + str(seq_length) + '/results.txt', "w") as f:
        # Convert the dictionary to a JSON string
        json_str = json.dumps(metric_results)
        # Write the JSON string to the text file
        f.write(json_str)


