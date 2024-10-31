"""Graph-attention Generative Adversarial Networks (GAT-GAN) Codebase.

Last updated Date: March 1st 2023
Code author: Srikrishna Iyer (srikrishna.rameshiyer@stengg.com)"""

import torch.utils.data
from torch import autograd
import torch
import torch.nn as nn

def train_validate(train_data,
                   device,
                   encoder,
                   decoder,
                   EPS,
                   discriminator,
                   dataloader,
                   optim_encoder,
                   optim_decoder,
                   optim_discriminator,
                   optim_encoder_reg,
                   train
                   ):
    total_rec_loss = 0 #initialize reconstruction loss
    total_disc_loss = 0 #initialize discriminator loss
    total_gen_loss = 0 #initialize generator loss
    ae_criterion = nn.MSELoss()
    generated_data = torch.empty((0, train_data.shape[2],train_data.shape[3])).to(device)
    real_data = torch.empty((0, train_data.shape[2],train_data.shape[3])).to(device)
    if train:
        encoder.train()
        decoder.train()
        discriminator.train()
    else:
        encoder.eval()
        decoder.eval()
        discriminator.eval()
    N_batch_runs = 0
    for (i, data) in enumerate(dataloader):
        N_batch_runs += 1
        """ Reconstruction loss """
        for p in discriminator.parameters():
            p.requires_grad = False

        # convert a tensor object to a PyTorch variable object and then move it to a specified device (gpu/cpu)
        real_data_v = autograd.Variable(data).to(device)
        #passing input to encoder model
        real_data_v = real_data_v.reshape(real_data_v.shape[0], real_data_v.shape[2], real_data_v.shape[3])
        encoding = encoder(real_data_v) # input : [batch size,seq_length,features], output : [batch size,seq_length*features]
        #decode the latent space using decoder
        fake = decoder(encoding)
        #MSE between decoded output and input
        ae_loss = ae_criterion(fake, real_data_v)
        #nn.mseloss().item() returns the value of the MSE loss as a Python float
        total_rec_loss += ae_loss.item()
        if train:
            optim_encoder.zero_grad()
            optim_decoder.zero_grad()
            ae_loss.backward() #Backward prop of gradients of MSEloss
            optim_encoder.step()
            optim_decoder.step()

        """ Discriminator loss """
        #Generate random gaussian noise and passed into discriminator
        z_real_gauss = autograd.Variable(torch.randn(data.size()[0], encoding.shape[1], encoding.shape[2]).float()).to(device)
        D_real_gauss = discriminator(z_real_gauss)

        # pass real preprocessed_data into discriminator
        z_fake_gauss = encoder(real_data_v)
        D_fake_gauss = discriminator(z_fake_gauss)

        D_loss = -torch.mean(torch.log(D_real_gauss + EPS) + torch.log(1 - D_fake_gauss + EPS))
        total_disc_loss += D_loss.item()

        if train:
            optim_discriminator.zero_grad()
            D_loss.backward()
            optim_discriminator.step()

        """ Generator loss """
        if train:
            encoder.train()
        else:
            encoder.eval()
        z_fake_gauss = encoder(real_data_v)
        #z_fake_gauss = z_fake_gauss.reshape(z_fake_gauss.shape[0], int(z_fake_gauss.shape[1] / n_features), n_features)

        D_fake_gauss = discriminator(z_fake_gauss)
        G_loss = -torch.mean(torch.log(D_fake_gauss + EPS + ae_loss.item()))
        #G_loss = torch.mean(D_fake_gauss+EPS+ae_loss.item())
        total_gen_loss += G_loss.item()

        if train:
            optim_encoder_reg.zero_grad()
            G_loss.backward()
            optim_encoder_reg.step()

        #print('\n Step [%d], recon_loss: %.4f, discriminator_loss :%.4f , generator_loss:%.4f'
        #          % (i, ae_loss.item(), D_loss.item(), G_loss.item()))


        generated_data = torch.cat((generated_data, z_fake_gauss), dim=0)
        real_data = torch.cat((real_data,real_data_v),dim=0)

    return total_rec_loss / N_batch_runs, total_disc_loss / N_batch_runs, total_gen_loss / N_batch_runs, generated_data.cpu().detach().numpy(), real_data.cpu().detach().numpy()
