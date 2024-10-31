import torch
import torch.nn as nn
import numpy as np
from modules import (
    ConvLayer,
    FeatureAttentionLayer,
    TemporalAttentionLayer,
    FeedForwardEncoder,
    lstm_embedding
)


class Generator(nn.Module):
    """ GAT-GAN generator model class.
    The generator is the encoder which maps the input time series to a latent space

    :param n_features: Number of input features
    :param window_size: Length of the input sequence
    :param out_dim: Number of features to output
    :param kernel_size: size of kernel to use in the 1-D convolution
    :param feat_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in feat-oriented GAT layer
    :param time_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in time-oriented GAT layer
    :param dropout: dropout rate
    :param alpha: negative slope used in the leaky relu activation function

    """

    def __init__(
        self,
        n_features,
        seq_length,
        fcn_dim,
        n_layers,
        kernel_size,
        feat_gat_embed_dim,
        time_gat_embed_dim,
        dropout,
        alpha
    ):
        super(Generator, self).__init__()
        self.seq_length = seq_length
        self.n_features = n_features
        self.conv = nn.ModuleList([ConvLayer(n_features, kernel_size) for i in range(n_layers)])
        self.conv1d = ConvLayer(n_features, kernel_size)
        self.lstm_embedding = lstm_embedding(n_features,n_features)
        self.feature_gat = nn.ModuleList(
            [FeatureAttentionLayer(n_features, seq_length, dropout, alpha, feat_gat_embed_dim) for i in
             range(n_layers)])
        self.temporal_gat = nn.ModuleList(
            [TemporalAttentionLayer(n_features, seq_length, dropout, alpha, time_gat_embed_dim) for i in
             range(n_layers)])
        self.ff = FeedForwardEncoder(fcn_dim, n_features)
        self.n_layers = n_layers
    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - sequence length, k - number of features
        residual = x
        for i in range(self.n_layers):
            #x = self.conv[i](x) #remove for ablation 3
            #context_x = x[:,-16:,:]
            #c_embeddings = self.lstm_embedding(context_x)
            #x = torch.cat((x,c_embeddings),1)
            x = self.feature_gat[i](x)  #remove for ablation 1
            x = self.temporal_gat[i](x) #remove for ablation2
            #x = self.conv[i](x) #remove for ablation 3
        residual = residual.reshape(residual.shape[0], -1)
        x = x.reshape(x.shape[0], -1)
        x = self.ff(x, residual)
        x = x.reshape(x.shape[0],self.seq_length,self.n_features)
        return x