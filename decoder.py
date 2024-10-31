import torch.nn as nn
from modules import (FeatureAttentionLayer,
                     TemporalAttentionLayer,
                     ConvLayer,
                     FeedForwardDecoder)

class Decoder(nn.Module):
    """ GAT-GAN decoder model class.
    The decoder maps the input latent feature space back to its original feature space

    :param n_features: Number of input features
    :param window_size: Length of the input sequence
    :param out_dim: Number of features to output
    :param kernel_size: size of kernel to use in the 1-D convolution
    :param feat_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in feat-oriented GAT layer
    :param time_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in time-oriented GAT layer
    :param dropout: dropout rate
    :param alpha: negative slope used in the leaky rely activation function

    """

    def __init__(
        self,
        n_features,
        seq_length,
        fcn_dim,
        n_layers,
        kernel_size,
        feat_gat_embed_dim=None,
        time_gat_embed_dim=None,
        dropout=0.2,
        alpha=0.2
    ):
        super(Decoder, self).__init__()
        self.feature_gat = nn.ModuleList([FeatureAttentionLayer(n_features, seq_length, dropout, alpha, feat_gat_embed_dim) for i in range(n_layers)])
        self.temporal_gat = nn.ModuleList([TemporalAttentionLayer(n_features, seq_length, dropout, alpha, time_gat_embed_dim) for i in range(n_layers)])
        self.ff = FeedForwardDecoder(fcn_dim,n_features)
        self.conv1d = ConvLayer(n_features, kernel_size)
        self.conv = nn.ModuleList([ConvLayer(n_features, kernel_size) for i in range(n_layers)])
        self.n_layers = n_layers
        self.n_features = n_features
        self.seq_length = seq_length
    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - sequence length, k - number of features
        residual = x
        for i in range(self.n_layers):
            x = self.feature_gat[i](x) #remove for ablation 1
            x = self.temporal_gat[i](x) #remove for ablation 2
        x = x.reshape(x.shape[0], -1)
        residual = residual.reshape(residual.shape[0], -1)
        x = self.ff(x, residual)
        x = x.reshape(x.shape[0], self.seq_length, self.n_features)
        return x
