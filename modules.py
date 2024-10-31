import torch
from torch import nn, Tensor
import math
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer
from torch.nn import functional as F
from typing import Optional

class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features, kernel_size=7):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.utils.spectral_norm(nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size))
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.leakyrelu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back

class FeedForwardEncoder(nn.Module):
    def __init__(self, dim, num_channels):
        super(FeedForwardEncoder, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(int(dim*num_channels), int(dim*num_channels)),
            nn.LeakyReLU(),
            nn.BatchNorm1d(int(dim*num_channels)),
            nn.Linear(int(dim*num_channels),int(dim*num_channels)),
            nn.Softmax()
        )

    def forward(self, x, xl):
        x_out = self.ff(x)
        x_out = x_out + xl
        return x_out

class FeedForwardDecoder(nn.Module):
    def __init__(self, dim, num_channels):
        super(FeedForwardDecoder, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim*num_channels, dim*num_channels),
            nn.LeakyReLU(),
            nn.BatchNorm1d(dim*num_channels),
            nn.Linear(dim*num_channels,dim*num_channels),
            nn.Softmax()
        )

    def forward(self, x, xl):
        x_out = self.ff(x)
        x_out = x_out + xl
        return x_out

class FeedForwardScore(nn.Module):
    def __init__(self, dim, num_channels):
        super(FeedForwardScore, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim*num_channels, dim*num_channels),
            nn.LeakyReLU(),
            nn.BatchNorm1d(dim * num_channels)
        )
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(dim*num_channels,1)

    def forward(self, x, residual):
        return self.sigmoid(self.linear(self.ff(x)+residual))

# Feature-oriented Graph attention network
# Proposed by Zhao et. al., 2020 (https://ieeexplore.ieee.org/abstract/document/9338317)
class FeatureAttentionLayer(nn.Module):
    """Single Graph Feature/Spatial Attention Layer
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky relu activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_bias: whether to include a bias term in the attention layer
    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_bias=True):
        super(FeatureAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else window_size
        self.num_nodes = n_features
        self.use_bias = use_bias

        #linear transformation is done after concatenation in GATv2
        self.embed_dim *= 2
        lin_input_dim = 2 * window_size
        a_input_dim = self.embed_dim


        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(n_features, n_features))
            nn.init.zeros_(self.bias.data)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For feature attention we represent a node as the values of a particular feature across all timestamps
        x = x.permute(0, 2, 1)

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        a_input = self._make_attention_input(x)                 # (b, k, k, 2*window_size)
        a_input = self.leakyrelu(self.lin(a_input))             # (b, k, k, embed_dim)
        e = torch.matmul(a_input, self.a).squeeze(3)            # (b, k, k, 1)


        if self.use_bias:
            e += self.bias

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        # Computing new node features using the attention
        h = self.sigmoid(torch.matmul(attention, x))

        return h.permute(0, 2, 1)

    def _make_attention_input(self, v):
        """Preparing the feature attention mechanism.
        Creating matrix with all possible combinations of concatenations of node.
        Each node consists of all values of that node within the window
            v1 || v1,
            ...
            v1 || vK,
            v2 || v1,
            ...
            v2 || vK,
            ...
            ...
            vK || v1,
            ...
            vK || vK,
        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)  # (b, K*K, 2*window_size)

        return combined.view(v.size(0), K, K, 2 * self.window_size)


class TemporalAttentionLayer(nn.Module):
    """Single Graph Temporal Attention Layer
    :param n_features: number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer

    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_bias=True):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias

        # Because linear transformation is performed after concatenation in GATv2
        self.embed_dim *= 2
        lin_input_dim = 2 * n_features
        a_input_dim = self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(window_size, window_size))
            nn.init.zeros_(self.bias.data)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For temporal attention a node is represented as all feature values at a specific timestamp

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu

        a_input = self._make_attention_input(x)              # (b, n, n, 2*n_features)
        a_input = self.leakyrelu(self.lin(a_input))          # (b, n, n, embed_dim)
        e = torch.matmul(a_input, self.a).squeeze(3)         # (b, n, n, 1)

        if self.use_bias:
            e += self.bias  # (b, n, n, 1)

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        h = self.sigmoid(torch.matmul(attention, x))    # (b, n, k)

        return h

    def _make_attention_input(self, v):
        """Preparing the temporal attention mechanism.
        Creating matrix with all possible combinations of concatenations of node values:
            (v1, v2..)_t1 || (v1, v2..)_t1
            (v1, v2..)_t1 || (v1, v2..)_t2

            ...
            ...

            (v1, v2..)_tn || (v1, v2..)_t1
            (v1, v2..)_tn || (v1, v2..)_t2

        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        return combined.view(v.size(0), K, K, 2 * self.n_features)


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

'''class Classifier(nn.Module):
    def __init__(self, input_size, num_units):
        super(Classifier, self).__init__()

        self.gru = nn.GRU(input_size=input_size, hidden_size=num_units, batch_first=True)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(num_units, 1)

    def forward(self, x, t):
        # Pass input through GRU cell
        #print('X:',x.shape)
        gru_output1, gru_last_state1 = self.gru(x)
        gru_output1 = self.tanh(gru_output1)
        # Use pack_padded_sequence to handle variable-length sequences
        packed_output = nn.utils.rnn.pack_padded_sequence(gru_output1, t, batch_first=True,enforce_sorted=False)
        # Pass pack_padded_sequence through GRU cell
        #gru_output2, gru_last_state2 = self.gru(packed_output.preprocessed_data.reshape(x.shape[0],x.shape[1],x.shape[2]))
        gru_output2, gru_last_state2 = self.gru(gru_output1)
        # Pass packed output through fully connected layer
        y_hat = self.fc(gru_last_state2)

        # Pass through sigmoid activation
        y_hat_logit = torch.sigmoid(y_hat)

        return y_hat_logit, y_hat'''

class Predictor(nn.Module):
    def __init__(self, input_size, num_units, seq_length, pred_length):
        super(Predictor, self).__init__()

        self.gru = nn.GRU(input_size=input_size, hidden_size=num_units, batch_first=True)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(seq_length, pred_length)

    def forward(self, x, t):
        # Pass input through GRU cell
        gru_output1, gru_last_state1 = self.gru(x)
        gru_output1 = self.tanh(gru_output1)
        # Use pack_padded_sequence to handle variable-length sequences
        packed_output = nn.utils.rnn.pack_padded_sequence(gru_output1, t, batch_first=True,enforce_sorted=False)
        # Pass pack_padded_sequence through GRU cell
        gru_output2, gru_last_state2 = self.gru(packed_output.data.reshape(x.shape[0],x.shape[1],x.shape[2]))
        # Pass packed output through fully connected layer
        gru_output2 = gru_output2.permute((0,2,1))
        y_hat = self.fc(gru_output2)
        #shape : batch_size x features x pred_window
        # Pass through sigmoid activation
        y_hat_logit = torch.sigmoid(y_hat)

        return y_hat_logit.permute(0,2,1), y_hat

class lstm_embedding(nn.Module):
    def __init__(self, input_size, num_units):
        super(lstm_embedding, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=num_units, batch_first=True)
        self.tanh = nn.Tanh()
    def forward(self,x):
        # Pass input through GRU cell
        lstm_output1, lstm_last_state1 = self.lstm(x)
        lstm_output1 = self.tanh(lstm_output1)
        lstm_output2, lstm_last_state2 = self.lstm(lstm_output1)
        return lstm_output2

# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    '''if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":'''
    return FixedPositionalEncoding

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))

class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src



class TSTransformerEncoder(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim*max_len, d_model*max_len)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)
        self.downsample = nn.Linear(d_model,int(d_model/2))
        self.feat_dim = feat_dim
        self.fc = nn.Linear(max_len, 1)
    def forward(self, X):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """
        inp = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        inp = self.project_inp(inp) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = inp.reshape(inp.shape[0], self.max_len, int(inp.shape[1] / self.max_len))
        inp = inp.permute(1, 0, 2) # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp)  # (seq_length, batch_size, d_model)
        output = self.downsample(output) #(seq_length,batch_size,d_model/2)
        output = output.permute(1, 2, 0)
        #output = output.reshape(output.shape[0],output.shape[1]*output.shape[2])
        y_hat = self.fc(output)
        # Pass through sigmoid activation
        y_hat_logit = torch.sigmoid(y_hat)

        return y_hat_logit,output.permute(0,2,1)