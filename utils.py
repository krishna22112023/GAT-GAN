from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os
"""
utils.py
(1) normalize_data : Returns min max normalized preprocessed_data
(2) get_data : To import the preprocessed_data
(3) visualization : To visualize PCA and t-SNE plots
(4) extract_time: Returns Maximum sequence length and each sequence length.
"""
def normalize_data(data):

    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals)

    return normalized_data


def get_data(dataset, input_path, normalize,seq_len):

    ori_data = np.loadtxt(input_path+dataset+'.csv', delimiter=",", skiprows=1)
    print('Input preprocessed_data loaded :',np.shape(ori_data))

    # Flip the preprocessed_data to make chronological preprocessed_data
    ori_data = ori_data[::-1]
    # Normalize the preprocessed_data
    if normalize:
        ori_data = normalize_data(ori_data)

    '''# Preprocess the dataset
    temp_data = []
    # Cut preprocessed_data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)'''

    total_samples = ori_data.shape[0]
    num_samples = int(total_samples / seq_len)
    total_samples = num_samples * seq_len
    temp_data = ori_data[:total_samples].reshape((num_samples, seq_len, ori_data.shape[1]))
    temp_data = temp_data[:, :, :3]
    print('preprocessed_data loaded successfuly : ',np.shape(temp_data))

    # shuffle the datasets (to make it similar to i.i.d)
    '''idx = np.random.permutation(len(temp_data))
    preprocessed_data = []
    for i in range(len(temp_data)):
        preprocessed_data.append(temp_data[idx[i]])'''
    return temp_data


def visualization(ori_data, generated_data, analysis, epoch, path):

    print('Creating visualization outputs')
    path = path +'epoch_'+str(epoch)+'/seq_'+str(ori_data.shape[1])+'/'
    os.makedirs(path, exist_ok=True)
    """Using PCA or tSNE for generated and original preprocessed_data visualization.
  
    Args:
      - ori_data: original preprocessed_data
      - generated_data: generated synthetic preprocessed_data
      - analysis: tsne or pca
    """
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]
    no, seq_len, dim = ori_data.shape

    for i in range(10):
        fig, ax = plt.subplots(dim)
        for j in range(dim):
            ax[j].plot(np.arange(seq_len), generated_data[i, :, j], label='generated_data')
            ax[j].plot(np.arange(seq_len), ori_data[i, :, j], label='original_data')
        plt.legend()
        plt.savefig(path+'TimeSeriesExample_' + str(i) + '.jpg')
        plt.close()

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat,
                                            np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(pca_results[:, 0], pca_results[:, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        ax.legend()
        plt.title('PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
        plt.xlim([-0.1,0.2])
        plt.ylim([-0.1,0.1])
        plt.savefig(path+'/PCA.jpg')
        plt.close()

    elif analysis == 'tsne':

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        ax.legend()

        plt.title('t-SNE plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
        plt.savefig(path+'t_SNE.jpg')
        plt.close()
        print('Visualization outputs saved')


def extract_time(data):
    """Returns Maximum sequence length and each sequence length.

    Args:
      - preprocessed_data: original preprocessed_data

    Returns:
      - time: extracted time information
      - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))
        time.append(len(data[i][:, 0]))

    return time, max_seq_len
