# The official PyTorch GAT-GAN implementation.

Authors: Srikrishna Iyer, Teng Teck Hou

Reference: Srikrishna Iyer, Teng Teck Hou, "GAT-GAN : A Graph-Attention-based Time-Series Generative Adversarial Network" , arXiv preprint arXiv:2306.01999
Submitted to : Neural Information Processing Systems (NeurIPS), 2023.

Paper Link: [GAT-GAN](https://arxiv.org/abs/2306.01999)

Contact: srikrish001@e.ntu.edu.sg

## Requirements

To setup the enviroment:

```setup
pip install -r requirements.txt
```

## Datasets

This repository contains implementations of the following real-world datasets.

- Broken rotor data (motor) : https://ieee-dataport.org/keywords/broken-rotor-bar-fault
- Traffic PEMS-SF : https://archive.ics.uci.edu/ml/datasets/PEMS-SF
- MIT-BIH ECH Arrhythmia : https://www.physionet.org/content/mitdb/1.0.0/

To run the pipeline for training and evaluation on GAT-GAN framwork, simply run python -m main.py

## Code explanation

(1) utils.py
- extract and preprocess data
- visualizations for t-SNE and PCA

(2) Metrics directory
  (a) frechet_transformer_distance.py
  - Proposed metric FTD to evaluate fidelity and diversity of synthetic data with FID score computed using Transformer
  (b) predictive_metrics.py
  - Use Post-hoc RNN to predict one-step ahead

(3) main.py
- Train, tune and evaluate the GAT-GAN model

(4) modules.py
- deep learning modules of GAT-GAN : 1D CNN, FCN, FeatureGAT, TemporalGAT, LSTM, Transformer encoder

(5) train.py
- GAT-GAN training function

(6) decoder.py, discriminator.py, generator.py
- implementation of GAT-GAN components 

## Command inputs:

-   data_set: motor, ecg, or traffic
-   save_metrics: save predictive score and FTD scores (note: Usually takes time)
-   normalize : normalize the dataset
-   seq_lenth: sequence length
-   n_blocks_gen: No. of repeatable blocks in the generator
-   n_blocks_disc: No. of repeatable blocks in the discriminator
-   feat_gat_embed_dim : embedding dim of feature-oriented GAT
-   time_gat_embed_dim : embedding dim of time-oriented GAT
-   dropout : Dropout rate for GAT-GAN
-   alpha : negative sloped used in leaky relu activation function for GAT
-   epochs: number of training iterations
-   bs: the number of samples in each batch
-   init_lr_gen : learning rate for generator
-   init_lr_disc : learning rate for discriminator
-   shuffle_dataset : random shuffle
-   use_cuda : use gpu 
-   metric_iterations: number of iterations for metric computation
-   use_trained_params : enabled hyperparameter tuning and uses the tuned results to train model

Note that network parameters should be optimized for different datasets.
Note that the above model can be used for custom datasets. make sure the --dataset argument matches the file name in /data directory

## Example command

```shell
$ python main.py --data_set motor --normalize True --seq_len 16 --n_blocks_gen 1
--n_blocks_disc 4 --dropout 0.2 --epochs 151 --bs 16 --init_lr_gen 0.0022
--use_cuda True --init_lr_disc 0.00079 --metric_iteration 10 use_trained_params False
```

## Outputs

-   real_data.csv: original data
-   gen_data.csv : generated synthetic data
-   results.txt: predictive 10 observation list , mean & std , frechet transformer distance 10 observation list, mean & std
-   visualization: actual vs real line plots, PCA and tSNE plots


