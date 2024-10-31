"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

predictive_metrics.py

Note: Use Post-hoc RNN to predict one-step ahead (last feature)
"""

# Necessary Packages
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error
from metrics.utils import extract_time
from modules import Predictor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def norm_mean_absolute_error(ori_data,generated_data):
  mae = np.mean(np.abs(ori_data-generated_data))
  std = np.std(ori_data)
  mae_norm_sd = np.sqrt(mae)/std
  return mae_norm_sd

def predictive_score_metrics (ori_data, generated_data):

  """Report the performance of Post-hoc RNN one-step ahead prediction.
  
  Args:
    - ori_data: original preprocessed_data
    - generated_data: generated synthetic preprocessed_data
    
  Returns:
    - predictive_score: MAE of the predictions on the original preprocessed_data
  """
  prediction_window = 8
  # Basic Parameters
  no, seq_len, dim = np.asarray(ori_data).shape
    
  # Set maximum sequence length and each sequence length
  ori_time, ori_max_seq_len = extract_time(ori_data)
  generated_time, generated_max_seq_len = extract_time(ori_data)
  max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  
     
  ## Builde a post-hoc RNN predictive network 
  # Network parameters
  hidden_dim = dim
  iterations = 5000
  batch_size = 128
  init_lr = 0.001

  predictor = Predictor(num_units = hidden_dim, input_size = dim, seq_length = seq_len-prediction_window,pred_length=prediction_window).to(device).float()
  predictor.train()

  # Loss for the discriminator
  criterion = torch.nn.L1Loss()

  # optimizer
  p_optimizer = torch.optim.Adam(predictor.parameters(), lr=init_lr)

  p_loss = 0
  for itt in range(iterations):
    # Set mini-batch
    idx = np.random.permutation(len(generated_data))
    train_idx = idx[:batch_size]

    X_mb = np.array(list(generated_data[i][:-prediction_window, :] for i in train_idx))
    T_mb = np.array(list(generated_time[i] - prediction_window for i in train_idx))
    Y_mb = np.array(list(generated_data[i][-prediction_window:, :] for i in train_idx))

    X_mb , T_mb, Y_mb = torch.FloatTensor(X_mb).to(device), torch.FloatTensor(T_mb), torch.FloatTensor(Y_mb).to(device)
    y_logit,y_pred = predictor(X_mb,T_mb)
    p_loss = criterion(y_logit,Y_mb)

    p_optimizer.zero_grad()
    p_loss.backward()
    p_optimizer.step()
    #print('step: ' + str(itt) + '/' + str(iterations) + ', step_p_loss: ' + str(np.round(np.sqrt(p_loss.cpu().detach().numpy()), 4)))

  ## Test the trained model on the original preprocessed_data
  idx = np.random.permutation(len(ori_data))
  test_idx = idx[:no]

  X_mb = np.array(list(ori_data[i][:-prediction_window, :] for i in test_idx))
  T_mb = np.array(list(generated_time[i] - prediction_window for i in test_idx))
  Y_mb = np.array(list(ori_data[i][-prediction_window:, :] for i in test_idx))

  X_mb, T_mb = torch.FloatTensor(X_mb).to(device), torch.FloatTensor(T_mb)

  with torch.no_grad():
    # Set the model to evaluation mode
    predictor.eval()
    y_logit, y_pred = predictor(X_mb, T_mb)
    y_logit = y_logit.cpu().numpy()
  # Compute the performance in terms of MAE
  #y_logit = np.squeeze(y_logit,axis=-1)
  MAE_temp = 0
  for i in range(no):
    MAE_temp = MAE_temp + mean_absolute_error(Y_mb[i], y_logit[i])
    
  predictive_score = MAE_temp / no
    
  return predictive_score
    