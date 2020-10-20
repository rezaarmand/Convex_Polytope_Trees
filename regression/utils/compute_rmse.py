import torch
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import mean_squared_error

def compute_rmse(model, data_set, batch_size=1000, cuda=False):
  dtype=torch.FloatTensor
  if cuda:
    dtype = torch.cuda.FloatTensor

  # --- sequential loader
  data_loader = torch.utils.data.DataLoader(
                        data_set,
                        batch_size=batch_size,
                        shuffle=False)

  # --- test on data_loader
  target_arr = np.empty((0), float)

  score_arr_discrete = np.empty((0), float)

  for i, (data, target) in enumerate(data_loader):

    target_arr = np.append(target_arr, target.detach().numpy(), axis = 0)

    ## maybe 1 in the following need to changed to 0, because model gives the prediction of all classes

    if model(data).shape[0] < 3:
      return 0.1, 0.1

    predicted_score_discrete = model(data, discrete=True).squeeze().detach().numpy()

    # print(predicted_score_discrete)
    # print(predicted_score_discrete.shape)
    score_arr_discrete = np.append(score_arr_discrete, predicted_score_discrete, axis = 0)

  ## maybe data type of target arr need to be changed to float for roc_auc function
  #print('now score arr')
  #print(score_arr_discrete)
  #print('now target')
  #print(target_arr)
  print('num leafs', np.unique(score_arr_discrete).shape)
  assert(target_arr.shape == score_arr_discrete.shape)

  
  #rmse_score_discrete = mean_squared_error(target_arr, score_arr_discrete)
  rmse_score_discrete = ((target_arr- score_arr_discrete)**2).mean()
  return np.sqrt(rmse_score_discrete)*1.9457662


