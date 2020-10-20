import torch
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import roc_auc_score

def compute_auc(model, data_set, batch_size=1000, cuda=False, return_n_leafs=False):
  dtype=torch.FloatTensor
  if cuda:
    dtype = torch.cuda.FloatTensor

  # --- sequential loader
  data_loader = torch.utils.data.DataLoader(
                        data_set,
                        batch_size=batch_size,
                        shuffle=False)

  # --- test on data_loader
  target_arr = np.empty((0), int)
  score_arr_discrete = np.empty((0), float)
  first_time = 0

  for i, (data, target) in enumerate(data_loader):

    target_arr = np.append(target_arr, target.detach().numpy(), axis = 0)

    ## maybe 1 in the following need to changed to 0, because model gives the prediction of all classes



    if model(data).shape[0] < 3:
      return 0.1, 0.1



    predicted_score_discrete = model(data, discrete=True).detach().numpy()[:, 1]


    score_arr_discrete = np.append(score_arr_discrete, predicted_score_discrete, axis = 0)

    if first_time<1:
      stacked_predicted_score_discrete = predicted_score_discrete
      first_time += 2
    else:
      stacked_predicted_score_discrete = np.concatenate((stacked_predicted_score_discrete, predicted_score_discrete), axis=0)

  ## maybe data type of target arr need to be changed to float for roc_auc function
  n_leafs = np.unique(stacked_predicted_score_discrete, axis=0).shape[0]
  print('num leafs', n_leafs)
  del predicted_score_discrete, stacked_predicted_score_discrete
  #print(predicted_score_discrete[0:20])

  auc_score_discrete = roc_auc_score(target_arr, score_arr_discrete)



  if return_n_leafs:
    return auc_score_discrete, n_leafs

  return auc_score_discrete


