import torch
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import roc_auc_score

def compute_auc(model, data_set, batch_size=1000, cuda=False):
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

  for i, (data, target) in enumerate(data_loader):

    target_arr = np.append(target_arr, target.detach().numpy(), axis = 0)

    ## maybe 1 in the following need to changed to 0, because model gives the prediction of all classes



    if model(data).shape[0] < 3:
      return 0.1, 0.1



    predicted_score_discrete = model(data, discrete=True).detach().numpy()[:, 1]


    score_arr_discrete = np.append(score_arr_discrete, predicted_score_discrete, axis = 0)

  ## maybe data type of target arr need to be changed to float for roc_auc function

  auc_score_discrete = roc_auc_score(target_arr, score_arr_discrete)



  return auc_score_discrete


