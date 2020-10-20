import torch
from torch.autograd import Variable
import numpy as np

def compute_score(model, data_set, batch_size=1000, cuda=False, return_n_leafs=False):
  dtype=torch.FloatTensor
  if cuda:
    dtype = torch.cuda.FloatTensor

  # --- sequential loader
  data_loader = torch.utils.data.DataLoader(
                        data_set,
                        batch_size=batch_size,
                        shuffle=False)

  # --- test on data_loader
  correct = 0
  correct_discrete = 0
  first_time = 0
  for data, target in data_loader:
    data = Variable(data.type(dtype))
    target = Variable(target.type(dtype).long())
    # output = model(data)
    # pred = output.data.max(1, keepdim=True)[1]
    # correct += float(pred.eq(target.data.view_as(pred)).sum())
    # if (pred == pred[0]).all():
    #   print("all samples predicted to same class.")

    output = model(data, discrete=True)
    pred = output.data.max(1, keepdim=True)[1]
    correct_discrete += float(pred.eq(target.data.view_as(pred)).sum())
    if first_time<1:
      predicted_score_discrete = output.detach().numpy()
      first_time += 2
    else:
      predicted_score_discrete = np.concatenate((predicted_score_discrete, output.detach().numpy()), axis=0)
      #print('hi')

    
 # score = correct/len(data_loader.dataset)
  score_discrete = correct_discrete/len(data_loader.dataset)
  n_leafs = np.unique(predicted_score_discrete, axis=0).shape[0]
  print('num leafs', n_leafs)
  del predicted_score_discrete
  #print('num leafs', np.unique(predicted_score_discrete, axis=0))
  if return_n_leafs:
    return score_discrete, n_leafs
  return  score_discrete


