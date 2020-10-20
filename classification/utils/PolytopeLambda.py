import torch
import torch.nn as nn
import torch.nn.functional as F

class PolytopeLambda(nn.Module):
  def __init__(self, polytope_sides = 30, num_features = 2048):
    super().__init__()


    self.output_size = 1 
    self.num_features = num_features
    self.k_max = polytope_sides
    self.fp= nn.Sequential(nn.Linear(self.num_features, self.k_max), 
                              nn.LogSigmoid())
    #print(self.k_max)

  def forward(self, X):

    X = X.view(-1, self.num_features)

    return self.fp(-X).mean(dim=1).unsqueeze(-1)




