"""
This file implements the smooth gating/split function including the linear
combination of features. If given, the features are sent through a non linear
module first, which may also be optimized thanks to autograd..
"""
import torch.nn as nn

class Gate(nn.Module):
  def __init__(self, input_size, initial_steepness, non_linear_module=None, polytope_sides=30):
    super(Gate, self).__init__()
    self.steepness = initial_steepness
    self.input_size = input_size

    # --- setup non-linear split feature module
    self.non_linear = None
    if non_linear_module is not None:
      self.non_linear = non_linear_module(polytope_sides=polytope_sides, num_features=input_size)
      self.input_size = self.non_linear.output_size

    # --- setup linear combination of features and sigmoid
    ### here we have normalization of the weights,check wether that norm remain one during training
    #self.linear = nn.Linear(self.input_size, 1)
    self.linear = nn.Linear(1, 1)
    norm = self.linear.weight.data.norm()
    self.sigmoid = nn.Sigmoid()

  def forward(self, X):
    if self.non_linear is not None:
      X = self.non_linear(X)
    #print(X.shape)

    gating_logits = self.linear(X)
    #print(gating_logits.shape)
    gating_weight = self.sigmoid(gating_logits * self.steepness)
    #print(gating_weight.shape)
    return gating_weight
