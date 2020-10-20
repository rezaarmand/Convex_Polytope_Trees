import torch
import torch.nn as nn
import torch.nn.functional as F

class PolytopeLambda(nn.Module):
  def __init__(self, polytope_sides = 50, num_features = 2048):
    super().__init__()


    self.output_size = 1 
    self.num_features = num_features
    self.k_max = polytope_sides

    self.linear = nn.Linear(self.num_features, self.k_max)
    self.log_sigmoid = nn.LogSigmoid()

    self.dropout = nn.Dropout(p = 0.2)

  def forward(self, X):

    X = -X.view(-1, self.num_features)

    batch_size = X.shape[0]
    w = self.linear.weight
    w = w.view(1, w.shape[0], w.shape[1]).expand(batch_size, -1, -1)

    b = self.linear.bias
    b = b.view(1, b.shape[0]).expand(batch_size, -1)

    next_embed = torch.bmm(self.dropout(w), X.unsqueeze(-1)) + b.unsqueeze(-1)
    next_embed = next_embed.squeeze(-1)

    return next_embed.mean(dim=1).unsqueeze(-1)




