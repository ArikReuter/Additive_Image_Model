import torch
from torch import nn

class One_Layer_MLP(torch.nn.Module):
  def __init__(self,  n_input_units):

    super(One_Layer_MLP, self).__init__()

    self.n_input_units = n_input_units
    self.fc1 = torch.nn.Linear(n_input_units, 1)

  def forward(self, x):

    x = self.fc1(x)
    return x


class Linear_skip_block(nn.Module):
  """
  Block of linear layer + softplus + skip connection +  dropout  + batchnorm
  """
  def __init__(self, n_input, dropout_rate):
    super(Linear_skip_block, self).__init__()

    self.fc = nn.Linear(n_input, n_input)
    self.act = torch.nn.LeakyReLU()

    self.bn = nn.BatchNorm1d(n_input, affine = True)
    self.drop = nn.Dropout(dropout_rate)

  def forward(self, x):
    x0 = x
    x = self.fc(x)
    x = self.act(x)
    x = x0 + x
    x = self.drop(x)
    x = self.bn(x)

    return x

class Linear_block(nn.Module):
  """
  Block of linear layer dropout  + batchnorm
  """
  def __init__(self, n_input, n_output, dropout_rate):
    super(Linear_block, self).__init__()

    self.fc = nn.Linear(n_input, n_output)
    self.act = torch.nn.LeakyReLU()
    self.bn = nn.BatchNorm1d(n_output, affine = True)
    self.drop = nn.Dropout(dropout_rate)

  def forward(self, x):
    x = self.fc(x)
    x = self.act(x)
    x = self.drop(x)
    x = self.bn(x)

    return x

class MLP(nn.Module):
  def __init__(self, n_input_units, n_hidden_units, n_skip_layers, dropout_rate):

    super(MLP, self).__init__()
    self.n_input_units = n_input_units
    self.n_hidden_units = n_hidden_units
    self.n_skip_layers = n_skip_layers
    self.dropout_rate = dropout_rate

    self.linear1 = Linear_block(n_input_units, n_hidden_units, dropout_rate)    # initial linear layer
    self.hidden_layers = torch.nn.Sequential(*[Linear_skip_block(n_hidden_units, dropout_rate) for _ in range(n_skip_layers)])  #hidden skip-layers

    self.linear_final =  torch.nn.Linear(n_hidden_units, 1)

  def forward(self, x):
    x = self.linear1(x)
    x = self.hidden_layers(x)
    x = self.linear_final(x)

    return(x)

class CNN(nn.Module):
  def __init__(self, pretrained_model, mlp):
    super(CNN, self).__init__()

    self.pretrained_model = pretrained_model
    self.pretrained_model.requires_grad = False
    self.mlp = mlp

  def forward(self, x, f):
    with torch.no_grad():
      x = self.pretrained_model.encode(x)
    x = self.mlp(x)

    return x