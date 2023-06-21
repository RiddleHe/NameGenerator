import torch
import torch.nn.functional as F

class Linear:

  def __init__(self, fan_in, fan_out, bias=True):
    self.weights = torch.randn((fan_in, fan_out), device=device)
    self.bias = None if not bias else torch.randn((fan_out), device=device)

  def __call__(self, x):
    self.out = x@self.weights
    if self.bias is not None:
      self.out += self.bias
    return self.out

  def parameters(self):
    return [self.weights] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:

  def __init__(self, dim, eps=1e-5, momentum=0.01):
    self.eps = eps
    self.momentum = momentum
    self.training = True
    # weights
    self.gamma = torch.ones((dim), device=device)
    self.beta = torch.zeros((dim), device=device)
    # buffer
    self.running_mean = torch.zeros((dim), device=device)
    self.running_var = torch.ones((dim), device=device)

  def __call__(self, x):
    if self.training:
      if x.ndim==2:
        dim = 0
      else:
        dim = (0,1)
      x_mean = x.mean(dim, keepdim=True)
      x_var = x.var(dim, keepdim=True)
    else:
      x_mean = self.running_mean
      x_var = self.running_var
    self.out = (x-x_mean)/torch.sqrt(x_var+self.eps)
    self.out = self.gamma*self.out+self.beta
    if self.training:
      self.running_mean = (1-self.momentum)*self.running_mean+self.momentum*x_mean
      self.running_var = (1-self.momentum)*self.running_var+self.momentum*x_var
    return self.out

  def parameters(self):
    return [self.gamma] + [self.beta]

class Tanh:

  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out

  def parameters(self):
    return []

class Embedding:

  def __init__(self, n_emb, n_dim):
    self.weight = torch.randn((n_emb, n_dim), device=device)

  def __call__(self, x):
    self.out = self.weight[x]
    return self.out

  def parameters(self):
    return [self.weight]

class Flatten:

  def __init__(self, n):
    self.n = n

  def __call__(self, x):
    B, T, C = x.shape
    x = x.view(B, T//self.n, C*self.n)
    if x.shape[1]==1:
      x = x.squeeze()
    self.out = x
    return self.out

  def parameters(self):
    return []

class Sequential:

  def __init__(self, layers):
    self.layers = layers

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    self.out = x
    return self.out

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]