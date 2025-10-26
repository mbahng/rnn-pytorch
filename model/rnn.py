import torch
from torch import Tensor, Size
import torch.nn as nn

class Rnn(nn.Module): 

  def __init__(self, idim: Size, hdim: Size, odim: Size, max_length: int): 
    self.idim = idim 
    self.hdim = hdim 
    self.odim = odim 
    self.max_length = max_length

    self.U = nn.Linear(self.idim[0], self.hdim[0]) # input -> hidden
    self.V = nn.Linear(self.hdim[0], self.odim[0]) # hidden -> output 
    self.W = nn.Linear(self.hdim[0], self.hdim[0]) # hidden -> hidden 
    
    # store the hidden states
    hidden_states = torch.zeros((max_length, self.hdim[0]))
    self.register_buffer("hidden_states", hidden_states)

  def forward(self, x: Tensor): 
    B, L, D = x.size()

    # first update the hidden states 

    output = torch.zeros((self.max_length + 1, self.odim[0])) 
    output[0] = x
    for i in range(1, self.max_length + 1): 
      output[i] = self.U(output[i-1]) + self.V()
      
      

class UnidirectionalRNN(nn.Module): 
  """
  Unidirectional RNN
  """
  def __init__(self): 
    ...


class BiRNN(nn.Module): 
  """
  Bidirectional RNN
  """

