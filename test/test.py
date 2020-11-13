import torch
from main import main, networks
import networks
import main

scalar_tensor = torch.ones(1)

def test_losses():
   """ Test loss functions to make sure they give scalar tensors"""
   assert NKL_loss(scalar_tensor, scalar_tensor).size == scalar_tensor.size
   assert z_sample.size(scalar_tensor, 1) == scalar_tensor.size