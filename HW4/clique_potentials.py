import numpy as np
import torch
from typing import Callable, List, Tuple, Union, Dict, Any


def C1_potential(x):
    """
    Potential function for clique C1 (nodes x1-x6)
    φC1(x1, x2, ..., x6) = exp(-[x1, x2, ..., x6] Σ^(-1)[x1, x2, ..., x6]^T)
    
    Args:
        x: Tensor of shape (batch_size, 10) containing the current state of the Markov Chain
        
    Returns:
        Tensor of shape (batch_size,) containing potential values
    """

    # Extract the values for nodes x1-x6 from the current state
    x1_6 = x[:, :6]
   
    # Define the covariance matrix Σ
    Sigma = torch.tensor([
        [2, 0.5, 0.2, 0.5, 0.5, 0.2],
        [0.5, 2, 0.8, 0.8, 0.5, 0.5],
        [0.2, 0.8, 1, 0.2, 0.1, 0.1],
        [0.5, 0.8, 0.2, 1, 0.5, 0.5],
        [0.5, 0.5, 0.1, 0.5, 2, 0.8],
        [0.2, 0.5, 0.1, 0.5, 0.8, 2]
    ], dtype=x.dtype, device=x.device)
    
    # Compute Σ^(-1)
    Sigma_inv = torch.inverse(Sigma)
    
    # Compute quadratic form: [x1, x2, ..., x6] Σ^(-1)[x1, x2, ..., x6]^T
    quad_form = torch.sum(torch.matmul(x1_6, Sigma_inv) * x1_6, dim=1)

    return torch.exp(-quad_form)

def C2_potential(x):
    """
    Potential function for clique C2 (nodes x7-x10)
    φC2(x7, x8, x9, x10) = exp(Σ(xi^4) - Σ(xi^2 xj^2)) for i=7 to 10 and 7≤i<j≤10
    
    Args:
        x: Tensor of shape (batch_size, 10) containing the current state of the Markov Chain

    Returns:
        Tensor of shape (batch_size,) containing potential values
    """
    # Extract the values for nodes x7-x10 from the current state
    x7_10 = x[:, 6:]

    # Compute sum of x^4 terms
    x4_sum = torch.sum(x7_10**4, dim=1)
    
    # Compute sum of x^2 * y^2 terms for all pairs
    x2_sum = 0
    for i in range(4):
        for j in range(i+1, 4):
            x2_sum += x7_10[:, i]**2 * x7_10[:, j]**2
    
    return torch.exp(x4_sum - x2_sum)

def C3_potential(x):
    """
    Potential function for clique C3 (nodes x5, x7)
    φC3(x5, x7) = exp(-x5^4 x7^6)
    
    Args:
        x: Tensor of shape (batch_size, 10) containing the current state of the Markov Chain
        
    Returns:
        Tensor of shape (batch_size,) containing potential values
    """
    # Extract the values for nodes x5 and x7 from the current state
    x5 = x[:, 4]
    x7 = x[:, 6]

    return torch.exp(-x5**4 * x7**6)

def C4_potential(x):
    """
    Potential function for clique C4 (nodes x6, x8)
    φC4(x6, x8) = exp(-x6^2 x8^8)
    
    Args:
        x: Tensor of shape (batch_size, 10) containing the current state of the Markov Chain
        
    Returns:
        Tensor of shape (batch_size,) containing potential values
    """
    # Extract the values for nodes x6 and x8 from the current state
    x6 = x[:, 5]
    x8 = x[:, 7]

    return torch.exp(-x6**2 * x8**8)

def log_joint(potentials: List[Callable[[torch.Tensor], torch.Tensor]]):
    """
    Log joint probability of a list of potential functions
    """
    def log_joint_fn(x):
        return sum(torch.log(potential(x)) for potential in potentials)
    return log_joint_fn


def conditional_per_variable(var_idx: int):
    """
    Create a conditional distribution for a given variable
    """
    
    if var_idx in [0, 1, 2, 3]:
        return log_joint([C1_potential])
    elif var_idx in [8, 9]:
        return log_joint([C2_potential])
    elif var_idx == 4:
        return log_joint([C1_potential, C3_potential])
    elif var_idx == 5:
        return log_joint([C1_potential, C4_potential])
    elif var_idx == 6:
        return log_joint([C2_potential, C3_potential])
    elif var_idx == 7:
        return log_joint([C2_potential, C4_potential])
    else:
        raise ValueError(f"Variable index {var_idx} not in any clique")
    

def gaussian_sampler_C1(x):
    """
    Gaussian sampler for clique C1
    """
    #Extract the values for nodes x1-x6 from the current state
    x1_6 = x[:, :6]

    #Define the covariance matrix Σ
    Sigma = torch.tensor([
        [2, 0.5, 0.2, 0.5, 0.5, 0.2],
        [0.5, 2, 0.8, 0.8, 0.5, 0.5],
        [0.2, 0.8, 1, 0.2, 0.1, 0.1],
        [0.5, 0.8, 0.2, 1, 0.5, 0.5],
        [0.5, 0.5, 0.1, 0.5, 2, 0.8],
        [0.2, 0.5, 0.1, 0.5, 0.8, 2]
    ], dtype=x.dtype, device=x.device)
    
    #Compute Σ^(-1)
    Sigma_inv = torch.inverse(Sigma)

    #Compute the mean of the conditional distribution
    mean = torch.matmul(x1_6, Sigma_inv)

    #Compute the covariance matrix of the conditional distribution
    Sigma_cond = torch.inverse(Sigma_inv)

    #Sample from the conditional distribution
    samples = torch.distributions.multivariate_normal.MultivariateNormal(mean, Sigma_cond).sample()

    x_samples = torch.cat((samples, x[:, 6:]), dim=1)

    return x_samples

