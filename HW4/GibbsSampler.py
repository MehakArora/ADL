import numpy as np
import torch
from typing import Callable, List, Tuple, Union, Dict, Any
import matplotlib.pyplot as plt
from tqdm import tqdm

def metropolis_hastings(
    log_target: Callable[[torch.Tensor], torch.Tensor],
    proposal_sampler: Callable[[torch.Tensor], torch.Tensor],
    initial_state: torch.Tensor,
    n_samples: int,
    burn_in: int = 1000,
    thinning: int = 1,
    verbose: bool = True,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> torch.Tensor:
    """
    General Metropolis-Hastings algorithm for sampling from a target distribution using PyTorch.
    Can leverage GPU acceleration if available.
    
    Parameters:
    -----------
    log_target : Callable
        Function to compute log of target distribution (can be unnormalized)
    proposal_sampler : Callable
        Function to sample from proposal distribution given current state
    initial_state : torch.Tensor
        Starting state for the chain
    n_samples : int
        Number of samples to generate
    burn_in : int
        Number of initial samples to discard
    thinning : int
        Keep every nth sample
    verbose : bool
        Whether to show progress bar
    device : str
        Device to run computations on ('cuda' or 'cpu')
        
    Returns:
    --------
    torch.Tensor
        Array of samples from the target distribution
    """
    # Setup
    current_state = initial_state.clone().to(device)
    current_log_prob = log_target(current_state)
    
    # Storage for samples
    total_iterations = burn_in + n_samples * thinning
    samples = torch.zeros((n_samples, initial_state.shape[1]), device=device)

    # Setup progress tracking
    iterator = tqdm(range(total_iterations)) if verbose else range(total_iterations)
    
    # Acceptance tracking
    accepts = 0
    
    # Run MCMC using Metropolis-Hastings
    for i in iterator:
        # Propose new state
        proposed_state = proposal_sampler(current_state)
        proposed_log_prob = log_target(proposed_state)
        
        # Calculate acceptance ratio
        # For symmetric proposals, the proposal terms cancel out
        log_alpha = proposed_log_prob - current_log_prob
        
        # Accept or reject
        if torch.log(torch.rand(1, device=device)) < log_alpha:
            current_state = proposed_state
            current_log_prob = proposed_log_prob
            accepts += 1
            
        # Store sample if past burn-in and thinning matches
        if i >= burn_in and (i - burn_in) % thinning == 0:
            samples[(i - burn_in) // thinning] = current_state
            
        # Update progress information
        if verbose and i % 500 == 0:
            iterator.set_description(f"Acceptance rate: {accepts/(i+1):.4f}")
    
    return samples


def gibbs_sampler(
    conditional_samplers: List[Callable[[torch.Tensor], torch.Tensor]],
    initial_state: torch.Tensor,
    n_samples: int,
    burn_in: int = 1000,
    thinning: int = 1,
    verbose: bool = True,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> torch.Tensor:
    """
    General Gibbs sampling algorithm for sampling from a joint distribution.
    
    Parameters:
    -----------
    conditional_samplers : List[Callable]
        List of functions, each sampling from the conditional distribution of one variable
        given all others. Each function should take the current state and return a new value
        for its corresponding variable.
    initial_state : torch.Tensor
        Starting state for the chain
    n_samples : int
        Number of samples to generate
    burn_in : int
        Number of initial samples to discard
    thinning : int
        Keep every nth sample
    verbose : bool
        Whether to show progress bar
    device : str
        Device to run computations on ('cuda' or 'cpu')
        
    Returns:
    --------
    torch.Tensor
        Array of samples from the joint distribution
    """
    # Setup
    n_vars = initial_state.shape[1]
    current_state = initial_state.clone().to(device)
    
    # Check consistency
    if len(conditional_samplers) != n_vars:
        raise ValueError(f"Expected {n_vars} conditional samplers, got {len(conditional_samplers)}")
    
    # Storage for samples
    total_iterations = burn_in + n_samples * thinning
    samples = torch.zeros((n_samples, n_vars), device=device)
    
    # Setup progress tracking
    iterator = tqdm(range(total_iterations)) if verbose else range(total_iterations)
    
    # Run Gibbs sampler
    for i in iterator:
        # Update each variable
        for j in range(n_vars):
            current_state[:, j] = conditional_samplers[j](current_state)[:, j]
            
        # Store sample if past burn-in and thinning matches
        if i >= burn_in and (i - burn_in) % thinning == 0:
            samples[(i - burn_in) // thinning] = current_state
    
    return samples


def gaussian_proposal_sampler(scale: float = 1.0, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Callable:
    """
    Creates a Gaussian proposal sampler with specified scale using PyTorch.
    
    Parameters:
    -----------
    scale : float or torch.Tensor
        Standard deviation of the Gaussian proposal. Can be a scalar or a tensor
        of the same dimension as the state.
    device : str
        Device to run computations on ('cuda' or 'cpu')
        
    Returns:
    --------
    Callable
        Function that takes current state and returns proposed state
    """
    def sampler(current_state: torch.Tensor) -> torch.Tensor:
        return current_state + scale * torch.randn_like(current_state, device=current_state.device)
    
    return sampler


def metropolis_hastings_sampler(log_target: Callable[[torch.Tensor], torch.Tensor], 
                               proposal_sampler: Callable[[torch.Tensor], torch.Tensor], 
                               initial_state: torch.Tensor, 
                               n_samples: int, 
                               burn_in: int = 1000, 
                               thinning: int = 1, 
                               verbose: bool = True, 
                               device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> torch.Tensor:
    """
    General Metropolis-Hastings sampler for sampling from a target distribution using PyTorch.
    Can leverage GPU acceleration if available.
    
    Parameters:
    -----------
    log_target : Callable
        Function to compute log of target distribution (can be unnormalized)
    proposal_sampler : Callable
        Function to sample from proposal distribution given current state
    initial_state : torch.Tensor
        Starting state for the chain
    n_samples : int
        Number of samples to generate
    burn_in : int
        Number of initial samples to discard
    thinning : int
        Keep every nth sample
    verbose : bool
        Whether to show progress bar
    device : str
        Device to run computations on ('cuda' or 'cpu')
        
    Returns:
    --------
    sampler_function: Callable[[torch.Tensor], torch.Tensor]
        Function that takes current state and returns proposed state
    """
    def sampler_function(current_state: torch.Tensor) -> torch.Tensor:
        samples = metropolis_hastings(log_target=log_target, proposal_sampler=proposal_sampler, initial_state=current_state, 
                                      n_samples=n_samples, burn_in=burn_in, thinning=thinning, verbose=verbose, device=device)
        return samples
    return sampler_function


def visualize_samples(samples, var_names=None, figsize=(15, 10)):
    """
    Visualize MCMC samples.
    
    Parameters:
    -----------
    samples : torch.Tensor
        Array of MCMC samples
    var_names : List[str], optional
        Names of variables
    figsize : Tuple[int, int], optional
        Figure size
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
        
    n_vars = samples.shape[1]
    
    if var_names is None:
        var_names = [f"x{i+1}" for i in range(n_vars)]
    
    # Create figure
    fig, axs = plt.subplots(n_vars, 2, figsize=figsize)
    
    for i in range(n_vars):
        # Trace plot
        axs[i, 0].plot(samples[:, i])
        axs[i, 0].set_title(f"{var_names[i]} trace")
        
        # Histogram
        axs[i, 1].hist(samples[:, i], bins=30, density=True)
        axs[i, 1].set_title(f"{var_names[i]} histogram")
    
    plt.tight_layout()
    return fig, axs




if __name__ == "__main__":
    import clique_potentials as cp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 1 
    initial_state = torch.randn(batch_size, 10).to(device)
    
    #Log conditionals per variable for Gibbs sampling
    log_conditional1 = cp.conditional_per_variable(0)
    log_conditional2 = cp.conditional_per_variable(1)
    log_conditional3 = cp.conditional_per_variable(2)
    log_conditional4 = cp.conditional_per_variable(3)
    log_conditional5 = cp.conditional_per_variable(4)
    log_conditional6 = cp.conditional_per_variable(5)
    log_conditional7 = cp.conditional_per_variable(6)
    log_conditional8 = cp.conditional_per_variable(7)
    log_conditional9 = cp.conditional_per_variable(8)
    log_conditional10 = cp.conditional_per_variable(9)
    
    #Proposal samplers for Metropolis-Hastings
    proposal_sampler = gaussian_proposal_sampler(0.3, device=device)

    #Samplers for Gibbs sampling

    n_samples = 1  
    burn_in = 1000
    thinning = 1
    verbose = False 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sampler1 = metropolis_hastings_sampler(log_target=log_conditional1, proposal_sampler=proposal_sampler, initial_state=initial_state, 
                                   n_samples=n_samples, 
                                   burn_in=burn_in, 
                                   thinning=thinning, 
                                   verbose=verbose, 
                                   device=device)
    sampler2 = metropolis_hastings_sampler(log_target=log_conditional2, proposal_sampler=proposal_sampler, initial_state=initial_state, 
                                   n_samples=n_samples, 
                                   burn_in=burn_in, 
                                   thinning=thinning, 
                                   verbose=verbose, 
                                   device=device)
    sampler3 = metropolis_hastings_sampler(log_target=log_conditional3, proposal_sampler=proposal_sampler, initial_state=initial_state, 
                                   n_samples=n_samples, 
                                   burn_in=burn_in, 
                                   thinning=thinning, 
                                   verbose=verbose, 
                                   device=device)
    sampler4 = metropolis_hastings_sampler(log_target=log_conditional4, proposal_sampler=proposal_sampler, initial_state=initial_state, 
                                   n_samples=n_samples, 
                                   burn_in=burn_in, 
                                   thinning=thinning, 
                                   verbose=verbose, 
                                   device=device)
    sampler5 = metropolis_hastings_sampler(log_target=log_conditional5, proposal_sampler=proposal_sampler, initial_state=initial_state, 
                                   n_samples=n_samples, 
                                   burn_in=burn_in, 
                                   thinning=thinning, 
                                   verbose=verbose, 
                                   device=device)
    sampler6 = metropolis_hastings_sampler(log_target=log_conditional6, proposal_sampler=proposal_sampler, initial_state=initial_state, 
                                   n_samples=n_samples, 
                                   burn_in=burn_in, 
                                   thinning=thinning, 
                                   verbose=verbose, 
                                   device=device)
    sampler7 = metropolis_hastings_sampler(log_target=log_conditional7, proposal_sampler=proposal_sampler, initial_state=initial_state, 
                                   n_samples=n_samples, 
                                   burn_in=burn_in, 
                                   thinning=thinning, 
                                   verbose=verbose, 
                                   device=device)
    sampler8 = metropolis_hastings_sampler(log_target=log_conditional8, proposal_sampler=proposal_sampler, initial_state=initial_state, 
                                   n_samples=n_samples, 
                                   burn_in=burn_in, 
                                   thinning=thinning, 
                                   verbose=verbose, 
                                   device=device)
    sampler9 = metropolis_hastings_sampler(log_target=log_conditional9, proposal_sampler=proposal_sampler, initial_state=initial_state, 
                                   n_samples=n_samples, 
                                   burn_in=burn_in, 
                                   thinning=thinning, 
                                   verbose=verbose, 
                                   device=device)
    sampler10 = metropolis_hastings_sampler(log_target=log_conditional10, proposal_sampler=proposal_sampler, initial_state=initial_state, 
                                   n_samples=n_samples, 
                                   burn_in=burn_in, 
                                   thinning=thinning, 
                                   verbose=verbose, 
                                   device=device)
    
    samplers = [sampler1, sampler2, sampler3, sampler4, sampler5, sampler6, sampler7, sampler8, sampler9, sampler10]
    log_conditionals = [log_conditional1, log_conditional2, log_conditional3, log_conditional4, log_conditional5, log_conditional6, log_conditional7, log_conditional8, log_conditional9, log_conditional10]

    burn_in = 1000
    thinning = 1
    verbose = True
    n_samples = 100000
    gibbs_samples = gibbs_sampler(conditional_samplers=samplers, 
                                  initial_state=initial_state, 
                                  n_samples=n_samples, 
                                  burn_in=burn_in, 
                                  thinning=thinning, 
                                  verbose=verbose, 
                                  device=device)
    

    print(gibbs_samples.shape)
    visualize_samples(gibbs_samples)

    #Save as numpy array
    np.save("gibbs_samplesall.npy", gibbs_samples.cpu().numpy())
