import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Import the clique potentials from existing file
import clique_potentials as cp

def joint_potential(x):
    """
    Compute the joint potential (unnormalized probability) of the UGM.
    
    Parameters:
    -----------
    x : torch.Tensor
        Input tensor of shape [batch_size, 10]
        
    Returns:
    --------
    torch.Tensor
        Joint potential values of shape [batch_size]
    """
    # Compute individual clique potentials
    p_c1 = cp.C1_potential(x)
    p_c2 = cp.C2_potential(x)
    p_c3 = cp.C3_potential(x)
    p_c4 = cp.C4_potential(x)

    #check if any of the potentials are 0
    if p_c1 == 0 or p_c2 == 0 or p_c3 == 0 or p_c4 == 0:
        import pdb; pdb.set_trace()
    
    # Multiply potentials to get joint potential
    return p_c1 * p_c2 * p_c3 * p_c4

def log_joint_potential(x):
    """
    Compute the log of the joint potential.
    
    Parameters:
    -----------
    x : torch.Tensor
        Input tensor of shape [batch_size, 10]
        
    Returns:
    --------
    torch.Tensor
        Log joint potential values of shape [batch_size]
    """
    # Compute log of individual clique potentials
    log_p_c1 = torch.log(cp.C1_potential(x) + 1e-10)
    log_p_c2 = torch.log(cp.C2_potential(x) + 1e-10)
    log_p_c3 = torch.log(cp.C3_potential(x) + 1e-10)
    log_p_c4 = torch.log(cp.C4_potential(x) + 1e-10)
    
    #check if any of the potentials are inf 
    if torch.isinf(log_p_c1).any() or torch.isinf(log_p_c2).any() or torch.isinf(log_p_c3).any() or torch.isinf(log_p_c4).any():
        import pdb; pdb.set_trace()
        if torch.isinf(log_p_c1).any():
            ind = torch.where(torch.isinf(log_p_c1))
            x_inf = x[ind[0][0]].unsqueeze(0)
            p1 = cp.C1_potential(x_inf)
        elif torch.isinf(log_p_c2).any():
            ind = torch.where(torch.isinf(log_p_c2))
            x_inf = x[ind[0][0]].unsqueeze(0)
            p2 = cp.C2_potential(x_inf)
        elif torch.isinf(log_p_c3).any():
            ind = torch.where(torch.isinf(log_p_c3))
            x_inf = x[ind[0][0]].unsqueeze(0)
            p3 = cp.C3_potential(x_inf)
        elif torch.isinf(log_p_c4).any():
            ind = torch.where(torch.isinf(log_p_c4))
            x_inf = x[ind[0][0]].unsqueeze(0)
            p4 = cp.C4_potential(x_inf)
            

    
    # Sum logs to get log joint potential
    return log_p_c1 + log_p_c2 + log_p_c3 + log_p_c4

def negative_energy(x):
    """
    Compute the negative energy (log probability) of the UGM.
    
    Parameters:
    -----------
    x : torch.Tensor
        Input tensor of shape [batch_size, 10]
        
    Returns:
    --------
    torch.Tensor
        Negative energy values of shape [batch_size]
    """
    return log_joint_potential(x)

def energy(x):
    """
    Compute the energy (negative log probability) of the UGM.
    
    Parameters:
    -----------
    x : torch.Tensor
        Input tensor of shape [batch_size, 10]
        
    Returns:
    --------
    torch.Tensor
        Energy values of shape [batch_size]
    """
    return -log_joint_potential(x)

def clipped_energy(x, max_energy=100.0):
    """
    Compute the energy with safeguards against numerical instability.
    
    Parameters:
    -----------
    x : torch.Tensor
        Input tensor of shape [batch_size, 10]
    max_energy : float
        Maximum energy value to prevent explosions
        
    Returns:
    --------
    torch.Tensor
        Energy values of shape [batch_size], clipped to prevent numerical issues
    """
    # For numerical stability, we'll handle clique potentials separately
    
    # Compute log of individual clique potentials with safety checks
    log_p_c1 = torch.log(cp.C1_potential(x) + 1e-10)
    
    # For C2, we'll implement a more robust computation
    # The raw formula is exp(sum_i(x_i^4) - sum_i,j(x_i^2 * x_j^2))
    # We'll break this into more numerically stable components
    x7_10 = x[:, 6:]
    
    # Constrain extreme values to prevent explosions in the C2 potential
    x7_10_constrained = torch.clamp(x7_10, -5.0, 5.0)
    
    # Recompute C2 with constrained values
    x4_sum = torch.sum(x7_10_constrained**4, dim=1)
    x2_sum = 0
    for i in range(4):
        for j in range(i+1, 4):
            x2_sum += x7_10_constrained[:, i]**2 * x7_10_constrained[:, j]**2
    
    log_p_c2_raw = x4_sum - x2_sum
    # Clip to prevent extreme values
    log_p_c2 = torch.clamp(log_p_c2_raw, -max_energy, max_energy)
    
    log_p_c3 = torch.log(cp.C3_potential(x) + 1e-10)
    log_p_c4 = torch.log(cp.C4_potential(x) + 1e-10)
    
    # Sum logs to get log joint potential and clip extreme values
    log_joint = log_p_c1 + log_p_c2 + log_p_c3 + log_p_c4
    energy = -log_joint
    
    # Clip final energy to reasonable range
    return torch.clamp(energy, -max_energy, max_energy)

def langevin_dynamics(
    energy_fn,
    n_samples=10000,
    n_dims=10,
    n_steps=1000,
    step_size=0.01,
    initial_samples=None,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=True,
    max_grad_norm=1.0
):
    """
    Generate samples using Langevin dynamics.
    
    Parameters:
    -----------
    energy_fn : Callable
        Energy function that takes x and returns energy values
    n_samples : int
        Number of samples to generate
    n_dims : int
        Dimensionality of each sample
    n_steps : int
        Number of Langevin steps
    step_size : float
        Step size for Langevin dynamics
    initial_samples : torch.Tensor, optional
        Initial state for the chain, if None, random initialization is used
    device : str
        Device to run computations on ('cuda' or 'cpu')
    verbose : bool
        Whether to show progress bar
    max_grad_norm : float
        Maximum allowed gradient norm for gradient clipping
        
    Returns:
    --------
    torch.Tensor
        Generated samples of shape [n_samples, n_dims]
    list
        Energy values at each step for a random sample (for monitoring)
    """
    # Initialize samples
    if initial_samples is not None:
        samples = initial_samples.clone().to(device)
    else:
        samples = 2 * torch.rand(n_samples, n_dims, device=device) - 1
    
    # Track energy for a random sample to monitor convergence
    track_idx = np.random.randint(0, n_samples)
    energy_history = []
    
    # Run Langevin dynamics
    iterator = tqdm(range(n_steps)) if verbose else range(n_steps)
    for i in iterator:
        # Enable gradient computation
        samples.requires_grad_(True)
        
        # Compute energy and its gradient
        e = energy_fn(samples)
        if i % 100 == 0:
            with torch.no_grad():
                avg_energy = e.mean().item()
                if verbose:
                    iterator.set_description(f"Step {i}, Avg Energy: {avg_energy:.4f}")
        
        # Store energy of tracked sample
        with torch.no_grad():
            energy_history.append(e[track_idx].item())
        
        # Compute gradient
        grad = torch.autograd.grad(e.sum(), samples)[0]
        
        # Detach for next iteration
        samples = samples.detach()
        
        # Gradient clipping to prevent instability
        with torch.no_grad():
            grad_norm = torch.norm(grad, dim=1, keepdim=True)
            scaled_grad = grad * torch.clamp(max_grad_norm / (grad_norm + 1e-6), max=1.0)
        
        # Langevin update with clipped gradient
        noise = torch.randn_like(samples) * np.sqrt(2 * step_size)
        samples = samples - step_size * scaled_grad + noise
        
        # Constrain samples to reasonable range to ensure stability
        with torch.no_grad():
            samples = torch.clamp(samples, -10.0, 10.0)
        
    return samples, energy_history

def visualize_samples(samples, title="Langevin Dynamics Samples", figsize=(15, 10)):
    """
    Visualize the generated samples.
    
    Parameters:
    -----------
    samples : torch.Tensor or np.ndarray
        Generated samples
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    
    n_dims = samples.shape[1]
    
    fig, axes = plt.subplots(n_dims, 1, figsize=figsize)
    
    for i in range(n_dims):
        ax = axes[i]
        ax.hist(samples[:, i], bins=30, alpha=0.7)
        ax.set_xlabel(f'x{i+1}')
        ax.set_ylabel('Frequency')
        
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    return fig

def visualize_pairwise(samples, pairs=None, title="Pairwise Distributions", figsize=(15, 10)):
    """
    Visualize pairwise distributions of specified variable pairs.
    
    Parameters:
    -----------
    samples : torch.Tensor or np.ndarray
        Generated samples
    pairs : list of tuples, optional
        List of pairs of indices to visualize, if None, use clique pairs
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    
    if pairs is None:
        # Default to the pairs in C3 and C4 (0-indexed)
        pairs = [(4, 6), (5, 7)]
    
    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=figsize)
    
    if n_pairs == 1:
        axes = [axes]
    
    for i, (idx1, idx2) in enumerate(pairs):
        ax = axes[i]
        ax.scatter(samples[:, idx1], samples[:, idx2], alpha=0.3, s=1)
        ax.set_xlabel(f'x{idx1+1}')
        ax.set_ylabel(f'x{idx2+1}')
        ax.set_title(f'Variables x{idx1+1} and x{idx2+1}')
        
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    return fig

def plot_energy_history(energy_history, title="Energy During Langevin Dynamics", figsize=(10, 6)):
    """
    Plot the energy history during Langevin dynamics.
    
    Parameters:
    -----------
    energy_history : list
        Energy values at each step
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(energy_history)
    ax.set_xlabel('Step')
    ax.set_ylabel('Energy')
    ax.set_title(title)
    ax.grid(True)
    
    return fig


def compute_statistics(samples):
    """
    Compute basic statistics of the samples.
    
    Parameters:
    -----------
    samples : torch.Tensor or np.ndarray
        Generated samples
    
    Returns:
    --------
    dict
        Dictionary of statistics
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    
    stats = {
        "mean": np.mean(samples, axis=0),
        "std": np.std(samples, axis=0),
        "min": np.min(samples, axis=0),
        "max": np.max(samples, axis=0),
        "median": np.median(samples, axis=0)
    }
    
    return stats

def main():
    # Set up parameters
    n_samples = 100000  # Number of samples to generate
    n_dims = 10         # Number of dimensions (variables)
    n_steps = 10000     # Number of Langevin steps
    step_size = 0.001   # Step size for Langevin dynamics
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Running Langevin dynamics on {device}...")
    
    # Generate samples using Langevin dynamics
    start_time = time.time()
    samples, energy_history = langevin_dynamics(
        energy_fn=clipped_energy,  # Use the more stable energy function
        n_samples=n_samples,
        n_dims=n_dims,
        n_steps=n_steps,
        step_size=step_size,
        device=device,
        verbose=True,
        max_grad_norm=0.5  # Limit the gradient norm
    )
    end_time = time.time()
    
    print(f"Sampling completed in {end_time - start_time:.2f} seconds.")
    
    # Compute statistics
    stats = compute_statistics(samples)
    print("\nSample Statistics:")
    for stat_name, stat_values in stats.items():
        print(f"{stat_name}: {stat_values}")
    
    # Visualize samples
    samples_fig = visualize_samples(samples)
    
    # Visualize pairwise distributions
    pairs_fig = visualize_pairwise(samples)
    
    # Plot energy history
    energy_fig = plot_energy_history(energy_history)
        
    # Save results
    np.save("langevin_samples.npy", samples.cpu().numpy())
    
    # Save figures
    samples_fig.savefig("langevin_samples_histograms.png")
    pairs_fig.savefig("langevin_pairwise_distributions.png")
    energy_fig.savefig("langevin_energy_history.png")
    
    print("Results saved to files.")
    plt.show()

if __name__ == "__main__":
    main() 