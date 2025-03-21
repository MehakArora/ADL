import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx

class EdgeConv(MessagePassing):
    """Custom edge convolution for message passing in UGM."""
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='add')  # "Add" aggregation
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, x=x)
        
    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        edge_features = torch.cat([x_i, x_j], dim=1)  # [E, 2 * in_channels]
        return self.mlp(edge_features)


class CliquePotentialGNN(nn.Module):
    """GNN model for learning node embeddings in an UGM."""
    def __init__(self, num_nodes, hidden_dim=64, num_layers=3, conv_type='sage'):
        super(CliquePotentialGNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # Initial node embedding layer (maps node values to initial embeddings)
        self.node_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            if conv_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif conv_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            elif conv_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim))
            elif conv_type == 'edge':
                self.convs.append(EdgeConv(hidden_dim, hidden_dim))
            else:
                raise ValueError(f"Unknown convolution type: {conv_type}")
        
        # Output layer for node features
        self.node_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x, edge_index):
        """
        Forward pass of the GNN.
        
        Parameters:
        -----------
        x : torch.Tensor
            Node features, shape [num_nodes, 1]
        edge_index : torch.Tensor
            Graph connectivity, shape [2, num_edges]
            
        Returns:
        --------
        torch.Tensor
            Node embeddings, shape [num_nodes, hidden_dim]
        """
        # Initial node embeddings
        h = self.node_embedding(x.view(-1, 1))
        
        # Save the original embeddings
        h_orig = h.clone()
        
        # Message passing layers
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index)
            h = F.leaky_relu(h_new)
            
            # Add a residual connection after the first layer
            if i == 0:
                h = h + h_orig
            
            # Apply dropout during training
            if self.training:
                h = F.dropout(h, p=0.1)
                
        # Final node representations
        node_embeddings = self.node_out(h)
        
        return node_embeddings


class CliqueEnergy(nn.Module):
    """Neural network for estimating the energy of a clique."""
    def __init__(self, clique_size, hidden_dim=64, energy_dim=32):
        super(CliqueEnergy, self).__init__()
        
        input_dim = clique_size * hidden_dim
        
        # Different architecture based on clique size
        if clique_size >= 4:  # For larger cliques (C1, C2)
            self.energy_net = nn.Sequential(
                nn.Linear(input_dim, energy_dim * 2),
                nn.LeakyReLU(),
                nn.Linear(energy_dim * 2, energy_dim),
                nn.LeakyReLU(),
                nn.Linear(energy_dim, energy_dim // 2),
                nn.LeakyReLU(),
                nn.Linear(energy_dim // 2, 1)
            )
        else:  # For smaller cliques (C3, C4)
            self.energy_net = nn.Sequential(
                nn.Linear(input_dim, energy_dim),
                nn.LeakyReLU(),
                nn.Linear(energy_dim, energy_dim // 2),
                nn.LeakyReLU(),
                nn.Linear(energy_dim // 2, 1)
            )
            
    def forward(self, clique_embeddings):
        """
        Compute energy for a clique.
        
        Parameters:
        -----------
        clique_embeddings : torch.Tensor
            Node embeddings for the clique, shape [clique_size, hidden_dim] or [batch, clique_size, hidden_dim]
            
        Returns:
        --------
        torch.Tensor
            Energy scalar
        """
        # Flatten the embeddings
        batch_mode = len(clique_embeddings.shape) > 2
        
        if batch_mode:
            batch_size, clique_size, hidden_dim = clique_embeddings.shape
            flat_embeddings = clique_embeddings.reshape(batch_size, -1)
        else:
            flat_embeddings = clique_embeddings.reshape(1, -1)
        
        # Compute energy
        energy = self.energy_net(flat_embeddings)
        
        return energy


class UGMPotentialModel(nn.Module):
    """Complete model for learning potentials in an UGM."""
    def __init__(self, num_nodes, cliques, hidden_dim=64, gnn_layers=3, pot_hidden_dim=32, conv_type='sage'):
        super(UGMPotentialModel, self).__init__()
        
        self.num_nodes = num_nodes
        self.cliques = cliques
        
        # GNN for node embeddings
        self.gnn = CliquePotentialGNN(num_nodes, hidden_dim, gnn_layers, conv_type)
        
        # Clique potential networks - one for each clique
        self.clique_potentials = nn.ModuleList()
        for clique in cliques:
            # Size of input depends on number of nodes in the clique
            clique_size = len(clique)
            
            # MLP for this clique's potential
            energy_net = CliqueEnergy(clique_size, hidden_dim, pot_hidden_dim)
            self.clique_potentials.append(energy_net)
    
    def forward(self, x, edge_index):
        """
        Compute energy of the graphical model.
        
        Parameters:
        -----------
        x : torch.Tensor
            Node values, shape [batch_size, num_nodes]
        edge_index : torch.Tensor
            Graph connectivity, shape [2, num_edges]
            
        Returns:
        --------
        tuple
            (total_energy, clique_energies)
        """
        batch_size = x.shape[0]
        
        # Create batch of node features for all samples
        batch_x = x.reshape(batch_size * self.num_nodes, -1)  
        # Create batched edge_index by shifting indices for each graph
        batched_edge_index = torch.cat([edge_index + i*self.num_nodes for i in range(batch_size)], dim=1)
        # Process all at once
        all_embeddings = self.gnn(batch_x, batched_edge_index)
        # Reshape back to [batch_size, num_nodes, hidden_dim]
        node_embeddings = all_embeddings.reshape(batch_size, self.num_nodes, -1)
        
        # Compute potential for each clique
        all_clique_energies = []
        for j, clique in enumerate(self.cliques):
            # Extract embeddings for this clique
            clique_embs = torch.stack([node_embeddings[:, idx, :] for idx in clique], dim=1)
            
            # Compute energy for this clique
            energy = self.clique_potentials[j](clique_embs)
            all_clique_energies.append(energy)
        
        # Combine energies from all cliques
        stacked_energies = torch.cat(all_clique_energies, dim=1)  # [batch_size, num_cliques]
        total_energy = torch.sum(stacked_energies, dim=1)  # [batch_size]
        
        return total_energy, all_clique_energies


def prepare_graph_from_cliques(num_nodes, cliques):
    """
    Create a graph structure from clique definitions.
    
    Parameters:
    -----------
    num_nodes : int
        Total number of nodes (variables)
    cliques : List[List[int]]
        List of cliques, each containing node indices
        
    Returns:
    --------
    torch.Tensor
        Edge index tensor
    """
    edges = set()
    
    # Add edges between all nodes in each clique
    for clique in cliques:
        for i in range(len(clique)):
            for j in range(i+1, len(clique)):
                # Add both directions for undirected graph
                edges.add((clique[i], clique[j]))
                edges.add((clique[j], clique[i]))
    
    # Convert to tensor
    edge_list = torch.tensor(list(edges), dtype=torch.long).t()
    
    return edge_list


def create_networkx_graph(num_nodes, edge_index, cliques):
    """
    Create a NetworkX graph for visualization.
    
    Parameters:
    -----------
    num_nodes : int
        Number of nodes
    edge_index : torch.Tensor
        Edge connectivity
    cliques : List[List[int]]
        List of cliques
        
    Returns:
    --------
    networkx.Graph
        Graph representation
    """
    G = nx.Graph()
    
    # Add nodes
    for i in range(num_nodes):
        G.add_node(i, name=f"x{i+1}")
    
    # Add edges
    edges = edge_index.t().tolist()
    for src, dst in edges:
        G.add_edge(src, dst)
    
    # Add clique information
    for i, clique in enumerate(cliques):
        clique_name = f"C{i+1}"
        for node in clique:
            if 'cliques' not in G.nodes[node]:
                G.nodes[node]['cliques'] = []
            G.nodes[node]['cliques'].append(clique_name)
    
    return G


def visualize_graph(G, figsize=(10, 8)):
    """
    Visualize the graph structure.
    
    Parameters:
    -----------
    G : networkx.Graph
        Graph to visualize
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    plt.figure(figsize=figsize)
    
    # Node positions using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Node colors based on cliques
    node_colors = []
    for node in G.nodes():
        if 'cliques' in G.nodes[node]:
            # Color based on the first clique the node belongs to
            clique_id = int(G.nodes[node]['cliques'][0][1]) - 1
            node_colors.append(clique_id)
        else:
            node_colors.append(0)
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color=node_colors, 
            cmap=plt.cm.viridis, node_size=500, font_size=10,
            font_weight='bold', edge_color='gray', width=1.0)
    
    plt.title("UGM Graph Structure")
    
    return plt.gcf()


def train_ugm_gnn(samples, cliques, lr=0.001, batch_size=64, epochs=300, 
                  weight_decay=1e-5, verbose=True, conv_type='sage', 
                  hidden_dim=64, gnn_layers=3, noise_type='gaussian',
                  noise_scale=0.5, clip_grad=1.0):
    """
    Train a GNN to learn potentials for the UGM.
    
    Parameters:
    -----------
    samples : np.ndarray
        Samples from the UGM, shape (n_samples, n_variables)
    cliques : List[List[int]]
        List of cliques, each containing variable indices
    lr : float
        Learning rate
    batch_size : int
        Batch size
    epochs : int
        Number of training epochs
    weight_decay : float
        L2 regularization strength
    verbose : bool
        Whether to show progress
    conv_type : str
        Type of convolution to use ('gcn', 'sage', 'gat', 'edge')
    hidden_dim : int
        Dimension of hidden layers
    gnn_layers : int
        Number of GNN layers
    noise_type : str
        Type of noise for contrastive learning ('gaussian', 'uniform', 'shuffle')
    noise_scale : float
        Scale of the noise for contrastive learning
    clip_grad : float
        Gradient clipping value
        
    Returns:
    --------
    tuple
        (model, losses, edge_index)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert samples to PyTorch tensors
    samples_tensor = torch.FloatTensor(samples)
    
    # Prepare data loader
    dataset = torch.utils.data.TensorDataset(samples_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create graph structure
    num_nodes = samples.shape[1]
    edge_index = prepare_graph_from_cliques(num_nodes, cliques)
    edge_index = edge_index.to(device)
    
    # Create model
    model = UGMPotentialModel(num_nodes, cliques, hidden_dim, gnn_layers, 
                              hidden_dim//2, conv_type).to(device)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                    factor=0.5, patience=20,
                                                    verbose=verbose)
    
    # Training loop
    losses = []
    
    iterator = tqdm(range(epochs)) if verbose else range(epochs)
    for epoch in iterator:
        model.train()
        epoch_losses = []
        
        for batch_idx, (batch_data,) in enumerate(dataloader):
            batch_data = batch_data.to(device)
            
            # Generate noise samples based on specified noise type
            if noise_type == 'gaussian':
                noise_samples = batch_data + noise_scale * torch.randn_like(batch_data).to(device)
            elif noise_type == 'uniform':
                noise_samples = batch_data + noise_scale * (torch.rand_like(batch_data) * 2 - 1).to(device)
            elif noise_type == 'shuffle':
                # Shuffle each variable independently
                noise_samples = batch_data.clone()
                for i in range(noise_samples.shape[1]):
                    idx = torch.randperm(noise_samples.shape[0])
                    noise_samples[:, i] = noise_samples[idx, i]
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")
            
            # Forward pass for real data
            real_energies, real_clique_energies = model(batch_data, edge_index)
            
            # Forward pass for noise data
            noise_energies, noise_clique_energies = model(noise_samples, edge_index)
            
            # NCE loss: push down energy of real data, push up energy of noise
            loss = -torch.mean(real_energies) + torch.mean(noise_energies)
            
            # Add regularization to encourage balanced contributions from each clique
            clique_balance_reg = 0.0
            for i in range(len(real_clique_energies)):
                for j in range(i+1, len(real_clique_energies)):
                    clique_balance_reg += torch.abs(
                        torch.mean(real_clique_energies[i]) - torch.mean(real_clique_energies[j])
                    )
            
            loss += 0.01 * clique_balance_reg
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Average loss for this epoch
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        if verbose and (epoch + 1) % 20 == 0:
            iterator.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model, losses, edge_index


def visualize_clique_potentials(model, edge_index, cliques, grid_size=50, range_min=-3, range_max=3):
    """
    Visualize the learned potentials for all cliques.
    
    Parameters:
    -----------
    model : UGMPotentialModel
        Trained model
    edge_index : torch.Tensor
        Graph connectivity
    cliques : List[List[int]]
        List of cliques
    grid_size : int
        Size of the evaluation grid
    range_min, range_max : float
        Range of values to evaluate
        
    Returns:
    --------
    dict
        Dictionary of figures
    """
    device = model.gnn.node_embedding[0].weight.device
    model.eval()
    figures = {}
    
    # Create a tensor with the original clique indices for reference
    clique_indices = {tuple(sorted(c)): i for i, c in enumerate(cliques)}
    
    for c_idx, clique in enumerate(cliques):
        if len(clique) == 2:  # We only visualize 2D potentials
            # Get the variable indices in the clique
            i, j = clique
            
            # Create a grid of values to evaluate
            x_vals = torch.linspace(range_min, range_max, grid_size)
            y_vals = torch.linspace(range_min, range_max, grid_size)
            xx, yy = torch.meshgrid(x_vals, y_vals, indexing='ij')
            grid = torch.stack([xx, yy], dim=-1)  # [grid_size, grid_size, 2]
            
            # Evaluate the energy function across the grid
            energies = torch.zeros(grid_size, grid_size)
            
            for i_idx in range(grid_size):
                for j_idx in range(grid_size):
                    # Create a sample with the grid values for the clique nodes
                    x_sample = torch.zeros(1, model.num_nodes, device=device)
                    x_sample[0, i] = grid[i_idx, j_idx, 0]
                    x_sample[0, j] = grid[i_idx, j_idx, 1]
                    
                    # Compute the energy
                    with torch.no_grad():
                        energy, clique_energies = model(x_sample, edge_index)
                        # We only want the energy for this specific clique
                        energies[i_idx, j_idx] = clique_energies[c_idx].item()
            
            # Plot the potential function
            fig, ax = plt.subplots(figsize=(8, 6))
            c = ax.pcolormesh(xx.cpu().numpy(), yy.cpu().numpy(), energies.cpu().numpy(), 
                             shading='auto', cmap='viridis')
            fig.colorbar(c, ax=ax)
            ax.set_xlabel(f"x{i+1}")
            ax.set_ylabel(f"x{j+1}")
            ax.set_title(f"Potential for Clique C{c_idx+1}")
            
            figures[f"clique_{c_idx+1}"] = fig
    
    return figures


def generate_samples_from_model(model, edge_index, num_samples=1000, num_steps=1000, step_size=0.1, max_grad_norm=1.0):
    """
    Generate samples from the learned model using Langevin dynamics.
    
    Parameters:
    -----------
    model : UGMPotentialModel
        Trained model
    edge_index : torch.Tensor
        Graph connectivity
    num_samples : int
        Number of samples to generate
    num_steps : int
        Number of Langevin steps
    step_size : float
        Step size for Langevin dynamics
    max_grad_norm : float
        Maximum gradient norm for clipping
        
    Returns:
    --------
    torch.Tensor
        Generated samples
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Initialize random samples
    samples = torch.randn(num_samples, model.num_nodes).to(device)
    
    # Run Langevin dynamics
    for step in tqdm(range(num_steps), desc="Generating samples"):
        # Add noise
        noise = torch.randn_like(samples) * np.sqrt(2 * step_size)
        
        # Compute energy gradient
        samples.requires_grad_(True)
        energy, _ = model(samples, edge_index)
        total_energy = torch.sum(energy)
        
        # Compute gradients
        grad = torch.autograd.grad(total_energy, samples)[0]
        
        # Update samples using clipped gradients
        with torch.no_grad():
            # Apply gradient clipping
            grad_norm = torch.norm(grad, dim=1, keepdim=True)
            scaled_grad = grad * torch.clamp(max_grad_norm / (grad_norm + 1e-6), max=1.0)
            
            # Update samples
            samples = samples - step_size * scaled_grad + noise
            
            # Optional: Constrain samples to reasonable range
            samples = torch.clamp(samples, -10.0, 10.0)
    
    return samples


def evaluate_model(model, edge_index, true_samples, generated_samples=None, num_gen_samples=1000):
    """
    Evaluate the model by comparing true and generated samples.
    
    Parameters:
    -----------
    model : UGMPotentialModel
        Trained model
    edge_index : torch.Tensor
        Graph connectivity
    true_samples : torch.Tensor
        True samples from the data
    generated_samples : torch.Tensor, optional
        Generated samples, if None will generate new ones
    num_gen_samples : int
        Number of samples to generate if generated_samples is None
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Convert true samples to tensor
    if isinstance(true_samples, np.ndarray):
        true_samples = torch.FloatTensor(true_samples)
    
    true_samples = true_samples.to(device)
    
    # Generate samples if not provided
    if generated_samples is None:
        generated_samples = generate_samples_from_model(
            model, edge_index, num_samples=num_gen_samples)
    
    generated_samples = generated_samples.to(device)
    
    # Compute energies
    with torch.no_grad():
        true_energies, true_clique_energies = model(true_samples, edge_index)
        gen_energies, gen_clique_energies = model(generated_samples, edge_index)
        
        # Compute average energies
        true_energy_mean = true_energies.mean().item()
        gen_energy_mean = gen_energies.mean().item()
        
        # Compute energy gap
        energy_gap = np.abs(true_energy_mean - gen_energy_mean)
        
        # Compare mean and variance of each variable
        true_means = true_samples.mean(dim=0).cpu().numpy()
        gen_means = generated_samples.mean(dim=0).cpu().numpy()
        
        true_vars = true_samples.var(dim=0).cpu().numpy()
        gen_vars = generated_samples.var(dim=0).cpu().numpy()
        
        mean_error = np.mean(np.abs(true_means - gen_means))
        var_error = np.mean(np.abs(true_vars - gen_vars))
    
    return {
        "true_energy_mean": true_energy_mean,
        "gen_energy_mean": gen_energy_mean,
        "energy_gap": energy_gap,
        "mean_error": mean_error,
        "var_error": var_error,
        "true_means": true_means,
        "gen_means": gen_means,
        "true_vars": true_vars,
        "gen_vars": gen_vars
    }


def visualize_marginals(true_samples, generated_samples, figsize=(15, 10)):
    """
    Visualize marginal distributions of each variable.
    
    Parameters:
    -----------
    true_samples : torch.Tensor or np.ndarray
        True samples from the data
    generated_samples : torch.Tensor or np.ndarray
        Generated samples from the model
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Convert to numpy if tensors
    if isinstance(true_samples, torch.Tensor):
        true_samples = true_samples.cpu().numpy()
    if isinstance(generated_samples, torch.Tensor):
        generated_samples = generated_samples.cpu().numpy()
    
    n_vars = true_samples.shape[1]
    
    fig, axes = plt.subplots(n_vars, 1, figsize=figsize)
    
    for i in range(n_vars):
        ax = axes[i]
        # Plot histograms of true and generated samples
        ax.hist(true_samples[:, i], bins=30, alpha=0.5, label='True samples')
        ax.hist(generated_samples[:, i], bins=30, alpha=0.5, label='Generated samples')
        ax.set_xlabel(f'Variable x{i+1}')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    plt.tight_layout()
    return fig


def train_and_evaluate_ugm():
    """
    Complete pipeline: train model, visualize results, and evaluate.
    
    Returns:
    --------
    dict
        Results including model, losses, and visualizations
    """
    # Load samples
    samples = np.load("langevin_samples.npy")
    
    # Check if we need more samples
    if samples.shape[0] < 1000:
        print(f"Warning: Only {samples.shape[0]} samples available. Results may be limited.")
    
    # Define clique structure (0-indexed)
    cliques = [
        [0, 1, 2, 3, 4, 5],  # C1: x1 to x6
        [6, 7, 8, 9],        # C2: x7 to x10
        [4, 6],              # C3: x5, x7
        [5, 7]               # C4: x6, x8
    ]
    
    # Visualize the graph structure
    edge_index = prepare_graph_from_cliques(10, cliques)
    G = create_networkx_graph(10, edge_index, cliques)
    graph_viz = visualize_graph(G)
    
    # Train model with appropriate hyperparameters
    model, losses, edge_index = train_ugm_gnn(
        samples, 
        cliques, 
        lr=0.001,
        batch_size=min(64, max(16, samples.shape[0] // 10)),  # Adjust batch size based on sample count
        epochs=300,
        weight_decay=1e-5,
        conv_type='sage',
        hidden_dim=64,
        gnn_layers=3,
        noise_type='gaussian',
        noise_scale=0.5
    )
    
    # Visualize training loss
    plt.figure(figsize=(10,10))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('training_loss.png')
    
    # Visualize learned potentials
    potential_viz = visualize_clique_potentials(model, edge_index, cliques)
    
    # Generate samples from the learned model
    num_gen_samples = min(10000, samples.shape[0] * 10)
    generated_samples = generate_samples_from_model(
        model, 
        edge_index, 
        num_samples=num_gen_samples, 
        num_steps=1000, 
        step_size=0.1
    )
    
    # Evaluate the model
    metrics = evaluate_model(model, edge_index, samples, generated_samples)
    
    # Visualize marginal distributions
    marginals_fig = visualize_marginals(samples, generated_samples)
    
    # Print evaluation metrics
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
    
    return {
        "model": model,
        "losses": losses,
        "edge_index": edge_index,
        "generated_samples": generated_samples,
        "metrics": metrics,
        "visualizations": {
            "graph": graph_viz,
            "loss": loss_fig,
            "potentials": potential_viz,
            "marginals": marginals_fig
        }
    }


if __name__ == "__main__":
    results = train_and_evaluate_ugm()
    
    # Save results
    torch.save(results["model"].state_dict(), "ugm_model.pth")
    np.save("generated_samples.npy", results["generated_samples"].numpy())
    
    # Save loss plot
    plt.figure(results["visualizations"]["loss"].number)
    plt.savefig("training_loss.png")
    
    # Save potential visualizations
    for name, fig in results["visualizations"]["potentials"].items():
        plt.figure(fig.number)
        plt.savefig(f"{name}_potential.png")
    
    # Save marginals plot
    plt.figure(results["visualizations"]["marginals"].number)
    plt.savefig("marginal_distributions.png")
    
    print("Training and evaluation complete. Results saved to files.")
