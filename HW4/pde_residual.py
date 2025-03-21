import torch

def a_func(x):
    """
    x: torch tensor of [BATCH_SIZE, dim, ..]
    """
    if len(x.shape) == 2:
        x = x.unsqueeze(2)
    return torch.bmm(x.transpose(1,2), x).squeeze(2)

def f_func(x):
    """
    x: torch tensor of [BATCH_SIZE, dim, ..]
    """
    if len(x.shape) == 2:
        x = x.unsqueeze(2)
    return torch.sin(torch.bmm(x.transpose(1,2), x).squeeze(2))

def pde_residuals(model, x_points):
    """
    Compute the residuals of the elliptic PDE: -∇·(a(x)∇u(x)) - f(x)
    
    Args:
        model: Neural network model that predicts u(x)
        x_points: tensor of shape [batch_size, 2], requires_grad should be True
    
    Returns:
        tensor of shape [batch_size, 1] containing PDE residuals at each point
    """
    # Ensure x requires gradients
    x_points.requires_grad_(True)
    
    # Get model prediction u(x)
    u = model(x_points)
    
    # Calculate coefficient function a(x) using a_func
    a_x = a_func(x_points)
    
    # Calculate source term f(x) using f_func
    f_x = f_func(x_points)
    
    # Initialize list to hold divergence terms
    divergence_terms = []
    
    # Compute ∇·(a(x)∇u) term by term for each dimension
    for i in range(x_points.shape[1]):
        # Compute ∂u/∂xᵢ 
        du_dxi = torch.autograd.grad(
            u, x_points, 
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0][:, i].reshape(-1, 1)
        
        # Compute a(x) * ∂u/∂xᵢ
        a_times_du_dxi = a_x * du_dxi
        
        # Compute ∂/∂xᵢ(a(x) * ∂u/∂xᵢ)
        div_term = torch.autograd.grad(
            a_times_du_dxi, x_points,
            grad_outputs=torch.ones_like(a_times_du_dxi),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0][:, i].reshape(-1, 1)
        
        divergence_terms.append(div_term)
    
    # Sum up the terms to get ∇·(a(x)∇u)
    div_a_grad_u = sum(divergence_terms)
    
    # Compute -∇·(a(x)∇u) - f(x)
    residuals = -div_a_grad_u - f_x
    
    return residuals

# Example usage:
if __name__ == "__main__":
    # Create a simple model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)
            
        def forward(self, x):
            return self.linear(x)
    
    # Create a model and test points
    model = SimpleModel()
    test_points = torch.rand(10, 2, requires_grad=True)
    
    # Compute residuals
    res = pde_residuals(model, test_points)
    print(f"Residuals shape: {res.shape}")
    print(f"Residuals: {res}") 