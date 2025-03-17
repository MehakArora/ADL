import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer import (
    TransformerModel, 
    extract_layer_intermediates,
    analyze_gradients_by_layer,
    inspect_model_state,
    example_forward_pass,
    train_with_gradient_tracking
)
from utils import (
    AttentionAnalyzer, 
    IntermediateActivationTracker, 
    GradientAnalyzer,
    analyze_transformer_debugging,
    visualize_all_attention_heads
)

def debug_transformer_during_training(debug = False):
    """
    Example showing how to debug a transformer model during training
    """
    # Create a small transformer model for demonstration
    en_vocab_size = 5000
    fr_vocab_size = 6000
    embed_dim = 256
    num_heads = 8
    sequence_len = 100
    num_layers = 2
    
    # Create model with debugging enabled
    model = TransformerModel(
        en_vocab_size=en_vocab_size,
        fr_vocab_size=fr_vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        sequence_len=sequence_len,
        num_layers=num_layers,
        debug=debug
    )
    
    # Register hooks for gradient tracking
    model.register_hooks()
    
    # Create sample data
    batch_size = 4
    src = torch.randint(1, en_vocab_size, (batch_size, sequence_len))
    tgt = torch.randint(1, fr_vocab_size, (batch_size, sequence_len))
    
    # Create analyzers
    attention_analyzer = AttentionAnalyzer(model)
    activation_tracker = IntermediateActivationTracker(model)
    gradient_analyzer = GradientAnalyzer(model)
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is padding
    
    # Training loop (for demonstration)
    for iteration in range(5):
        print(f"\n===== ITERATION {iteration+1} =====")
        
        # Forward pass
        outputs, attention_maps = model(src, tgt[:, :-1])  # shift right for teacher forcing
        
        # Calculate loss
        loss = criterion(
            outputs.contiguous().view(-1, outputs.size(-1)), 
            tgt[:, 1:].contiguous().view(-1)
        )
        
        print(f"Loss: {loss.item():.4f}")
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Capture data with analyzers
        attention_analyzer.capture()
        activation_tracker.capture()
        gradient_analyzer.capture()
        
        # Check for issues
        if iteration == 0 or iteration == 4:  # First and last iteration
            print("\nAttention Analysis:")
            attention_analyzer.identify_important_heads('encoder', 'entropy')
            
            print("\nActivation Analysis:")
            activation_tracker.identify_activation_issues()
            
            print("\nGradient Analysis:")
            gradient_analyzer.identify_gradient_issues()
        
        # Update weights
        optimizer.step()
    
    # Perform detailed analysis after training
    print("\n===== DETAILED ANALYSIS AFTER TRAINING =====")
    
    # Analyze intermediate matrices
    print("\nIntermediate Matrix Analysis:")
    intermediates = extract_layer_intermediates(model)
    
    # Analyze gradients by layer
    print("\nGradient Analysis by Layer:")
    analyze_gradients_by_layer(model, top_k=3)
    
    # Get complete model state
    print("\nModel State Inspection:")
    model_state = inspect_model_state(model)
    
    # Create visualizations
    print("\nCreating Visualizations...")
    
    # Visualize encoder attention patterns
    encoder_attn_fig = attention_analyzer.visualize_top_heads('encoder', 'entropy')
    plt.savefig('encoder_attention.png')
    plt.close()
    
    # Visualize decoder cross-attention
    decoder_cross_attn_fig = attention_analyzer.visualize_top_heads('decoder_cross', 'entropy')
    plt.savefig('decoder_cross_attention.png')
    plt.close()
    
    # Visualize activation stats across layers
    activation_fig = activation_tracker.compare_layer_activations('encoder', 'output', 'norm')
    plt.savefig('encoder_activations.png')
    plt.close()
    
    # Visualize gradient norms across layers
    gradient_fig = gradient_analyzer.compare_layer_gradients('encoder', 'norm1_weight', 'norm')
    plt.savefig('encoder_gradients.png')
    plt.close()
    
    print("Debugging completed. Visualizations saved to disk.")

def debug_transformer_one_forward_pass():
    """
    Example showing how to debug a transformer model with a single forward pass
    """
    # Create a small transformer model for demonstration
    en_vocab_size = 5000
    fr_vocab_size = 6000
    embed_dim = 256
    num_heads = 8
    sequence_len = 100
    num_layers = 2
    
    # Create model with debugging enabled
    model = TransformerModel(
        en_vocab_size=en_vocab_size,
        fr_vocab_size=fr_vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        sequence_len=sequence_len,
        num_layers=num_layers,
        debug=True
    )
    
    # Create sample data
    batch_size = 4
    src = torch.randint(1, en_vocab_size, (batch_size, sequence_len))
    tgt = torch.randint(1, fr_vocab_size, (batch_size, sequence_len))
    
    # Run comprehensive analysis
    results = analyze_transformer_debugging(model, src, tgt)
    
    # Print summary
    print(f"\nLoss: {results['loss']:.4f}")
    
    print("\nAttention Analysis:")
    if results['top_encoder_heads']:
        print(f"Found {len(results['top_encoder_heads'])} important encoder heads")
    
    print("\nActivation Issues:")
    if results['activation_issues']:
        print(f"Found {len(results['activation_issues'])} potential activation issues")
    else:
        print("No activation issues detected")
    
    print("\nGradient Issues:")
    if results['gradient_issues']:
        print(f"Found {len(results['gradient_issues'])} potential gradient issues")
    else:
        print("No gradient issues detected")
    
    # Save visualizations
    print("\nSaving visualizations...")
    for name, fig in results['figures'].items():
        fig.savefig(f"{name}.png")
        plt.close(fig)
    
    print("Debugging completed. Visualizations saved to disk.")

def debug_custom_example():
    """
    Example showing how to use the debugging tools with your own model 
    and visualization preferences
    """
    # Create model and data (replace with your own)
    model = TransformerModel(...)  # Your model here
    src = ...  # Your source data
    tgt = ...  # Your target data
    
    # Enable debugging and register hooks
    model.debug = True
    model.register_hooks()
    
    # Forward pass
    outputs, attention_maps = model(src, tgt)
    
    # Get complete model state
    model_state = inspect_model_state(model, detailed=True)
    
    # Extract and analyze intermediate matrices
    intermediates = extract_layer_intermediates(model, detailed=True)
    
    # Create loss and do backward pass for gradient analysis
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(...)  # Your loss calculation
    loss.backward()
    
    # Analyze gradients
    analyze_gradients_by_layer(model, top_k=5)
    
    # Create custom visualizations
    # Example: Visualize attention patterns for specific layer and head
    plt.figure(figsize=(12, 10))
    
    # Get encoder attention from layer 0, head 0
    encoder_attention = model_state['attention_maps']['encoder_attentions'][0][0, 0].detach().cpu().numpy()
    
    plt.subplot(2, 2, 1)
    plt.imshow(encoder_attention, cmap='viridis')
    plt.colorbar()
    plt.title('Encoder Self-Attention (Layer 1, Head 1)')
    
    # Get decoder cross-attention from layer 0, head 0
    decoder_cross_attention = model_state['attention_maps']['decoder_cross_attentions'][0][0, 0].detach().cpu().numpy()
    
    plt.subplot(2, 2, 2)
    plt.imshow(decoder_cross_attention, cmap='viridis')
    plt.colorbar()
    plt.title('Decoder Cross-Attention (Layer 1, Head 1)')
    
    # Create custom gradient analysis
    plt.subplot(2, 2, 3)
    # Example: Plot histogram of gradients for a specific parameter
    weight_grad = model.encoder.layers[0].self_attn.q_proj.weight.grad.flatten().detach().cpu().numpy()
    plt.hist(weight_grad, bins=50)
    plt.title('Gradient Distribution (Encoder Layer 1, Query Projection)')
    
    # Example: Plot norm of output from each encoder layer
    plt.subplot(2, 2, 4)
    norms = []
    for layer_idx, layer_data in enumerate(intermediates['encoder_layers'].values()):
        if 'output' in layer_data:
            output_tensor = layer_data['output']
            norm = torch.norm(output_tensor).item()
            norms.append(norm)
    
    plt.bar(range(1, len(norms) + 1), norms)
    plt.xlabel('Encoder Layer')
    plt.ylabel('Output Norm')
    plt.title('Encoder Layer Output Norms')
    
    plt.tight_layout()
    plt.savefig('custom_analysis.png')
    plt.close()
    
    print("Custom debugging completed. Visualization saved to disk.")

def train_with_gradient_tracking_example():
    """
    Example of using the train_with_gradient_tracking function
    """
    # Create a small transformer model for demonstration
    en_vocab_size = 5000
    fr_vocab_size = 6000
    embed_dim = 256
    num_heads = 8
    sequence_len = 100
    num_layers = 2
    
    # Create model with debugging enabled
    model = TransformerModel(
        en_vocab_size=en_vocab_size,
        fr_vocab_size=fr_vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        sequence_len=sequence_len,
        num_layers=num_layers,
        debug=True
    )
    
    # Create sample data
    batch_size = 32
    src = torch.randint(1, en_vocab_size, (batch_size, sequence_len))
    tgt = torch.randint(1, fr_vocab_size, (batch_size, sequence_len))
    
    # Create a simple dataset from the sample data
    dataset = torch.utils.data.TensorDataset(src, tgt)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Train the model with gradient tracking
    print("Starting training with gradient tracking...")
    results = train_with_gradient_tracking(model, train_loader, num_epochs=2, lr=0.0001)
    
    # Extract results
    gradient_tracker = results['gradient_tracker']
    attention_history = results['attention_history']
    gradient_history = results['gradient_history']
    
    # Create visualizations
    print("\nCreating visualizations from training history...")
    
    # Plot gradient norms over time
    norm_fig = gradient_tracker.plot_gradient_norms(top_k=5, last_n=None)
    plt.savefig('gradient_norms_over_time.png')
    plt.close()
    
    # Analyze final gradient state
    analyze_gradients_by_layer(model, top_k=3)
    
    # Analyze attention patterns from the last iteration
    last_attention = attention_history[-1]['patterns']
    
    # Visualize encoder self-attention
    if 'encoder' in last_attention:
        fig = visualize_all_attention_heads(
            last_attention['encoder'][0],  # First layer
            title="Final Encoder Layer 1 Attention Heads"
        )
        plt.savefig('final_encoder_attention.png')
        plt.close()
    
    # Visualize decoder cross-attention
    if 'decoder_cross' in last_attention:
        fig = visualize_all_attention_heads(
            last_attention['decoder_cross'][0],  # First layer
            title="Final Decoder Layer 1 Cross-Attention Heads"
        )
        plt.savefig('final_decoder_cross_attention.png')
        plt.close()
    
    print("Training and visualization completed.")