import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
import os
import time
from matplotlib.ticker import MaxNLocator
from IPython.display import display, HTML
import torch.nn as nn 

class AttentionAnalyzer:
    """
    A class to analyze attention patterns during training or inference
    
    This tool helps visualize and analyze how attention patterns evolve,
    identify important patterns, and detect potential issues.
    """
    def __init__(self, model):
        self.model = model
        self.attention_history = []
        self.iteration = 0
        self.current_attention = None
    
    def capture(self, sample_idx=0):
        """Capture current attention patterns from the model"""
        # Extract attention patterns from the model
        attention_maps = self.model.get_attention_maps()
        
        if attention_maps is None or 'model' not in attention_maps:
            print("No attention maps available to capture")
            return
        
        # Get model-level attention maps
        model_maps = attention_maps['model']
        
        # Store processed attention data
        attention_data = {}
        
        # Process encoder attention
        if 'encoder_attentions' in model_maps:
            encoder_attns = model_maps['encoder_attentions']
            # Store sample_idx for batch dimension
            attention_data['encoder'] = [attn[sample_idx].detach().cpu() for attn in encoder_attns]
        
        # Process decoder self-attention
        if 'decoder_self_attentions' in model_maps:
            decoder_self_attns = model_maps['decoder_self_attentions']
            attention_data['decoder_self'] = [attn[sample_idx].detach().cpu() for attn in decoder_self_attns]
        
        # Process decoder cross-attention
        if 'decoder_cross_attentions' in model_maps:
            decoder_cross_attns = model_maps['decoder_cross_attentions']
            attention_data['decoder_cross'] = [attn[sample_idx].detach().cpu() for attn in decoder_cross_attns]
        
        # Store current attention
        self.current_attention = attention_data
        
        # Add to history with metadata
        self.attention_history.append({
            'iteration': self.iteration,
            'timestamp': time.time(),
            'attention': attention_data
        })
        
        self.iteration += 1
        
        return attention_data
    
    def compute_entropy(self, attention_tensor):
        """Compute entropy of attention weights (measure of focus)"""
        # attention_tensor shape: [num_heads, seq_len_q, seq_len_k]
        # Higher entropy = more uniform attention, lower = more focused
        return -torch.sum(attention_tensor * torch.log(attention_tensor + 1e-10), dim=-1)
    
    def compute_attention_stats(self, attention_data=None):
        """Compute statistics for attention patterns"""
        if attention_data is None:
            attention_data = self.current_attention
            
        if attention_data is None:
            print("No attention data available")
            return None
        
        stats = {}
        
        for attn_type, layers in attention_data.items():
            layer_stats = []
            
            for layer_idx, layer_attn in enumerate(layers):
                # Get attention tensor shape [num_heads, seq_len_q, seq_len_k]
                num_heads, seq_len_q, seq_len_k = layer_attn.shape
                
                # Compute statistics per head
                head_stats = []
                for head_idx in range(num_heads):
                    head_attn = layer_attn[head_idx]
                    
                    # Compute entropy (measure of attention focus)
                    entropy = -torch.sum(head_attn * torch.log(head_attn + 1e-10), dim=-1).mean().item()
                    
                    # Compute sparsity (percentage of attention weights below threshold)
                    threshold = 0.01
                    sparsity = (head_attn < threshold).float().mean().item()
                    
                    # Compute attention to diagonals (for self-attention, measure of local focus)
                    if seq_len_q == seq_len_k:
                        diag_indices = torch.arange(min(seq_len_q, seq_len_k))
                        diag_attn = head_attn[diag_indices, diag_indices].mean().item()
                    else:
                        diag_attn = None
                    
                    head_stats.append({
                        'entropy': entropy,
                        'sparsity': sparsity,
                        'diagonal_attention': diag_attn
                    })
                
                layer_stats.append(head_stats)
            
            stats[attn_type] = layer_stats
        
        return stats
    
    def plot_attention_evolution(self, attn_type='encoder', layer_idx=0, head_idx=0, metric='entropy', last_n=None):
        """
        Plot how attention patterns evolve during training
        
        Args:
            attn_type: 'encoder', 'decoder_self', or 'decoder_cross'
            layer_idx: Layer index
            head_idx: Head index
            metric: 'entropy', 'sparsity', or 'diagonal_attention'
            last_n: Number of most recent iterations to plot (None for all)
        """
        if not self.attention_history:
            print("No attention history available")
            return None
        
        # Extract data for plotting
        iterations = []
        metric_values = []
        
        history = self.attention_history
        if last_n is not None:
            history = history[-last_n:]
        
        for entry in history:
            if attn_type not in entry['attention']:
                continue
                
            attn_data = entry['attention'][attn_type]
            if layer_idx >= len(attn_data):
                continue
                
            layer_attn = attn_data[layer_idx]
            if head_idx >= layer_attn.shape[0]:
                continue
            
            # Compute required metric
            head_attn = layer_attn[head_idx]
            
            if metric == 'entropy':
                # Compute entropy
                value = -torch.sum(head_attn * torch.log(head_attn + 1e-10), dim=-1).mean().item()
            elif metric == 'sparsity':
                # Compute sparsity
                threshold = 0.01
                value = (head_attn < threshold).float().mean().item()
            elif metric == 'diagonal_attention':
                # Compute diagonal attention
                if head_attn.shape[0] == head_attn.shape[1]:
                    diag_indices = torch.arange(head_attn.shape[0])
                    value = head_attn[diag_indices, diag_indices].mean().item()
                else:
                    value = None
            else:
                continue
            
            if value is not None:
                iterations.append(entry['iteration'])
                metric_values.append(value)
        
        if not iterations:
            print(f"No data available for {attn_type}, layer {layer_idx}, head {head_idx}, metric {metric}")
            return None
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, metric_values, marker='o')
        plt.title(f"Evolution of {metric} for {attn_type}, Layer {layer_idx+1}, Head {head_idx+1}")
        plt.xlabel('Iteration')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.grid(True)
        plt.tight_layout()
        
        return plt.gcf()
    
    def compare_attention_patterns(self, iteration1, iteration2, attn_type='encoder', layer_idx=0, head_idx=0):
        """
        Compare attention patterns between two iterations
        
        Args:
            iteration1: First iteration index
            iteration2: Second iteration index
            attn_type: 'encoder', 'decoder_self', or 'decoder_cross'
            layer_idx: Layer index
            head_idx: Head index
        """
        if not self.attention_history:
            print("No attention history available")
            return None
            
        # Find entries for the specified iterations
        entry1 = None
        entry2 = None
        
        for entry in self.attention_history:
            if entry['iteration'] == iteration1:
                entry1 = entry
            elif entry['iteration'] == iteration2:
                entry2 = entry
                
            if entry1 and entry2:
                break
        
        if not entry1 or not entry2:
            print(f"Could not find data for iterations {iteration1} and/or {iteration2}")
            return None
        
        # Extract attention maps
        if attn_type not in entry1['attention'] or attn_type not in entry2['attention']:
            print(f"Attention type {attn_type} not available in one or both entries")
            return None
            
        attn_data1 = entry1['attention'][attn_type]
        attn_data2 = entry2['attention'][attn_type]
        
        if layer_idx >= len(attn_data1) or layer_idx >= len(attn_data2):
            print(f"Layer {layer_idx} not available in one or both entries")
            return None
            
        layer_attn1 = attn_data1[layer_idx]
        layer_attn2 = attn_data2[layer_idx]
        
        if head_idx >= layer_attn1.shape[0] or head_idx >= layer_attn2.shape[0]:
            print(f"Head {head_idx} not available in one or both entries")
            return None
        
        # Get attention weights for the specified head
        head_attn1 = layer_attn1[head_idx].numpy()
        head_attn2 = layer_attn2[head_idx].numpy()
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot first attention pattern
        sns.heatmap(head_attn1, ax=axes[0], cmap='viridis', vmin=0, vmax=1)
        axes[0].set_title(f"Iteration {iteration1}")
        
        # Plot second attention pattern
        sns.heatmap(head_attn2, ax=axes[1], cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title(f"Iteration {iteration2}")
        
        # Plot difference
        diff = head_attn2 - head_attn1
        max_diff = max(abs(diff.min()), abs(diff.max()))
        sns.heatmap(diff, ax=axes[2], cmap='coolwarm', vmin=-max_diff, vmax=max_diff)
        axes[2].set_title(f"Difference (Iteration {iteration2} - Iteration {iteration1})")
        
        plt.suptitle(f"Comparison of {attn_type.replace('_', ' ').title()} Attention, Layer {layer_idx+1}, Head {head_idx+1}")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        return fig

    def identify_important_heads(self, attn_type='encoder', metric='entropy', top_k=3):
        """
        Identify the most important attention heads based on a metric
        
        Args:
            attn_type: 'encoder', 'decoder_self', or 'decoder_cross'
            metric: 'entropy', 'sparsity', or 'diagonal_attention'
            top_k: Number of top heads to identify
        """
        if self.current_attention is None:
            print("No current attention data available")
            return None
        
        if attn_type not in self.current_attention:
            print(f"Attention type {attn_type} not available")
            return None
        
        # Compute attention statistics
        stats = self.compute_attention_stats()
        
        if stats is None or attn_type not in stats:
            return None
            
        # Collect metrics from all heads
        head_metrics = []
        
        for layer_idx, layer_stats in enumerate(stats[attn_type]):
            for head_idx, head_stats in enumerate(layer_stats):
                if metric in head_stats and head_stats[metric] is not None:
                    head_metrics.append({
                        'layer': layer_idx,
                        'head': head_idx,
                        metric: head_stats[metric]
                    })
        
        if not head_metrics:
            print(f"No {metric} data available for {attn_type}")
            return None
        
        # Sort based on metric (lower entropy/higher sparsity/higher diagonal attention)
        sort_reverse = (metric != 'entropy')
        sorted_heads = sorted(head_metrics, key=lambda x: x[metric], reverse=sort_reverse)
        
        # Take top-k
        top_heads = sorted_heads[:top_k]
        
        print(f"Top {len(top_heads)} {attn_type.replace('_', ' ')} heads by {metric}:")
        for i, head_info in enumerate(top_heads):
            print(f"  {i+1}. Layer {head_info['layer']+1}, Head {head_info['head']+1}: {metric}={head_info[metric]:.4f}")
        
        return top_heads

    def visualize_top_heads(self, attn_type='encoder', metric='entropy', top_k=3):
        """
        Visualize the top attention heads based on a metric
        
        Args:
            attn_type: 'encoder', 'decoder_self', or 'decoder_cross'
            metric: 'entropy', 'sparsity', or 'diagonal_attention'
            top_k: Number of top heads to visualize
        """
        top_heads = self.identify_important_heads(attn_type, metric, top_k)
        
        if top_heads is None or not top_heads:
            return None
            
        # Create figure with subplots for each head
        fig, axes = plt.subplots(1, len(top_heads), figsize=(5*len(top_heads), 5))
        if len(top_heads) == 1:
            axes = [axes]
        
        for i, head_info in enumerate(top_heads):
            layer_idx = head_info['layer']
            head_idx = head_info['head']
            
            # Get attention weights
            attn_weights = self.current_attention[attn_type][layer_idx][head_idx].numpy()
            
            # Plot
            sns.heatmap(attn_weights, ax=axes[i], cmap='viridis', vmin=0, vmax=1)
            axes[i].set_title(f"Layer {layer_idx+1}, Head {head_idx+1}\n{metric}={head_info[metric]:.4f}")
        
        plt.suptitle(f"Top {len(top_heads)} {attn_type.replace('_', ' ')} heads by {metric}")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        return fig

class IntermediateActivationTracker:
    """
    A class to track, analyze and visualize intermediate activations
    """
    def __init__(self, model):
        self.model = model
        self.activation_history = []
        self.iteration = 0
        self.current_activations = None
    
    def capture(self):
        """Capture current intermediate activations from the model"""
        # Get intermediates from the model
        intermediates = self.model.get_intermediates()
        
        if intermediates is None:
            print("No intermediates available to capture")
            return None
        
        # Process and store intermediate activations
        processed_activations = self._process_intermediates(intermediates)
        
        # Store current activations
        self.current_activations = processed_activations
        
        # Add to history
        self.activation_history.append({
            'iteration': self.iteration,
            'timestamp': time.time(),
            'activations': processed_activations
        })
        
        self.iteration += 1
        
        return processed_activations
    
    def _process_intermediates(self, intermediates):
        """Process raw intermediates into a more structured format"""
        processed = {}
        
        # Extract encoder activations
        if 'encoder' in intermediates:
            encoder_data = intermediates['encoder']
            if 'layers' in encoder_data:
                encoder_layers = []
                
                for layer_idx, layer_data in enumerate(encoder_data['layers']):
                    # Extract relevant tensors
                    layer_activations = {}
                    
                    for key, tensor in layer_data.items():
                        if isinstance(tensor, torch.Tensor):
                            # Store statistics rather than full tensors to save memory
                            layer_activations[key] = {
                                'shape': tensor.shape,
                                'min': tensor.min().item(),
                                'max': tensor.max().item(),
                                'mean': tensor.mean().item(),
                                'std': tensor.std().item(),
                                'norm': tensor.norm().item()
                            }
                    
                    encoder_layers.append(layer_activations)
                
                processed['encoder_layers'] = encoder_layers
        
        # Extract decoder activations
        if 'decoder' in intermediates:
            decoder_data = intermediates['decoder']
            if 'layers' in decoder_data:
                decoder_layers = []
                
                for layer_idx, layer_data in enumerate(decoder_data['layers']):
                    # Extract relevant tensors
                    layer_activations = {}
                    
                    for key, tensor in layer_data.items():
                        if isinstance(tensor, torch.Tensor):
                            # Store statistics rather than full tensors to save memory
                            layer_activations[key] = {
                                'shape': tensor.shape,
                                'min': tensor.min().item(),
                                'max': tensor.max().item(),
                                'mean': tensor.mean().item(),
                                'std': tensor.std().item(),
                                'norm': tensor.norm().item()
                            }
                    
                    decoder_layers.append(layer_activations)
                
                processed['decoder_layers'] = decoder_layers
        
        return processed
    
    def plot_activation_stats(self, activation_type, layer_idx, stats_type='norm', component=None, last_n=None):
        """
        Plot activation statistics over iterations
        
        Args:
            activation_type: 'encoder' or 'decoder'
            layer_idx: Layer index
            stats_type: 'min', 'max', 'mean', 'std', or 'norm'
            component: Specific component to plot (None for all)
            last_n: Number of most recent iterations to plot (None for all)
        """
        if not self.activation_history:
            print("No activation history available")
            return None
        
        # Determine layer key
        layer_key = f"{activation_type}_layers"
        
        # Process history
        history = self.activation_history
        if last_n is not None:
            history = history[-last_n:]
        
        # Collect data for plotting
        iterations = []
        stats_data = defaultdict(list)
        
        for entry in history:
            if 'activations' not in entry or layer_key not in entry['activations']:
                continue
                
            layer_data = entry['activations'][layer_key]
            
            if layer_idx >= len(layer_data):
                continue
                
            activations = layer_data[layer_idx]
            
            # Add iteration
            iterations.append(entry['iteration'])
            
            # Collect stats for components
            for comp_name, comp_stats in activations.items():
                if component is not None and comp_name != component:
                    continue
                    
                if stats_type in comp_stats:
                    stats_data[comp_name].append(comp_stats[stats_type])
        
        if not iterations or not stats_data:
            print(f"No data available for {activation_type}, layer {layer_idx}, stats {stats_type}")
            return None
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        for comp_name, values in stats_data.items():
            if len(values) != len(iterations):
                # Skip components with incomplete data
                continue
                
            plt.plot(iterations, values, marker='o', label=comp_name)
        
        plt.title(f"{stats_type.title()} of {activation_type.title()} Layer {layer_idx+1} Activations")
        plt.xlabel('Iteration')
        plt.ylabel(stats_type.title())
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        return plt.gcf()
    
    def compare_layer_activations(self, activation_type, component, stats_type='norm'):
        """
        Compare activation statistics across different layers
        
        Args:
            activation_type: 'encoder' or 'decoder'
            component: Component name to compare
            stats_type: 'min', 'max', 'mean', 'std', or 'norm'
        """
        if self.current_activations is None:
            print("No current activations available")
            return None
        
        # Determine layer key
        layer_key = f"{activation_type}_layers"
        
        if layer_key not in self.current_activations:
            print(f"No {activation_type} layer data available")
            return None
        
        layer_data = self.current_activations[layer_key]
        
        # Collect stats from each layer
        layer_stats = []
        for layer_idx, activations in enumerate(layer_data):
            if component not in activations:
                continue
                
            comp_stats = activations[component]
            if stats_type in comp_stats:
                layer_stats.append((layer_idx, comp_stats[stats_type]))
        
        if not layer_stats:
            print(f"No data available for component {component}, stats {stats_type}")
            return None
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Extract data for plotting
        layers, stats = zip(*layer_stats)
        layers = [l+1 for l in layers]  # 1-indexed for display
        
        plt.bar(layers, stats)
        plt.title(f"{stats_type.title()} of {component} Across {activation_type.title()} Layers")
        plt.xlabel('Layer')
        plt.ylabel(stats_type.title())
        plt.xticks(layers)
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        return plt.gcf()
    
    def identify_activation_issues(self, threshold=5.0):
        """
        Identify potential issues in activations (e.g., exploding/vanishing)
        
        Args:
            threshold: Threshold for flagging abnormal values
        """
        if self.current_activations is None:
            print("No current activations available")
            return None
        
        issues = []
        
        # Check encoder layers
        if 'encoder_layers' in self.current_activations:
            for layer_idx, layer_data in enumerate(self.current_activations['encoder_layers']):
                for comp_name, comp_stats in layer_data.items():
                    # Check for extremely large values (potential exploding)
                    if abs(comp_stats['max']) > threshold or abs(comp_stats['min']) > threshold:
                        issues.append({
                            'type': 'encoder',
                            'layer': layer_idx,
                            'component': comp_name,
                            'issue': 'potential_exploding',
                            'max': comp_stats['max'],
                            'min': comp_stats['min']
                        })
                    
                    # Check for extremely small values (potential vanishing)
                    if abs(comp_stats['norm']) < 1.0 / threshold:
                        issues.append({
                            'type': 'encoder',
                            'layer': layer_idx,
                            'component': comp_name,
                            'issue': 'potential_vanishing',
                            'norm': comp_stats['norm']
                        })
        
        # Check decoder layers
        if 'decoder_layers' in self.current_activations:
            for layer_idx, layer_data in enumerate(self.current_activations['decoder_layers']):
                for comp_name, comp_stats in layer_data.items():
                    # Check for extremely large values (potential exploding)
                    if abs(comp_stats['max']) > threshold or abs(comp_stats['min']) > threshold:
                        issues.append({
                            'type': 'decoder',
                            'layer': layer_idx,
                            'component': comp_name,
                            'issue': 'potential_exploding',
                            'max': comp_stats['max'],
                            'min': comp_stats['min']
                        })
                    
                    # Check for extremely small values (potential vanishing)
                    if abs(comp_stats['norm']) < 1.0 / threshold:
                        issues.append({
                            'type': 'decoder',
                            'layer': layer_idx,
                            'component': comp_name,
                            'issue': 'potential_vanishing',
                            'norm': comp_stats['norm']
                        })
        
        # Report issues
        if issues:
            print(f"Found {len(issues)} potential issues in activations:")
            for i, issue in enumerate(issues):
                if issue['issue'] == 'potential_exploding':
                    print(f"  {i+1}. EXPLODING: {issue['type']} layer {issue['layer']+1}, {issue['component']}")
                    print(f"     Max: {issue['max']:.4f}, Min: {issue['min']:.4f}")
                else:
                    print(f"  {i+1}. VANISHING: {issue['type']} layer {issue['layer']+1}, {issue['component']}")
                    print(f"     Norm: {issue['norm']:.8f}")
        else:
            print("No activation issues detected")
        
        return issues

class GradientAnalyzer:
    """
    A class to analyze gradient flow through the model during training
    
    This tool helps identify gradient vanishing/exploding issues and
    understand parameter sensitivity.
    """
    def __init__(self, model):
        self.model = model
        self.gradient_history = []
        self.iteration = 0
    
    def capture(self):
        """Capture current gradients from the model"""
        # Get gradients from the model
        gradients = self.model.get_gradients()
        
        if gradients is None:
            print("No gradients available to capture")
            return None
        
        # Process and store gradients
        processed_gradients = self._process_gradients(gradients)
        
        # Add to history
        self.gradient_history.append({
            'iteration': self.iteration,
            'timestamp': time.time(),
            'gradients': processed_gradients
        })
        
        self.iteration += 1
        
        return processed_gradients
    
    def _process_gradients(self, gradients):
        """Process raw gradients into a more structured format"""
        processed = {}
        
        # Process model-level gradients
        if 'model' in gradients:
            processed['model'] = {}
            for param_name, grad in gradients['model'].items():
                if isinstance(grad, torch.Tensor):
                    processed['model'][param_name] = {
                        'shape': grad.shape,
                        'min': grad.min().item(),
                        'max': grad.max().item(),
                        'mean': grad.mean().item(),
                        'std': grad.std().item(),
                        'norm': grad.norm().item()
                    }
        
        # Process encoder gradients
        if 'encoder' in gradients:
            encoder_grads = gradients['encoder']
            if 'layers' in encoder_grads:
                encoder_layers = []
                
                for layer_idx, layer_grads in enumerate(encoder_grads['layers']):
                    layer_processed = {}
                    
                    for param_name, grad in layer_grads.items():
                        if isinstance(grad, torch.Tensor):
                            layer_processed[param_name] = {
                                'shape': grad.shape,
                                'min': grad.min().item(),
                                'max': grad.max().item(),
                                'mean': grad.mean().item(),
                                'std': grad.std().item(),
                                'norm': grad.norm().item()
                            }
                    
                    encoder_layers.append(layer_processed)
                
                processed['encoder_layers'] = encoder_layers
        
        # Process decoder gradients
        if 'decoder' in gradients:
            decoder_grads = gradients['decoder']
            if 'layers' in decoder_grads:
                decoder_layers = []
                
                for layer_idx, layer_grads in enumerate(decoder_grads['layers']):
                    layer_processed = {}
                    
                    for param_name, grad in layer_grads.items():
                        if isinstance(grad, torch.Tensor):
                            layer_processed[param_name] = {
                                'shape': grad.shape,
                                'min': grad.min().item(),
                                'max': grad.max().item(),
                                'mean': grad.mean().item(),
                                'std': grad.std().item(),
                                'norm': grad.norm().item()
                            }
                    
                    decoder_layers.append(layer_processed)
                
                processed['decoder_layers'] = decoder_layers
        
        return processed
    
    def plot_gradient_norms(self, module_type, layer_idx=None, last_n=None, top_k=5):
        """
        Plot gradient norms over iterations
        
        Args:
            module_type: 'encoder', 'decoder' or 'model'
            layer_idx: Layer index (for encoder/decoder)
            last_n: Number of most recent iterations to plot (None for all)
            top_k: Number of parameters with largest gradients to show
        """
        if not self.gradient_history:
            print("No gradient history available")
            return None
        
        # Process history
        history = self.gradient_history
        if last_n is not None:
            history = history[-last_n:]
        
        # Determine what to plot
        param_data = defaultdict(list)
        iterations = []
        
        for entry in history:
            iterations.append(entry['iteration'])
            
            if module_type == 'model':
                if 'model' in entry['gradients']:
                    for param_name, stats in entry['gradients']['model'].items():
                        param_data[param_name].append(stats['norm'])
            elif layer_idx is not None:
                layer_key = f"{module_type}_layers"
                if layer_key in entry['gradients'] and layer_idx < len(entry['gradients'][layer_key]):
                    layer_grads = entry['gradients'][layer_key][layer_idx]
                    for param_name, stats in layer_grads.items():
                        param_data[param_name].append(stats['norm'])
        
        if not param_data:
            print(f"No gradient data available for {module_type}" + 
                  (f" layer {layer_idx}" if layer_idx is not None else ""))
            return None
        
        # Calculate average norm for each parameter
        avg_norms = {}
        for name, norms in param_data.items():
            if len(norms) == len(iterations):
                avg_norms[name] = sum(norms) / len(norms)
        
        # Sort by average norm and keep top-k
        sorted_names = sorted(avg_norms.keys(), key=lambda x: avg_norms[x], reverse=True)
        if top_k is not None:
            sorted_names = sorted_names[:top_k]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        for name in sorted_names:
            norms = param_data[name]
            if len(norms) == len(iterations):
                plt.plot(iterations, norms, marker='o', label=name)
        
        title = f"Gradient Norms for {module_type.title()}"
        if layer_idx is not None:
            title += f" Layer {layer_idx+1}"
        
        plt.title(title)
        plt.xlabel('Iteration')
        plt.ylabel('Gradient Norm')
        plt.yscale('log')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        return plt.gcf()
    
    def compare_layer_gradients(self, module_type, param_name, stats_type='norm'):
        """
        Compare gradients across different layers
        
        Args:
            module_type: 'encoder' or 'decoder'
            param_name: Parameter name to compare
            stats_type: 'min', 'max', 'mean', 'std', or 'norm'
        """
        if not self.gradient_history:
            print("No gradient history available")
            return None
        
        # Get most recent gradients
        latest_grads = self.gradient_history[-1]['gradients']
        
        # Determine layer key
        layer_key = f"{module_type}_layers"
        
        if layer_key not in latest_grads:
            print(f"No {module_type} layer gradient data available")
            return None
        
        # Collect stats from each layer
        layer_stats = []
        for layer_idx, layer_grads in enumerate(latest_grads[layer_key]):
            if param_name in layer_grads:
                param_stats = layer_grads[param_name]
                if stats_type in param_stats:
                    layer_stats.append((layer_idx, param_stats[stats_type]))
        
        if not layer_stats:
            print(f"No gradient data available for parameter {param_name}, stats {stats_type}")
            return None
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Extract data for plotting
        layers, stats = zip(*layer_stats)
        layers = [l+1 for l in layers]  # 1-indexed for display
        
        plt.bar(layers, stats)
        plt.title(f"{stats_type.title()} of {param_name} Gradients Across {module_type.title()} Layers")
        plt.xlabel('Layer')
        plt.ylabel(stats_type.title())
        plt.xticks(layers)
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        return plt.gcf()
    
    def identify_gradient_issues(self, threshold=5.0):
        """
        Identify potential issues in gradients (e.g., exploding/vanishing)
        
        Args:
            threshold: Threshold for flagging abnormal values
        """
        if not self.gradient_history:
            print("No gradient history available")
            return None
        
        # Get most recent gradients
        latest_grads = self.gradient_history[-1]['gradients']
        
        issues = []
        
        # Process model-level gradients
        if 'model' in latest_grads:
            for param_name, stats in latest_grads['model'].items():
                # Check for extremely large gradients (potential exploding)
                if abs(stats['max']) > threshold or abs(stats['min']) > threshold:
                    issues.append({
                        'type': 'model',
                        'parameter': param_name,
                        'issue': 'potential_exploding',
                        'max': stats['max'],
                        'min': stats['min']
                    })
                
                # Check for extremely small gradients (potential vanishing)
                if abs(stats['norm']) < 1.0 / threshold:
                    issues.append({
                        'type': 'model',
                        'parameter': param_name,
                        'issue': 'potential_vanishing',
                        'norm': stats['norm']
                    })
        
        # Check encoder layers
        if 'encoder_layers' in latest_grads:
            for layer_idx, layer_grads in enumerate(latest_grads['encoder_layers']):
                for param_name, stats in layer_grads.items():
                    # Check for extremely large gradients (potential exploding)
                    if abs(stats['max']) > threshold or abs(stats['min']) > threshold:
                        issues.append({
                            'type': 'encoder',
                            'layer': layer_idx,
                            'parameter': param_name,
                            'issue': 'potential_exploding',
                            'max': stats['max'],
                            'min': stats['min']
                        })
                    
                    # Check for extremely small gradients (potential vanishing)
                    if abs(stats['norm']) < 1.0 / threshold:
                        issues.append({
                            'type': 'encoder',
                            'layer': layer_idx,
                            'parameter': param_name,
                            'issue': 'potential_vanishing',
                            'norm': stats['norm']
                        })
        
        # Check decoder layers
        if 'decoder_layers' in latest_grads:
            for layer_idx, layer_grads in enumerate(latest_grads['decoder_layers']):
                for param_name, stats in layer_grads.items():
                    # Check for extremely large gradients (potential exploding)
                    if abs(stats['max']) > threshold or abs(stats['min']) > threshold:
                        issues.append({
                            'type': 'decoder',
                            'layer': layer_idx,
                            'parameter': param_name,
                            'issue': 'potential_exploding',
                            'max': stats['max'],
                            'min': stats['min']
                        })
                    
                    # Check for extremely small gradients (potential vanishing)
                    if abs(stats['norm']) < 1.0 / threshold:
                        issues.append({
                            'type': 'decoder',
                            'layer': layer_idx,
                            'parameter': param_name,
                            'issue': 'potential_vanishing',
                            'norm': stats['norm']
                        })
        
        # Report issues
        if issues:
            print(f"Found {len(issues)} potential gradient issues:")
            for i, issue in enumerate(issues):
                if issue['issue'] == 'potential_exploding':
                    if 'layer' in issue:
                        print(f"  {i+1}. EXPLODING: {issue['type']} layer {issue['layer']+1}, {issue['parameter']}")
                    else:
                        print(f"  {i+1}. EXPLODING: {issue['type']}, {issue['parameter']}")
                    print(f"     Max: {issue['max']:.4f}, Min: {issue['min']:.4f}")
                else:
                    if 'layer' in issue:
                        print(f"  {i+1}. VANISHING: {issue['type']} layer {issue['layer']+1}, {issue['parameter']}")
                    else:
                        print(f"  {i+1}. VANISHING: {issue['type']}, {issue['parameter']}")
                    print(f"     Norm: {issue['norm']:.8f}")
        else:
            print("No gradient issues detected")
        
        return issues

def analyze_transformer_debugging(model, src, tgt):
    """
    Comprehensive analysis of a transformer model's internal state
    
    This function performs a forward pass and analyzes attention patterns,
    intermediate activations, and gradients.
    
    Args:
        model: The transformer model
        src: Source input tensor
        tgt: Target input tensor
        
    Returns:
        Dictionary with analysis results and visualizations
    """
    # Create analyzers
    attention_analyzer = AttentionAnalyzer(model)
    activation_tracker = IntermediateActivationTracker(model)
    gradient_analyzer = GradientAnalyzer(model)
    
    # Enable debugging
    model.debug = True
    
    # Register hooks for gradient tracking
    model.register_hooks()
    
    # Forward pass
    outputs, attention_maps = model(src, tgt)
    
    # Capture data from all analyzers
    attention_data = attention_analyzer.capture()
    activation_data = activation_tracker.capture()
    
    # Create a loss and do backward pass to get gradients
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is padding
    loss = criterion(
        outputs.contiguous().view(-1, outputs.size(-1)), 
        tgt[:, 1:].contiguous().view(-1)
    )
    loss.backward()
    
    # Capture gradients
    gradient_data = gradient_analyzer.capture()
    
    # Analyze attention patterns
    attention_stats = attention_analyzer.compute_attention_stats()
    top_encoder_heads = attention_analyzer.identify_important_heads('encoder', 'entropy')
    
    # Check for activation issues
    activation_issues = activation_tracker.identify_activation_issues()
    
    # Check for gradient issues
    gradient_issues = gradient_analyzer.identify_gradient_issues()
    
    # Create visualizations
    figures = {}
    
    # Visualize top encoder attention heads
    if top_encoder_heads:
        figures['top_encoder_heads'] = attention_analyzer.visualize_top_heads('encoder', 'entropy')
    
    # Visualize attention patterns for all encoder heads in first layer
    if 'encoder' in attention_data:
        figures['encoder_attention_heads'] = visualize_all_attention_heads(
            attention_data['encoder'][0],
            title="Encoder Layer 1 Attention Heads"
        )
    
    # Visualize gradient norms across encoder layers
    figures['encoder_gradient_norms'] = gradient_analyzer.compare_layer_gradients(
        'encoder', 'linear1_weight'
    )
    
    # Visualize activation norms across encoder layers
    figures['encoder_activation_norms'] = activation_tracker.compare_layer_activations(
        'encoder', 'output', 'norm'
    )
    
    return {
        'loss': loss.item(),
        'attention_data': attention_data,
        'activation_data': activation_data,
        'gradient_data': gradient_data,
        'attention_stats': attention_stats,
        'top_encoder_heads': top_encoder_heads,
        'activation_issues': activation_issues,
        'gradient_issues': gradient_issues,
        'figures': figures
    }

def visualize_all_attention_heads(attention_weights, layer_idx=0, sample_idx=0, title='Attention Heads'):
    """
    Create a grid of heatmaps for all attention heads in a layer
    
    Args:
        attention_weights: Tensor of shape [batch_size, num_heads, seq_len_q, seq_len_k] or list of such tensors
        layer_idx: Which layer to visualize (if attention_weights is a list)
        sample_idx: Which batch sample to visualize
        title: Title for the plot
    """
    # Handle if input is a list of attention matrices (from different layers)
    if isinstance(attention_weights, list):
        attn = attention_weights[layer_idx]
    else:
        attn = attention_weights
    
    # Get number of heads
    num_heads = attn.shape[1]
    
    # Determine grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_heads)))
    
    plt.figure(figsize=(grid_size * 4, grid_size * 3))
    plt.suptitle(f"{title} (Layer {layer_idx+1})", fontsize=16)
    
    for head_idx in range(num_heads):
        weights = attn[sample_idx, head_idx].detach().cpu().numpy()
        
        plt.subplot(grid_size, grid_size, head_idx + 1)
        sns.heatmap(weights, cmap='viridis', vmin=0, vmax=1, xticklabels=5, yticklabels=5)
        plt.title(f"Head {head_idx+1}")
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
    return plt.gcf()