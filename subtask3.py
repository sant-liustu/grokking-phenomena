import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import os
import time
from torch.optim.lr_scheduler import StepLR

# Reuse ModularAdditionDataset from previous subtasks
class ModularAdditionDataset(Dataset):
    def __init__(self, p, embed_dim):
        assert embed_dim > p + 2, "Embedding dimension must be greater than p + 2"
        self.p = p
        self.embed_dim = embed_dim
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print("CUDA is available")
            print("Dataset created on CUDA, so you don't need to x.to(device) when loading data during training")
        else:
            print("Dataset created on CPU")
        
        x = torch.arange(p, device=device)
        y = torch.arange(p, device=device)
        self.x, self.y = torch.meshgrid(x, y, indexing='ij')
        self.result = (self.x + self.y) % p
        
        # Flatten tensors
        self.x = self.x.reshape(-1)
        self.y = self.y.reshape(-1)
        self.result = self.result.reshape(-1)
        
        # Convert to one-hot vectors
        self.x = nn.functional.one_hot(self.x, self.embed_dim)
        self.y = nn.functional.one_hot(self.y, self.embed_dim)
        
        # Create one-hot vectors for '+' and '='
        self.plus = nn.functional.one_hot(torch.tensor([p], device=device), embed_dim)
        self.equals = nn.functional.one_hot(torch.tensor([p + 1], device=device), embed_dim)
        
    def __len__(self):
        return len(self.result)
        
    def __getitem__(self, idx):
        x_onehot = self.x[idx].float()
        y_onehot = self.y[idx].float()
        result = self.result[idx]
        input_vector = torch.stack([x_onehot, self.plus.squeeze(0).float(), y_onehot, self.equals.squeeze(0).float()])
        return input_vector, result

# Modified Transformer with dropout
class DecoderBlock(nn.Module):
    def __init__(self, dim_model: int, n_heads: int, dropout_rate: float):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(dim_model, n_heads)
        self.self_attn_norm = nn.LayerNorm(dim_model)
        self.self_attn_dropout = nn.Dropout(p=dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(dim_model * 4, dim_model)
        )
        self.ffn_norm = nn.LayerNorm(dim_model)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)
        
        a1, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        a1 = self.self_attn_norm(x + a1)
        a1 = self.self_attn_dropout(a1)
        a2 = self.ffn(a1)
        a2 = self.ffn_norm(a1 + a2)

        return a2
    
class TransformerWithDropout(nn.Module):
    def __init__(self, num_layers: int, dim_model: int, num_heads: int, num_tokens: int, dropout_rate: float = 0.1):
        super().__init__()

        self.model = nn.Sequential(
            *[DecoderBlock(dim_model, num_heads, dropout_rate) for _ in range(num_layers)],
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, num_tokens)
        )

    def _position_encoding(self, seq_len, dim_model):
        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(dim_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / dim_model)
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return torch.tensor(angle_rads[np.newaxis, ...], dtype=torch.float32)

    def forward(self, x):
        position_embedding = self._position_encoding(x.shape[1], x.shape[2]).to(x.device)
        embedding = x + position_embedding
        embedding = embedding.permute(1, 0, 2)
        output = self.model(embedding)
        return output[-1, :, :]

def get_optimizer(optimizer_name, model_params, lr, momentum=0.9, weight_decay=1.0):
    """Create optimizer based on name and parameters."""
    if optimizer_name == "adam":
        return torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(model_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, save_path):
    train_accs = []
    val_accs = []
    steps = []
    current_step = 0
    
    # Add learning rate scheduler
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        for x, labels in train_loader:
            # x = x.to(device)
            # labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if current_step % 100 == 0:
                with torch.no_grad():
                    train_pred = outputs.argmax(dim=1)
                    train_acc = (train_pred == labels).float().mean().item()
                    
                    model.eval()
                    val_correct = 0
                    val_total = 0
                    for val_x, val_labels in val_loader:
                        val_x = val_x.to(device)
                        val_labels = val_labels.to(device)
                        val_outputs = model(val_x)
                        val_pred = val_outputs.argmax(dim=1)
                        val_correct += (val_pred == val_labels).sum().item()
                        val_total += val_labels.size(0)
                    val_acc = val_correct / val_total
                    model.train()
                    
                    train_accs.append(train_acc)
                    val_accs.append(val_acc)
                    steps.append(current_step)
                    end_time = time.time()
                    step_duration = end_time - start_time
                    start_time = end_time
                    
                    print(f'Step {current_step}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, Time per 100 steps: {step_duration:.2f}s')
            
            current_step += 1
        scheduler.step()
            
        if (epoch + 1) % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_accs': train_accs,
                'val_accs': val_accs,
                'steps': steps
            }, os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pt'))
            
    return steps, train_accs, val_accs

def run_experiment(
    optimizer_name='adamw',
    p=97,
    hidden_dim=128,
    num_heads=4,
    num_layers=2,
    batch_size=512,
    lr=1e-3,
    momentum=0.9,
    weight_decay=1.0,
    dropout_rate=0.1,
    training_fraction=0.5,
    num_epochs=1000,
    device='cuda',
    save_path='subtask3_optimizer_experiments'
):
    experiment_name = f"{optimizer_name}_wd{weight_decay}_dr{dropout_rate}_bs{batch_size}"
    save_dir = os.path.join(save_path, experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    
    dataset = ModularAdditionDataset(p, hidden_dim)
    train_size = int(training_fraction * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = TransformerWithDropout(num_layers, hidden_dim, num_heads, p, dropout_rate).to(device)
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr, momentum, weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    steps, train_accs, val_accs = train_and_evaluate(
        model, train_loader, val_loader, optimizer, criterion,
        num_epochs, device, save_dir
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_accs, 'r-', label='Train')
    plt.plot(steps, val_accs, 'g-', label='Validation')
    plt.xscale('log')
    plt.xlabel('Optimization Steps')
    plt.ylabel('Accuracy')
    plt.title(f'Optimizer: {optimizer_name}, WD: {weight_decay}, Dropout: {dropout_rate}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'learning_curves.png'))
    plt.close()
    
    return steps, train_accs, val_accs

def run_all_experiments():
    optimizers = ['adamw', 'adam', 'sgd', 'rmsprop']
    weight_decays = [0.0, 0.1, 1.0]
    dropout_rates = [0.0, 0.1, 0.2]
    batch_sizes = [256, 512, 1024]
    
    results = {}
    
    for opt in optimizers:
        for wd in weight_decays:
            for dr in dropout_rates:
                for bs in batch_sizes:
                    print(f"\nRunning experiment with {opt}, weight_decay={wd}, dropout={dr}, batch_size={bs}")
                    start_time = time.time()
                    steps, train_accs, val_accs = run_experiment(
                        optimizer_name=opt,
                        weight_decay=wd,
                        dropout_rate=dr,
                        batch_size=bs
                    )
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Experiment with {opt}, weight_decay={wd}, dropout={dr}, batch_size={bs} took {elapsed_time:.2f} seconds")
                    results[f"{opt}_wd{wd}_dr{dr}_bs{bs}"] = {
                        'steps': steps,
                        'train_accs': train_accs,
                        'val_accs': val_accs
                    }
    
    return results

if __name__ == "__main__":
    results = run_all_experiments()