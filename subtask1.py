import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import os
import time
print(torch.cuda.is_available())
# 1. Data Generation
class ModularAdditionDataset(Dataset):
    def __init__(self, p, embed_dim):
        assert embed_dim > p + 2, "Embedding dimension must be greater than p + 2"
        self.p = p
        self.embed_dim = embed_dim
        
        x = torch.arange(p)
        y = torch.arange(p)
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
        self.plus = nn.functional.one_hot(torch.tensor([p]), embed_dim)
        self.equals = nn.functional.one_hot(torch.tensor([p + 1]), embed_dim)
        
    def __len__(self):
        return len(self.result)
        
    def __getitem__(self, idx):
        x_onehot = self.x[idx].float()
        y_onehot = self.y[idx].float()
        result = self.result[idx]
        # Concatenate x, +, y, =, result
        input_vector = torch.stack([x_onehot, self.plus.squeeze(0).float(), y_onehot, self.equals.squeeze(0).float()])
        
        return input_vector, result 

# 2. Transformer Model
class DecoderBlock(nn.Module):
    def __init__(self, dim_model: int, n_heads: int):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(dim_model, n_heads)
        self.self_attn_norm = nn.LayerNorm(dim_model)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.GELU(),
            nn.Linear(dim_model * 4, dim_model)
        )
        self.ffn_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)
        
        a1, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        a1 = self.self_attn_norm(x + a1)
        a2 = self.ffn(a1)
        a2 = self.ffn_norm(a1 + a2)

        return a2

class SimpleTransformer(nn.Module):
    def __init__(self, num_layers: int, dim_model: int, num_heads: int, num_tokens: int):
        super().__init__()

        self.model = nn.Sequential(
            *[DecoderBlock(dim_model, num_heads) for _ in range(num_layers)],
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, num_tokens)
        )

    def _position_encoding(self, seq_len, dim_model):
        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(dim_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / dim_model)
        angle_rads = pos * angle_rates

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return torch.tensor(pos_encoding, dtype=torch.float32)

    def forward(self, x):
        #x是事先已经embedding好的，只需再加上position embedding
        position_embedding = self._position_encoding(x.shape[1], x.shape[2]).to(x.device)

        embedding = x + position_embedding

        embedding = embedding.permute(1, 0, 2)  # (seq_len, batch_size, dim_model)

        output = self.model(embedding)
        output = output[-1, :, :]  # (batch_size, num_tokens)


        return output
    

# 3. Training Loop
def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, save_path):
    train_accs = []
    val_accs = []
    steps = []
    current_step = 0
        
    start_time = time.time()
    for epoch in range(num_epochs):
        
        # Training
        model.train()
        for x, labels in train_loader:
            x = x.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracies every 100 steps
            if current_step % 100 == 0:
                with torch.no_grad():
                    # Training accuracy
                    train_pred = outputs.argmax(dim=1)
                    train_acc = (train_pred == labels).float().mean().item()
                    
                    # Validation accuracy
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
                    step100_duration = end_time - start_time
                    start_time = end_time
                    
                    print(f'Step {current_step}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, Time per 100 steps: {step100_duration:.2f}s')
            
            current_step += 1
        
        
        
        
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_accs': train_accs,
                'val_accs': val_accs,
                'steps': steps
            }, os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pt'))
            
    return steps, train_accs, val_accs

# 4. Main function to run experiment
def run_experiment(p=97, hidden_dim=128, num_heads=4, num_layers=2, 
                  batch_size=1024, lr=1e-3, weight_decay=1, training_fraction=0.3,
                  num_epochs=3000, device='cuda', save_path='transformer-checkpoints-training_fraction=0.3'):
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Create dataset
    dataset = ModularAdditionDataset(p, hidden_dim)
    train_size = int(training_fraction * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    model = SimpleTransformer(num_layers, hidden_dim, num_heads, p).to(device)
    
    # Setup optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    steps, train_accs, val_accs = train_and_evaluate(
        model, train_loader, val_loader, optimizer, criterion, 
        num_epochs, device, save_path
    )
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_accs, 'r-', label='Train')
    plt.plot(steps, val_accs, 'g-', label='Validation')
    plt.xscale('log')
    plt.xlabel('Optimization Steps')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Steps——training_fraction=0.3')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'grokking_plot.png'))
    plt.close()

#加一个对不同的training_fraction的实验


if __name__ == "__main__":
    run_experiment()