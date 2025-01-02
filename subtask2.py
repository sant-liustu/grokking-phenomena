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
    
# 2. MLP Model & LSTM Model
class MLPBlock(nn.Module):
    def __init__(self, dim_model: int, num_heads: int):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_model * num_heads),
            nn.ReLU(),
            nn.Linear(dim_model * num_heads, dim_model),
        )
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        return self.layer_norm(x + self.ffn(x))
  
class MLP(nn.Module):
    def __init__(self, num_layers: int, dim_model: int, num_heads: int, num_tokens: int, seq_len: int):
        super().__init__()

        self.position_embeddings = nn.Embedding(seq_len, dim_model)
        self.model = nn.Sequential(
            *[MLPBlock(dim_model * seq_len, num_heads) for _ in range(num_layers)],
            nn.LayerNorm(dim_model * seq_len),
            nn.Linear(dim_model * seq_len, num_tokens),
        )

    def forward(self, x):
        #这里假设x是已经经过token embedding的
        batch_size, seq_len, embed_dim = x.shape

        position_embeddings = self.position_embeddings(torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1))

        embedding = x + position_embeddings

        embedding = embedding.view(batch_size, -1)

        return self.model(embedding)

class LSTMModel(nn.Module):
    def __init__(self, num_layers: int, dim_model: int, hidden_dim: int, num_tokens: int, seq_len: int):
        super().__init__()

        self.position_embeddings = nn.Embedding(seq_len, dim_model)
        self.lstm = nn.LSTM(dim_model, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_tokens)

    def forward(self, x):
        #这里假设x是已经经过token embedding的
        batch_size, seq_len, embed_dim = x.shape

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        position_embeddings = self.position_embeddings(positions)

        embeddings = x + position_embeddings

        lstm_out, _ = self.lstm(embeddings)
        out = self.fc(lstm_out[:, -1, :])  # We only want the last output of the sequence

        return out
    

# 3. Training Loop
def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, save_path):
    train_accs = []
    val_accs = []
    steps = []
    current_step = 0
    step_start_time = time.time()
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
                    
                    step_end_time = time.time()
                    step_duration = step_end_time - step_start_time
                    step_start_time = step_end_time
                    print(f'Step {current_step}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, 100 Step Time: {step_duration:.2f} seconds')
            
            current_step += 1
        
            
        # Save checkpoint
        if (epoch + 1) % 500 == 0:
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
def run_experiment(model_type='MLP', p=97, hidden_dim=128, num_heads=4, num_layers=2, 
                  batch_size=512, lr=1e-3, weight_decay=1, training_fraction=0.5,
                  num_epochs=1000, device='cuda', save_path='checkpoints'):
    # Create save directory
    save_path = f"{model_type}-{save_path}-training_fraction={training_fraction}"
    os.makedirs(save_path, exist_ok=True)
    
    # Create dataset
    dataset = ModularAdditionDataset(p, hidden_dim)
    train_size = int(training_fraction * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    if model_type == 'MLP':
        model = MLP(num_layers, hidden_dim, num_heads, p, 4).to(device)
    elif model_type == 'LSTM':
        model = LSTMModel(num_layers, hidden_dim, hidden_dim, p, 4).to(device)
    else:
        raise ValueError("Invalid model type. Choose either 'MLP' or 'LSTM'.")
    
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
    plt.title(f'{model_type} Model - Training Fraction: {training_fraction}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'{model_type}_grokking_plot.png'))
    plt.close()

if __name__ == "__main__":
    run_experiment(model_type='MLP')
    run_experiment(model_type='LSTM')