import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import os
import time
import wandb
print(torch.cuda.is_available())
# 1. Data Generation
class ModularAdditionDataset(Dataset):
    def __init__(self, p, embed_dim, K):
        assert embed_dim > p + 2, "Embedding dimension must be greater than p + 2"
        self.p = p
        self.embed_dim = embed_dim
        self.K = K

        values = [torch.arange(p, device='cuda') for _ in range(K)]
        grids = torch.meshgrid(*values, indexing='ij')
        self.inputs = [grid.reshape(-1) for grid in grids]
        
        self.result = (sum(self.inputs) % p).to(torch.device('cuda'))
        # Convert to one-hot vectors
        self.inputs_onehot = [nn.functional.one_hot(inp, self.embed_dim).float() for inp in self.inputs]

        # Create one-hot vectors for '+' and '='
        self.plus = nn.functional.one_hot(torch.tensor([p], device='cuda'), embed_dim).float()
        self.equals = nn.functional.one_hot(torch.tensor([p + 1], device='cuda'), embed_dim).float()

    def __len__(self):
        return len(self.result)

    def __getitem__(self, idx):
        # Get one-hot encoded inputs
        input_onehots = [inp_onehot[idx] for inp_onehot in self.inputs_onehot]
        # Build the sequence: x1 + x2 + ... + xK =
        sequence = []
        for i, x in enumerate(input_onehots):
            sequence.append(x)
            if i < self.K - 1:
                sequence.append(self.plus.squeeze(0))
        sequence.append(self.equals.squeeze(0))

        input_vector = torch.stack(sequence)
        result = self.result[idx]

        return input_vector, result
# 改数据生成
# 2. Transformer Model
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
    def __init__(self, num_layers: int, dim_model: int, num_heads: int, num_tokens: int, dropout_rate: float = 0.2):
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
            # x = x.to(device)
            # labels = labels.to(device)
            
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

                    wandb.log({
                        'train_accuracy': train_acc,
                        'val_accuracy': val_acc,
                        'step': current_step
                    })
                    
                    print(f'Step {current_step}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, Time per 100 steps: {step100_duration:.2f}s')
            
            current_step += 1
            # Log accuracies to wandb

               
        # # Save checkpoint
        # if (epoch + 1) % 5000 == 0:
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'train_accs': train_accs,
        #         'val_accs': val_accs,
        #         'steps': steps
        #     }, os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pt'))
            
    return steps, train_accs, val_accs

# 4. Main function to run experiment
def run_experiment(p=97, hidden_dim=128, num_heads=4, num_layers=2, 
                  batch_size=2048, lr=1e-4, weight_decay=0.1, K=3,
                  num_epochs=10000, device='cuda', save_path='transformer-checkpoints-training_fraction=0.5'):
    wandb.login(key='')
    # Initialize wandb
    wandb.init(project="grokking-modular-addition", config={
        "p": p,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "K": K,
        "num_epochs": num_epochs,
        "device": device,
        "save_path": save_path
    })
    
    # Create save directory
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_file_path, save_path)
    os.makedirs(save_path, exist_ok=True)
    print('p:', p)
    
    # Create dataset
    dataset = ModularAdditionDataset(p, hidden_dim, K)
    train_size = int(0.5 * (p ** 3))
    val_size = int(0.5 * (p ** 3))

    # Randomly select train_size + val_size indices from the dataset
    total_size = train_size + val_size
    indices = torch.randperm(len(dataset))[:total_size]
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    model = TransformerWithDropout(num_layers, hidden_dim, num_heads, p).to(device)
    
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
    
    # Finish wandb run
    wandb.finish()

#加一个对不同的training_fraction的实验


if __name__ == "__main__":
    for K in range(3, 4):
        run_experiment(K = K)