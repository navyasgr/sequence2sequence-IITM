# ============================================================================
# SEQ2SEQ TRANSLITERATION - JUPYTER NOTEBOOK VERSION
# Latin to Devanagari using Aksharantar Dataset
# IIT Madras Assignment
# ============================================================================

# %% [markdown]
# # Seq2Seq Model for Latin to Devanagari Transliteration
# 
# This notebook implements a complete sequence-to-sequence model for transliterating
# Latin script to Devanagari script using the Aksharantar dataset.
# 
# **Author:** Navya  
# **Course:** IIT Madras - Deep Learning  
# **Task:** Character-level transliteration

# %% [markdown]
# ## 1. Setup and GPU Check

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ GPU not available. Training will be slower on CPU.")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %% [markdown]
# ## 2. Configuration
# 
# All hyperparameters are configurable here. This demonstrates the flexibility
# requirement of the assignment.

# %%
CONFIG = {
    # Model Architecture (All Configurable)
    'embedding_dim': 128,        # m - Character embedding dimension
    'hidden_dim': 256,           # h - Hidden state size
    'num_encoder_layers': 1,     # Number of encoder layers
    'num_decoder_layers': 1,     # Number of decoder layers
    'cell_type': 'LSTM',         # Cell type: 'RNN', 'LSTM', or 'GRU'
    'dropout': 0.3,              # Dropout probability
    
    # Training Hyperparameters
    'batch_size': 64,
    'learning_rate': 0.001,
    'num_epochs': 50,
    'teacher_forcing_ratio': 0.5,
    'max_grad_norm': 5.0,
    
    # Data Parameters
    'max_length': 50,
    'min_freq': 2,
    'train_split': 0.8,
    'val_split': 0.1,
    
    # Special Tokens
    'PAD_token': 0,
    'SOS_token': 1,
    'EOS_token': 2,
    'UNK_token': 3,
}

print("Configuration:")
print("=" * 80)
for key, value in CONFIG.items():
    print(f"  {key:25s}: {value}")
print("=" * 80)

# %% [markdown]
# ## 3. Theoretical Analysis
# 
# ### Total Computations
# 
# For an LSTM-based Seq2Seq model with:
# - Embedding size: m
# - Hidden size: h
# - Sequence length: n
# - Vocabulary size: V
# 
# **Encoder:** n × 4(hm + h²)  
# **Decoder:** n × 4(hm + h²)  
# **Output Projection:** n × hV  
# 
# **Total: O(n[8hm + 8h² + hV])**
# 
# ### Total Parameters
# 
# **Embedding:** V × m  
# **Encoder LSTM:** 4(hm + h² + h)  
# **Decoder LSTM:** 4(hm + h² + h)  
# **Output Layer:** hV + V  
# 
# **Total: V(m+1) + 8h(m+h+1) + hV**

# %%
def print_theoretical_analysis(m, h, n, V):
    """Print theoretical computations and parameters"""
    print("\n" + "="*80)
    print("THEORETICAL ANALYSIS")
    print("="*80)
    
    print(f"\nGiven:")
    print(f"  Embedding dimension (m) = {m}")
    print(f"  Hidden dimension (h) = {h}")
    print(f"  Sequence length (n) = {n}")
    print(f"  Vocabulary size (V) = {V}")
    print(f"  Cell type = {CONFIG['cell_type']}")
    
    # Computations for LSTM
    if CONFIG['cell_type'] == 'LSTM':
        encoder_comp = n * 4 * (h*m + h*h)
        decoder_comp = n * 4 * (h*m + h*h)
        output_comp = n * h * V
        total_comp = encoder_comp + decoder_comp + output_comp
        
        print(f"\n{'─'*80}")
        print("TOTAL COMPUTATIONS (LSTM, 1 layer each):")
        print(f"{'─'*80}")
        print(f"  Encoder:  n × 4(hm + h²) = {encoder_comp:,}")
        print(f"  Decoder:  n × 4(hm + h²) = {decoder_comp:,}")
        print(f"  Output:   n × hV        = {output_comp:,}")
        print(f"  {'─'*40}")
        print(f"  TOTAL:                    {total_comp:,}")
        print(f"\n  Formula: O(n[8hm + 8h² + hV])")
        
        # Parameters
        embedding_params = V * m
        encoder_params = 4 * (h*m + h*h + h)
        decoder_params = 4 * (h*m + h*h + h)
        output_params = h * V + V
        total_params = embedding_params + encoder_params + decoder_params + output_params
        
        print(f"\n{'─'*80}")
        print("TOTAL PARAMETERS (LSTM, 1 layer each):")
        print(f"{'─'*80}")
        print(f"  Embedding:     V × m              = {embedding_params:,}")
        print(f"  Encoder LSTM:  4(hm + h² + h)    = {encoder_params:,}")
        print(f"  Decoder LSTM:  4(hm + h² + h)    = {decoder_params:,}")
        print(f"  Output Layer:  hV + V            = {output_params:,}")
        print(f"  {'─'*40}")
        print(f"  TOTAL:                             {total_params:,}")
        print(f"\n  Formula: V(m+1) + 8h(m+h+1) + hV")
    
    print("="*80 + "\n")

# Example with typical values
print_theoretical_analysis(
    m=CONFIG['embedding_dim'],
    h=CONFIG['hidden_dim'],
    n=20,  # Average sequence length
    V=100  # Approximate vocabulary size
)

# %% [markdown]
# ## 4. Data Preparation

# %% [markdown]
# ### 4.1 Vocabulary Class

# %%
class Vocabulary:
    """Builds and manages character-level vocabulary"""
    
    def __init__(self, name, min_freq=1):
        self.name = name
        self.min_freq = min_freq
        self.char2index = {
            '<PAD>': CONFIG['PAD_token'],
            '<SOS>': CONFIG['SOS_token'],
            '<EOS>': CONFIG['EOS_token'],
            '<UNK>': CONFIG['UNK_token']
        }
        self.index2char = {v: k for k, v in self.char2index.items()}
        self.char2count = Counter()
        self.n_chars = 4
        
    def add_word(self, word):
        for char in word:
            self.char2count[char] += 1
    
    def build_vocab(self):
        for char, count in self.char2count.items():
            if count >= self.min_freq and char not in self.char2index:
                self.char2index[char] = self.n_chars
                self.index2char[self.n_chars] = char
                self.n_chars += 1
    
    def word_to_indices(self, word):
        return [self.char2index.get(char, CONFIG['UNK_token']) for char in word]
    
    def indices_to_word(self, indices):
        return ''.join([self.index2char.get(idx, '<UNK>') for idx in indices])

# %% [markdown]
# ### 4.2 Dataset Class

# %%
class TransliterationDataset(Dataset):
    """Custom dataset for transliteration pairs"""
    
    def __init__(self, data, src_vocab, tgt_vocab):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src_word, tgt_word = self.data[idx]
        src_indices = self.src_vocab.word_to_indices(src_word) + [CONFIG['EOS_token']]
        tgt_indices = self.tgt_vocab.word_to_indices(tgt_word) + [CONFIG['EOS_token']]
        return (torch.tensor(src_indices, dtype=torch.long),
                torch.tensor(tgt_indices, dtype=torch.long))

def collate_fn(batch):
    """Collate function for variable length sequences"""
    src_batch, tgt_batch = zip(*batch)
    src_lengths = torch.tensor([len(x) for x in src_batch])
    tgt_lengths = torch.tensor([len(x) for x in tgt_batch])
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=CONFIG['PAD_token'])
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=CONFIG['PAD_token'])
    return src_padded, tgt_padded, src_lengths, tgt_lengths

# %% [markdown]
# ### 4.3 Load and Prepare Data

# %%
# Load data (replace with actual dataset path)
# df = pd.read_csv('aksharantar_hindi.csv', header=None, names=['source', 'target'])
# data = list(zip(df['source'].values, df['target'].values))

# For demonstration, create sample data
print("Creating sample data...")
sample_data = [
    ('ghar', 'घर'), ('ajanabee', 'अजनबी'), ('kitab', 'किताब'),
    ('paani', 'पानी'), ('dost', 'दोस्त'), ('raat', 'रात'),
    ('subah', 'सुबह'), ('duniya', 'दुनिया'), ('aasman', 'आसमान'),
    ('khushi', 'खुशी'), ('pyaar', 'प्यार'), ('sapna', 'सपना'),
] * 100  # Repeat for more samples

data = sample_data

# Split data
def split_data(data, train_ratio, val_ratio):
    np.random.shuffle(data)
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return data[:train_end], data[train_end:val_end], data[val_end:]

train_data, val_data, test_data = split_data(data, CONFIG['train_split'], CONFIG['val_split'])

print(f"Train samples: {len(train_data)}")
print(f"Val samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}")

# Build vocabularies
src_vocab = Vocabulary('source', min_freq=CONFIG['min_freq'])
tgt_vocab = Vocabulary('target', min_freq=CONFIG['min_freq'])

for src, tgt in train_data:
    src_vocab.add_word(src)
    tgt_vocab.add_word(tgt)

src_vocab.build_vocab()
tgt_vocab.build_vocab()

print(f"\nSource vocabulary size: {src_vocab.n_chars}")
print(f"Target vocabulary size: {tgt_vocab.n_chars}")

# Create datasets
train_dataset = TransliterationDataset(train_data, src_vocab, tgt_vocab)
val_dataset = TransliterationDataset(val_data, src_vocab, tgt_vocab)
test_dataset = TransliterationDataset(test_data, src_vocab, tgt_vocab)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                         shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                       shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'],
                        shuffle=False, collate_fn=collate_fn)

# %% [markdown]
# ## 5. Model Architecture
# 
# The model consists of three required components:
# 1. **Embedding Layer** - Converts character indices to vectors
# 2. **Encoder RNN** - Processes input sequence
# 3. **Decoder RNN** - Generates output sequence

# %% [markdown]
# ### 5.1 Encoder

# %%
class Encoder(nn.Module):
    """RNN Encoder - supports RNN, LSTM, and GRU"""
    
    def __init__(self, input_size, embedding_dim, hidden_dim, 
                 num_layers, cell_type='LSTM', dropout=0.3):
        super(Encoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type
        
        self.embedding = nn.Embedding(input_size, embedding_dim, 
                                     padding_idx=CONFIG['PAD_token'])
        
        if cell_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_seq, lengths):
        embedded = self.dropout(self.embedding(input_seq))
        packed = pack_padded_sequence(embedded, lengths.cpu(), 
                                     batch_first=True, enforce_sorted=False)
        
        if self.cell_type == 'LSTM':
            packed_output, (hidden, cell) = self.rnn(packed)
            return pad_packed_sequence(packed_output, batch_first=True)[0], (hidden, cell)
        else:
            packed_output, hidden = self.rnn(packed)
            return pad_packed_sequence(packed_output, batch_first=True)[0], hidden

# %% [markdown]
# ### 5.2 Decoder

# %%
class Decoder(nn.Module):
    """RNN Decoder - generates output one character at a time"""
    
    def __init__(self, output_size, embedding_dim, hidden_dim,
                 num_layers, cell_type='LSTM', dropout=0.3):
        super(Decoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        
        self.embedding = nn.Embedding(output_size, embedding_dim,
                                     padding_idx=CONFIG['PAD_token'])
        
        if cell_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_token, hidden):
        embedded = self.dropout(self.embedding(input_token))
        
        if self.cell_type == 'LSTM':
            output, (hidden, cell) = self.rnn(embedded, hidden)
            hidden_state = (hidden, cell)
        else:
            output, hidden = self.rnn(embedded, hidden)
            hidden_state = hidden
        
        output = self.fc(output.squeeze(1))
        return output, hidden_state

# %% [markdown]
# ### 5.3 Complete Seq2Seq Model

# %%
class Seq2Seq(nn.Module):
    """Complete Seq2Seq model"""
    
    def __init__(self, input_vocab_size, output_vocab_size, 
                 embedding_dim, hidden_dim, num_encoder_layers,
                 num_decoder_layers, cell_type='LSTM', dropout=0.3):
        super(Seq2Seq, self).__init__()
        
        self.encoder = Encoder(input_vocab_size, embedding_dim, hidden_dim,
                               num_encoder_layers, cell_type, dropout)
        self.decoder = Decoder(output_vocab_size, embedding_dim, hidden_dim,
                               num_decoder_layers, cell_type, dropout)
        self.cell_type = cell_type
        
    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        output_vocab_size = self.decoder.output_size
        
        outputs = torch.zeros(batch_size, tgt_len, output_vocab_size).to(src.device)
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        decoder_input = torch.full((batch_size, 1), CONFIG['SOS_token'],
                                   dtype=torch.long, device=src.device)
        
        for t in range(tgt_len):
            output, hidden = self.decoder(decoder_input, hidden)
            outputs[:, t, :] = output
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            if teacher_force and t < tgt_len - 1:
                decoder_input = tgt[:, t].unsqueeze(1)
            else:
                decoder_input = output.argmax(1).unsqueeze(1)
        
        return outputs
    
    def predict(self, src, src_lengths, max_length=50):
        batch_size = src.size(0)
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        decoder_input = torch.full((batch_size, 1), CONFIG['SOS_token'],
                                   dtype=torch.long, device=src.device)
        predictions = []
        
        for _ in range(max_length):
            output, hidden = self.decoder(decoder_input, hidden)
            pred_token = output.argmax(1)
            predictions.append(pred_token)
            
            if (pred_token == CONFIG['EOS_token']).all():
                break
            
            decoder_input = pred_token.unsqueeze(1)
        
        return torch.stack(predictions, dim=1)

# %% [markdown]
# ## 6. Initialize Model

# %%
model = Seq2Seq(
    input_vocab_size=src_vocab.n_chars,
    output_vocab_size=tgt_vocab.n_chars,
    embedding_dim=CONFIG['embedding_dim'],
    hidden_dim=CONFIG['hidden_dim'],
    num_encoder_layers=CONFIG['num_encoder_layers'],
    num_decoder_layers=CONFIG['num_decoder_layers'],
    cell_type=CONFIG['cell_type'],
    dropout=CONFIG['dropout']
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model initialized with {total_params:,} trainable parameters")
print(f"Architecture: {CONFIG['cell_type']}-based Seq2Seq")

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=CONFIG['PAD_token'])
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# %% [markdown]
# ## 7. Training Functions

# %%
def train_epoch(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0
    
    for src, tgt, src_lengths, tgt_lengths in tqdm(dataloader, desc="Training"):
        src, tgt = src.to(device), tgt.to(device)
        src_lengths = src_lengths.to(device)
        
        optimizer.zero_grad()
        output = model(src, src_lengths, tgt, teacher_forcing_ratio)
        
        output = output[:, :-1, :].contiguous().view(-1, output.size(-1))
        tgt = tgt[:, 1:].contiguous().view(-1)
        
        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, tgt, src_lengths, tgt_lengths in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            src_lengths = src_lengths.to(device)
            
            output = model(src, src_lengths, tgt, teacher_forcing_ratio=0)
            output = output[:, :-1, :].contiguous().view(-1, output.size(-1))
            tgt = tgt[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def calculate_accuracy(model, dataloader, src_vocab, tgt_vocab, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for src, tgt, src_lengths, _ in dataloader:
            src = src.to(device)
            src_lengths = src_lengths.to(device)
            predictions = model.predict(src, src_lengths)
            
            for i in range(src.size(0)):
                pred_indices = [idx for idx in predictions[i].cpu().tolist() 
                               if idx not in [CONFIG['EOS_token'], CONFIG['PAD_token']]]
                tgt_indices = [idx for idx in tgt[i].cpu().tolist()
                              if idx not in [CONFIG['SOS_token'], CONFIG['EOS_token'], CONFIG['PAD_token']]]
                
                if pred_indices == tgt_indices:
                    correct += 1
                total += 1
    
    return correct / total if total > 0 else 0

def print_examples(model, dataloader, src_vocab, tgt_vocab, device, num_examples=5):
    model.eval()
    examples_shown = 0
    
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)
    
    with torch.no_grad():
        for src, tgt, src_lengths, _ in dataloader:
            src = src.to(device)
            src_lengths = src_lengths.to(device)
            predictions = model.predict(src, src_lengths)
            
            for i in range(min(src.size(0), num_examples - examples_shown)):
                src_chars = [idx for idx in src[i].cpu().tolist() 
                            if idx not in [CONFIG['PAD_token'], CONFIG['EOS_token']]]
                tgt_chars = [idx for idx in tgt[i].cpu().tolist()
                            if idx not in [CONFIG['SOS_token'], CONFIG['EOS_token'], CONFIG['PAD_token']]]
                pred_chars = [idx for idx in predictions[i].cpu().tolist()
                             if idx not in [CONFIG['EOS_token'], CONFIG['PAD_token']]]
                
                src_word = src_vocab.indices_to_word(src_chars)
                tgt_word = tgt_vocab.indices_to_word(tgt_chars)
                pred_word = tgt_vocab.indices_to_word(pred_chars)
                
                match = "✓" if pred_word == tgt_word else "✗"
                print(f"{match} Input: {src_word:20s} | Target: {tgt_word:20s} | Predicted: {pred_word}")
                
                examples_shown += 1
                if examples_shown >= num_examples:
                    break
            
            if examples_shown >= num_examples:
                break
    
    print("="*80 + "\n")

# %% [markdown]
# ## 8. Training Loop

# %%
train_losses = []
val_losses = []
best_val_loss = float('inf')

print("\n" + "="*80)
print("TRAINING")
print("="*80 + "\n")

for epoch in range(CONFIG['num_epochs']):
    print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
    
    train_loss = train_epoch(model, train_loader, optimizer, criterion,
                            device, CONFIG['teacher_forcing_ratio'])
    val_loss = evaluate(model, val_loader, criterion, device)
    scheduler.step(val_loss)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"✓ Best model saved")
    
    if (epoch + 1) % 5 == 0:
        print_examples(model, val_loader, src_vocab, tgt_vocab, device, 5)
    
    print()

# %% [markdown]
# ## 9. Results Visualization

# %%
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='s')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Validation Loss Over Time', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_plot.png', dpi=300)
plt.show()

# %% [markdown]
# ## 10. Final Evaluation

# %%
# Load best model
model.load_state_dict(torch.load('best_model.pt'))

test_loss = evaluate(model, test_loader, criterion, device)
test_accuracy = calculate_accuracy(model, test_loader, src_vocab, tgt_vocab, device)

print("\n" + "="*80)
print("FINAL EVALUATION")
print("="*80)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print()

print_examples(model, test_loader, src_vocab, tgt_vocab, device, 10)

# %% [markdown]
# ## 11. Model Summary

# %%
print("\n" + "="*80)
print("MODEL SUMMARY")
print("="*80)
print(f"\nArchitecture: {CONFIG['cell_type']}-based Seq2Seq")
print(f"Total Parameters: {total_params:,}")
print(f"Embedding Dimension: {CONFIG['embedding_dim']}")
print(f"Hidden Dimension: {CONFIG['hidden_dim']}")
print(f"Encoder Layers: {CONFIG['num_encoder_layers']}")
print(f"Decoder Layers: {CONFIG['num_decoder_layers']}")
print(f"\nBest Validation Loss: {best_val_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print("="*80)

# %% [markdown]
# ## 12. Conclusion
# 
# This notebook demonstrates a complete implementation of a Seq2Seq model for
# transliteration with all required components:
# 
# ✅ Character embeddings  
# ✅ Encoder RNN  
# ✅ Decoder RNN using encoder's final state  
# ✅ Configurable architecture (RNN/LSTM/GRU, layers, dimensions)  
# ✅ Theoretical analysis of computations and parameters  
# ✅ Training with teacher forcing  
# ✅ Proper evaluation metrics  
# ✅ Visualization of results  
# 
# **GitHub Repository:** https://github.com/navyasgr/Seq2Seq-Aksharantar-IITM-navya