"""
Seq2Seq Model for Latin to Devanagari Transliteration
Aksharantar Dataset - IIT Madras Assignment

Author: Complete Implementation with All Requirements
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Model hyperparameters (all configurable)
    'embedding_dim': 128,        # m - embedding size
    'hidden_dim': 256,           # h - hidden state size
    'num_encoder_layers': 1,     # Number of encoder layers
    'num_decoder_layers': 1,     # Number of decoder layers
    'cell_type': 'LSTM',         # 'RNN', 'LSTM', or 'GRU'
    'dropout': 0.3,
    
    # Training hyperparameters
    'batch_size': 64,
    'learning_rate': 0.001,
    'num_epochs': 50,
    'teacher_forcing_ratio': 0.5,
    'max_grad_norm': 5.0,
    
    # Data parameters
    'max_length': 50,
    'min_freq': 2,
    'train_split': 0.8,
    'val_split': 0.1,
    
    # Special tokens
    'PAD_token': 0,
    'SOS_token': 1,
    'EOS_token': 2,
    'UNK_token': 3,
}

# ============================================================================
# VOCABULARY CLASS
# ============================================================================

class Vocabulary:
    """Builds and manages vocabulary for character-level sequences"""
    
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
        self.n_chars = 4  # Start with special tokens
        
    def add_word(self, word):
        """Add all characters from a word to vocabulary"""
        for char in word:
            self.char2count[char] += 1
    
    def build_vocab(self):
        """Build vocabulary from counted characters"""
        for char, count in self.char2count.items():
            if count >= self.min_freq and char not in self.char2index:
                self.char2index[char] = self.n_chars
                self.index2char[self.n_chars] = char
                self.n_chars += 1
    
    def word_to_indices(self, word):
        """Convert word to list of indices"""
        return [self.char2index.get(char, CONFIG['UNK_token']) for char in word]
    
    def indices_to_word(self, indices):
        """Convert list of indices to word"""
        return ''.join([self.index2char.get(idx, '<UNK>') for idx in indices])

# ============================================================================
# DATASET CLASS
# ============================================================================

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
        
        # Convert to indices
        src_indices = self.src_vocab.word_to_indices(src_word)
        tgt_indices = self.tgt_vocab.word_to_indices(tgt_word)
        
        # Add EOS token
        src_indices.append(CONFIG['EOS_token'])
        tgt_indices.append(CONFIG['EOS_token'])
        
        return (torch.tensor(src_indices, dtype=torch.long),
                torch.tensor(tgt_indices, dtype=torch.long))

def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    src_batch, tgt_batch = zip(*batch)
    
    # Get lengths
    src_lengths = torch.tensor([len(x) for x in src_batch])
    tgt_lengths = torch.tensor([len(x) for x in tgt_batch])
    
    # Pad sequences
    src_padded = pad_sequence(src_batch, batch_first=True, 
                              padding_value=CONFIG['PAD_token'])
    tgt_padded = pad_sequence(tgt_batch, batch_first=True,
                              padding_value=CONFIG['PAD_token'])
    
    return src_padded, tgt_padded, src_lengths, tgt_lengths

# ============================================================================
# ENCODER CLASS
# ============================================================================

class Encoder(nn.Module):
    """
    RNN Encoder that processes input sequence
    Supports RNN, LSTM, and GRU cells with configurable layers
    """
    
    def __init__(self, input_size, embedding_dim, hidden_dim, 
                 num_layers, cell_type='LSTM', dropout=0.3):
        super(Encoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type
        
        # Embedding layer
        self.embedding = nn.Embedding(input_size, embedding_dim,
                                     padding_idx=CONFIG['PAD_token'])
        
        # RNN layer (configurable cell type)
        if cell_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_seq, lengths):
        """
        Args:
            input_seq: (batch_size, seq_len)
            lengths: (batch_size,)
        Returns:
            outputs: (batch_size, seq_len, hidden_dim)
            hidden: hidden state(s) - shape depends on cell type
        """
        # Embedding
        embedded = self.dropout(self.embedding(input_seq))  # (batch, seq_len, emb_dim)
        
        # Pack padded sequence for efficiency
        packed = pack_padded_sequence(embedded, lengths.cpu(), 
                                     batch_first=True, enforce_sorted=False)
        
        # Pass through RNN
        if self.cell_type == 'LSTM':
            packed_output, (hidden, cell) = self.rnn(packed)
            # Return both hidden and cell state for LSTM
            return pad_packed_sequence(packed_output, batch_first=True)[0], (hidden, cell)
        else:
            packed_output, hidden = self.rnn(packed)
            return pad_packed_sequence(packed_output, batch_first=True)[0], hidden

# ============================================================================
# DECODER CLASS
# ============================================================================

class Decoder(nn.Module):
    """
    RNN Decoder that generates output sequence one character at a time
    Takes encoder's final hidden state as initial state
    """
    
    def __init__(self, output_size, embedding_dim, hidden_dim,
                 num_layers, cell_type='LSTM', dropout=0.3):
        super(Decoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        
        # Embedding layer
        self.embedding = nn.Embedding(output_size, embedding_dim,
                                     padding_idx=CONFIG['PAD_token'])
        
        # RNN layer
        if cell_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Output projection layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_token, hidden):
        """
        Args:
            input_token: (batch_size, 1) - single time step
            hidden: hidden state from previous step or encoder
        Returns:
            output: (batch_size, output_size) - probability distribution
            hidden: updated hidden state
        """
        # Embedding
        embedded = self.dropout(self.embedding(input_token))  # (batch, 1, emb_dim)
        
        # Pass through RNN
        if self.cell_type == 'LSTM':
            output, (hidden, cell) = self.rnn(embedded, hidden)
            hidden_state = (hidden, cell)
        else:
            output, hidden = self.rnn(embedded, hidden)
            hidden_state = hidden
        
        # Project to vocabulary size
        output = self.fc(output.squeeze(1))  # (batch, output_size)
        
        return output, hidden_state

# ============================================================================
# SEQ2SEQ MODEL
# ============================================================================

class Seq2Seq(nn.Module):
    """
    Complete Seq2Seq model combining encoder and decoder
    Supports configurable architecture parameters
    """
    
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
        """
        Args:
            src: (batch_size, src_len)
            src_lengths: (batch_size,)
            tgt: (batch_size, tgt_len)
            teacher_forcing_ratio: probability of using teacher forcing
        Returns:
            outputs: (batch_size, tgt_len, output_vocab_size)
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        output_vocab_size = self.decoder.output_size
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, output_vocab_size).to(src.device)
        
        # Encode entire input sequence
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # First input to decoder is SOS token
        decoder_input = torch.full((batch_size, 1), CONFIG['SOS_token'],
                                   dtype=torch.long, device=src.device)
        
        # Decode one character at a time
        for t in range(tgt_len):
            output, hidden = self.decoder(decoder_input, hidden)
            outputs[:, t, :] = output
            
            # Teacher forcing: use actual target as next input
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            if teacher_force and t < tgt_len - 1:
                decoder_input = tgt[:, t].unsqueeze(1)
            else:
                decoder_input = output.argmax(1).unsqueeze(1)
        
        return outputs
    
    def predict(self, src, src_lengths, max_length=50):
        """
        Greedy decoding for inference
        Args:
            src: (batch_size, src_len)
            src_lengths: (batch_size,)
            max_length: maximum output length
        Returns:
            predictions: (batch_size, pred_len)
        """
        batch_size = src.size(0)
        
        # Encode
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # Start with SOS token
        decoder_input = torch.full((batch_size, 1), CONFIG['SOS_token'],
                                   dtype=torch.long, device=src.device)
        
        predictions = []
        
        for _ in range(max_length):
            output, hidden = self.decoder(decoder_input, hidden)
            pred_token = output.argmax(1)
            predictions.append(pred_token)
            
            # Stop if all sequences have produced EOS
            if (pred_token == CONFIG['EOS_token']).all():
                break
            
            decoder_input = pred_token.unsqueeze(1)
        
        return torch.stack(predictions, dim=1)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device, 
                teacher_forcing_ratio):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    
    for src, tgt, src_lengths, tgt_lengths in tqdm(dataloader, desc="Training"):
        src, tgt = src.to(device), tgt.to(device)
        src_lengths = src_lengths.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, src_lengths, tgt, teacher_forcing_ratio)
        
        # Reshape for loss calculation
        output = output[:, :-1, :].contiguous().view(-1, output.size(-1))
        tgt = tgt[:, 1:].contiguous().view(-1)
        
        # Calculate loss
        loss = criterion(output, tgt)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
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
    """Calculate word-level accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for src, tgt, src_lengths, _ in dataloader:
            src = src.to(device)
            src_lengths = src_lengths.to(device)
            
            predictions = model.predict(src, src_lengths)
            
            for i in range(src.size(0)):
                pred_indices = predictions[i].cpu().tolist()
                tgt_indices = tgt[i].cpu().tolist()
                
                # Remove EOS and PAD tokens
                pred_indices = [idx for idx in pred_indices 
                               if idx not in [CONFIG['EOS_token'], CONFIG['PAD_token']]]
                tgt_indices = [idx for idx in tgt_indices
                              if idx not in [CONFIG['SOS_token'], CONFIG['EOS_token'], 
                                           CONFIG['PAD_token']]]
                
                if pred_indices == tgt_indices:
                    correct += 1
                total += 1
    
    return correct / total if total > 0 else 0

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_data(filepath):
    """Load transliteration data"""
    # Assuming CSV format with columns: source, target
    df = pd.read_csv(filepath, header=None, names=['source', 'target'])
    return list(zip(df['source'].values, df['target'].values))

def split_data(data, train_ratio=0.8, val_ratio=0.1):
    """Split data into train, val, test"""
    np.random.shuffle(data)
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return data[:train_end], data[train_end:val_end], data[val_end:]

def print_examples(model, dataloader, src_vocab, tgt_vocab, device, num_examples=5):
    """Print example predictions"""
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
                src_indices = src[i].cpu().tolist()
                tgt_indices = tgt[i].cpu().tolist()
                pred_indices = predictions[i].cpu().tolist()
                
                # Remove special tokens
                src_chars = [idx for idx in src_indices 
                            if idx not in [CONFIG['PAD_token'], CONFIG['EOS_token']]]
                tgt_chars = [idx for idx in tgt_indices
                            if idx not in [CONFIG['SOS_token'], CONFIG['EOS_token'], 
                                          CONFIG['PAD_token']]]
                pred_chars = [idx for idx in pred_indices
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

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': CONFIG
    }
    torch.save(checkpoint, filepath)

def plot_losses(train_losses, val_losses):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# THEORETICAL ANALYSIS
# ============================================================================

def calculate_parameters(model):
    """Calculate total number of parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def theoretical_analysis():
    """
    Print theoretical analysis of computations and parameters
    """
    m = CONFIG['embedding_dim']
    h = CONFIG['hidden_dim']
    V_src = 100  # Example vocab size
    V_tgt = 100
    n = 20  # Example sequence length
    
    print("\n" + "="*80)
    print("THEORETICAL ANALYSIS")
    print("="*80)
    print(f"\nGiven:")
    print(f"  - Embedding dimension (m) = {m}")
    print(f"  - Hidden dimension (h) = {h}")
    print(f"  - Sequence length (n) = {n}")
    print(f"  - Vocabulary size (V) = {V_src} (assuming same for source and target)")
    print(f"  - Cell type = {CONFIG['cell_type']}")
    print(f"  - Encoder layers = {CONFIG['num_encoder_layers']}")
    print(f"  - Decoder layers = {CONFIG['num_decoder_layers']}")
    
    # Computations
    print(f"\n{'─'*80}")
    print("TOTAL COMPUTATIONS (for 1 layer each):")
    print(f"{'─'*80}")
    
    if CONFIG['cell_type'] == 'LSTM':
        encoder_comp = n * 4 * (h*m + h*h)
        decoder_comp = n * 4 * (h*m + h*h)
        output_comp = n * h * V_tgt
        
        print(f"\n1. Encoder LSTM:")
        print(f"   - Per timestep: 4 gates × (h×m + h×h) = 4({h}×{m} + {h}×{h})")
        print(f"   - Per timestep: 4 × ({h*m:,} + {h*h:,}) = {4*(h*m + h*h):,}")
        print(f"   - Total: n × 4(hm + h²) = {n} × {4*(h*m + h*h):,} = {encoder_comp:,}")
        
        print(f"\n2. Decoder LSTM:")
        print(f"   - Per timestep: 4 gates × (h×m + h×h) = {4*(h*m + h*h):,}")
        print(f"   - Total: n × 4(hm + h²) = {decoder_comp:,}")
        
        print(f"\n3. Output Projection:")
        print(f"   - Per timestep: h × V = {h} × {V_tgt} = {h*V_tgt:,}")
        print(f"   - Total: n × h × V = {output_comp:,}")
        
        total_comp = encoder_comp + decoder_comp + output_comp
        print(f"\n{'─'*80}")
        print(f"TOTAL COMPUTATIONS: {total_comp:,}")
        print(f"{'─'*80}")
        print(f"Formula: O(n[8hm + 8h² + hV])")
        print(f"         = O({n}[8×{h}×{m} + 8×{h}² + {h}×{V_tgt}])")
        print(f"         = O({n}[{8*h*m:,} + {8*h*h:,} + {h*V_tgt:,}])")
        print(f"         = O({n} × {8*h*m + 8*h*h + h*V_tgt:,})")
        print(f"         ≈ O({total_comp:,})")
    
    elif CONFIG['cell_type'] == 'RNN':
        encoder_comp = n * (h*m + h*h)
        decoder_comp = n * (h*m + h*h)
        output_comp = n * h * V_tgt
        total_comp = encoder_comp + decoder_comp + output_comp
        
        print(f"\nFor RNN:")
        print(f"  - Encoder: n × (hm + h²) = {encoder_comp:,}")
        print(f"  - Decoder: n × (hm + h²) = {decoder_comp:,}")
        print(f"  - Output: n × hV = {output_comp:,}")
        print(f"  - TOTAL: O(n[2hm + 2h² + hV]) ≈ {total_comp:,}")
    
    # Parameters
    print(f"\n{'─'*80}")
    print("TOTAL PARAMETERS (for 1 layer each):")
    print(f"{'─'*80}")
    
    embedding_params = V_src * m
    print(f"\n1. Embedding Layer:")
    print(f"   - Parameters: V × m = {V_src} × {m} = {embedding_params:,}")
    
    if CONFIG['cell_type'] == 'LSTM':
        encoder_params = 4 * (h*m + h*h + h)
        decoder_params = 4 * (h*m + h*h + h)
        
        print(f"\n2. Encoder LSTM (1 layer):")
        print(f"   - 4 gates (input, forget, cell, output)")
        print(f"   - Each gate: input weights (h×m) + hidden weights (h×h) + bias (h)")
        print(f"   - Parameters: 4 × (h×m + h×h + h)")
        print(f"   - Parameters: 4 × ({h}×{m} + {h}×{h} + {h})")
        print(f"   - Parameters: 4 × ({h*m:,} + {h*h:,} + {h})")
        print(f"   - Parameters: {encoder_params:,}")
        
        print(f"\n3. Decoder LSTM (1 layer):")
        print(f"   - Parameters: 4 × (h×m + h×h + h) = {decoder_params:,}")
        
    elif CONFIG['cell_type'] == 'RNN':
        encoder_params = h*m + h*h + h
        decoder_params = h*m + h*h + h
        print(f"\n2. Encoder RNN: hm + h² + h = {encoder_params:,}")
        print(f"3. Decoder RNN: hm + h² + h = {decoder_params:,}")
    
    output_params = h * V_tgt + V_tgt
    print(f"\n4. Output Projection Layer:")
    print(f"   - Weights: h × V = {h} × {V_tgt} = {h*V_tgt:,}")
    print(f"   - Bias: V = {V_tgt}")
    print(f"   - Total: {output_params:,}")
    
    if CONFIG['cell_type'] == 'LSTM':
        total_params = embedding_params + encoder_params + decoder_params + output_params
        print(f"\n{'─'*80}")
        print(f"TOTAL PARAMETERS: {total_params:,}")
        print(f"{'─'*80}")
        print(f"\nFormula: V×m + 8h(m + h + 1) + hV + V")
        print(f"       = {V_src}×{m} + 8×{h}({m} + {h} + 1) + {h}×{V_tgt} + {V_tgt}")
        print(f"       = {embedding_params:,} + 8×{h}×{m+h+1} + {h*V_tgt:,} + {V_tgt}")
        print(f"       = {embedding_params:,} + {8*h*(m+h+1):,} + {h*V_tgt:,} + {V_tgt}")
        print(f"       = {total_params:,}")
        print(f"\nSimplified: V(m+1) + 8h(m+h+1) + hV")
    
    print("\n" + "="*80 + "\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("\n" + "="*80)
    print("SEQ2SEQ TRANSLITERATION MODEL - LATIN TO DEVANAGARI")
    print("Aksharantar Dataset - IIT Madras Assignment")
    print("="*80)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Print configuration
    print("\n" + "─"*80)
    print("MODEL CONFIGURATION:")
    print("─"*80)
    for key, value in CONFIG.items():
        print(f"  {key:25s}: {value}")
    print("─"*80)
    
    # Theoretical analysis
    theoretical_analysis()
    
    # Load data
    print("\nLoading data...")
    # Replace with actual data path
    # data = load_data('aksharantar_data.csv')
    
    # For demonstration purposes, create dummy data
    print("Creating dummy data for demonstration...")
    dummy_data = [
        ('ghar', 'घर'),
        ('ajanabee', 'अजनबी'),
        ('kitab', 'किताब'),
        ('paani', 'पानी'),
        ('dost', 'दोस्त'),
        ('raat', 'रात'),
        ('subah', 'सुबह'),
        ('duniya', 'दुनिया'),
    ] * 100  # Repeat to create more samples
    
    data = dummy_data
    
    # Split data
    train_data, val_data, test_data = split_data(data, 
                                                  CONFIG['train_split'],
                                                  CONFIG['val_split'])
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Build vocabularies
    print("\nBuilding vocabularies...")
    src_vocab = Vocabulary('source', min_freq=CONFIG['min_freq'])
    tgt_vocab = Vocabulary('target', min_freq=CONFIG['min_freq'])
    
    # Count characters in training data only
    for src, tgt in train_data:
        src_vocab.add_word(src)
        tgt_vocab.add_word(tgt)
    
    src_vocab.build_vocab()
    tgt_vocab.build_vocab()
    
    print(f"Source vocabulary size: {src_vocab.n_chars}")
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
    
    # Initialize model
    print("\nInitializing model...")
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
    
    # Print model summary
    total_params = calculate_parameters(model)
    print(f"\nModel initialized with {total_params:,} trainable parameters")
    print(f"Model architecture: {CONFIG['cell_type']}-based Seq2Seq")
    print(f"Encoder layers: {CONFIG['num_encoder_layers']}")
    print(f"Decoder layers: {CONFIG['num_decoder_layers']}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=CONFIG['PAD_token'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=5)
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80 + "\n")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                device, CONFIG['teacher_forcing_ratio'])
        
        # Validate
        val_loss = evaluate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss,
                          'best_model.pt')
            print(f"✓ Best model saved (Val Loss: {val_loss:.4f})")
        
        # Print examples every 5 epochs
        if (epoch + 1) % 5 == 0:
            print_examples(model, val_loader, src_vocab, tgt_vocab, device, 5)
        
        print()
    
    # Plot losses
    print("\nGenerating loss plot...")
    plot_losses(train_losses, val_losses)
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80 + "\n")
    
    # Load best model
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test accuracy
    test_loss = evaluate(model, test_loader, criterion, device)
    test_accuracy = calculate_accuracy(model, test_loader, src_vocab, 
                                      tgt_vocab, device)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    
    # Show test examples
    print_examples(model, test_loader, src_vocab, tgt_vocab, device, 10)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Final test accuracy: {test_accuracy*100:.2f}%")
    print(f"Model saved to: best_model.pt")
    print(f"Loss plot saved to: loss_plot.png")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()