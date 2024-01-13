import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

class SelfAttentionHead(nn.Module):
    
    # dropout rate like in the original paper
    def __init__(self, d_model, dropout_rate = 0.1):
                
        super().__init__()
        
        self.d_model = d_model
        
        # to produce K, Q, V
        self.to_keys = nn.Linear(d_model, d_model)
        self.to_queries = nn.Linear(d_model, d_model)
        self.to_values = nn.Linear(d_model, d_model)
        
        self.scale_factor = math.sqrt(d_model)
        
        # apply before adding and normalization
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        
        self.layer_norm_attention = nn.LayerNorm(d_model)
        self.layer_norm_ffn = nn.LayerNorm(d_model)
        
        # feed forward neural network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, x):
        
        B, T, d_model = x.shape
        
        K = self.to_keys(x)
        
        Q = self.to_queries(x)
        
        V = self.to_values(x)
        
        # K = (B, T, d_model)
        # Q = (B, T, d_model)
        # V = (B, T, d_model)
        
        # A = (B, T, d_model) x (B, d_model, T) = (B, T, T)
        A_inner = torch.matmul(Q, K.transpose(1, 2)) / self.scale_factor
        
        mask = torch.tril(torch.ones(T, T))
        mask = mask.masked_fill(mask == 0, float("-inf"))
        mask = mask.to(self.device)
        
        # don't use dim = 1 (that doesn't work as the attention matrix is batched)
        # use dim = -1 (always the last dimension)
        A_activated = F.softmax(A_inner + mask, dim = -1)
        
        # (B, T, T) x (B, T, d_model) = (B, T, d_model)
        after_attention = torch.matmul(A_activated, V)
        
        after_attention_dropout = self.attention_dropout(after_attention)
        
        # residual (skip) connection
        after_residual_connection = x + after_attention_dropout
        
        after_layer_norm = self.layer_norm_attention(after_residual_connection)
        
        after_ffn = self.ffn(after_layer_norm)
        
        after_ffn_drouput = self.ffn_dropout(after_ffn)
        
        after_ffn_residual_connection = after_ffn + after_layer_norm
        
        after_ffn_layer_norm = self.layer_norm_ffn(after_ffn_residual_connection)
        
        return after_ffn_layer_norm
    
    
class Transformer(nn.Module):

    # n_attention_units = N (in the paper)
    def __init__(self, seq_length, vocab_size, emb_size, n_attention_units):

        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = emb_size)

        self.positional_embedding = nn.Parameter(torch.randn(seq_length, emb_size))

        # we don't use multi head attention for now
        units_to_add = [SelfAttentionHead(d_model = emb_size) for _ in range(n_attention_units)]

        self.attention_units = nn.Sequential(*units_to_add)

        self.to_output = nn.Linear(emb_size, vocab_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def save_model(self, path):
        """
        Saves the state dictionary of a PyTorch model with success and error messages.

        Parameters:
        model (nn.Module): The PyTorch model to save.
        path (str): Path where the state dictionary will be saved.
        """
        try:
            torch.save(self.state_dict(), path)
            print(f"Model successfully saved to {path}")
        except Exception as e:
            print(f"Error saving model: {e}")


    def load_model(self, path):
        """
        Loads the state dictionary into a PyTorch model with success and error messages.

        Parameters:
        model (nn.Module): The PyTorch model where the state dictionary will be loaded.
        path (str): Path from where the state dictionary will be loaded.
        """
        try:
            self.load_state_dict(torch.load(path, map_location=model.device))
            print(f"Model state loaded successfully from {path}")
        except FileNotFoundError:
            print(f"Error: No file found at {path}")
        except Exception as e:
            print(f"Error loading model: {e}")


    # works only with batches
    def forward(self, x):

        embedded = self.token_embedding(x)

        B, T, E = embedded.shape

        after_positional_encoding = embedded + self.positional_embedding[:T, :]

        after_attention = self.attention_units(after_positional_encoding)

        logits = self.to_output(after_attention)

        return logits