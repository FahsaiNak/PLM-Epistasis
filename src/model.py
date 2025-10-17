"""
model.py
---------
Contains model architectures for proten language model (PLM) regression tasks.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel
from captum.attr import IntegratedGradients, InputXGradient, NoiseTunnel
from typing import Tuple

class PLMRegressor(nn.Module):
    """
    A regressor using a pre-trained language model (PLM) backbone and a
    multi-head attention pooling layer.

    This model aggregates token embeddings into a single feature vector by using
    multiple learnable "query" vectors. Each head learns to focus on different
    aspects of the sequence, and their outputs are concatenated to form a rich
    representation for the final prediction.

    Parameters
    ----------
    model_name : str
        Hugging Face model name (e.g., "facebook/esm2_t33_650M_UR50D").
    num_heads : int, optional
        The number of attention heads to use in the pooling layer. Default is 4.
    dropout : float, optional
        The dropout probability before the final regression head. Default is 0.2.
    """
    def __init__(self, model_name: str, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.plm = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.hidden_dim = self.plm.config.hidden_size
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        # Create multiple learnable query vectors, one for each attention head.
        self.query_vectors = nn.Parameter(torch.randn(self.num_heads, self.hidden_dim))

        # The input dimension for the regressor is the concatenated output of all heads.
        self.regressor = nn.Linear(self.num_heads * self.hidden_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, tuple]:
        """
        Performs the forward pass with multi-head attention pooling.

        Returns
        -------
        torch.Tensor
            The flattened regression output. Shape: (batch_size,).
        tuple
            A tuple of attention tensors from the base PLM model.
        """
        # Get token embeddings from the base PLM.
        outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_state = outputs.last_hidden_state # Shape: [Batch, Length, Dim]

        # --- Multi-Head Attention Pooling ---
        # 1. Compute attention scores for all heads simultaneously.
        #    last_hidden_state: [B, L, D], query_vectors.T: [D, H] -> attn_scores: [B, L, H]
        attn_scores = torch.matmul(last_hidden_state, self.query_vectors.T)
        
        # 2. Permute dimensions to group by head for masking and softmax.
        #    attn_scores: [B, L, H] -> [B, H, L]
        attn_scores = attn_scores.permute(0, 2, 1)

        # 3. Mask padding tokens to prevent them from contributing to the pooling.
        #    We expand the mask to match the scores' dimensions.
        attn_scores[attention_mask.unsqueeze(1).expand_as(attn_scores) == 0] = -1e9

        # 4. Compute attention weights for each head independently.
        #    Softmax is applied over the sequence length dimension (L).
        attn_weights = torch.softmax(attn_scores, dim=2) # Shape: [B, H, L]

        # 5. Calculate the weighted sum of embeddings for each head.
        #    attn_weights.unsqueeze(-1): [B, H, L, 1]
        #    last_hidden_state.unsqueeze(1): [B, 1, L, D]
        #    The product is broadcast to [B, H, L, D]. Summing over L gives [B, H, D].
        pooled_per_head = torch.sum(last_hidden_state.unsqueeze(1) * attn_weights.unsqueeze(-1), dim=2)

        # 6. Concatenate the outputs of all heads to form the final feature vector.
        #    pooled_per_head: [B, H, D] -> pooled_concat: [B, H * D]
        pooled_concat = pooled_per_head.view(-1, self.num_heads * self.hidden_dim)
        # --- End Pooling ---

        # Apply dropout and the final regression head.
        x = self.dropout(pooled_concat)
        preds = self.regressor(x).flatten()
        
        return preds, outputs.attentions