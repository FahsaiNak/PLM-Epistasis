"""
model.py
---------
Contains flexible model architectures for PLM-based tasks, allowing users
to choose between CLS and mean pooling strategies for sequence representation.
"""

import torch
from torch import nn
from transformers import AutoModel
from typing import Tuple, Optional

class PLMRegressor(nn.Module):
    """
    A PLM regressor with a switchable pooling strategy.

    Parameters
    ----------
    model_name : str
        Hugging Face model name (e.g., "facebook/esm2_t33_650M_UR50D").
    pooling_strategy : str, optional
        The pooling strategy to use: 'cls' or 'mean'. Default is 'mean'.
    head_hidden_dim : int or None, optional
        Hidden dimension for an optional MLP head. If 0 or None, uses a
        simple linear head. Default is 0.
    dropout : float, optional
        Dropout probability for the prediction head. Default is 0.2.
    """
    def __init__(
        self,
        model_name: str,
        pooling_strategy: str = 'cls',
        head_hidden_dim: Optional[int] = 0,
        dropout: float = 0.2
    ):
        super().__init__()
        self.pooling_strategy = pooling_strategy
        self.plm = AutoModel.from_pretrained(model_name, output_attentions=True)
        
        # Build the regressor head (linear or MLP)
        if head_hidden_dim is not None and head_hidden_dim > 0:
            self.regressor = nn.Sequential(
                nn.Linear(self.plm.config.hidden_size, head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, 1)
            )
        else:
            self.regressor = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.plm.config.hidden_size, 1)
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, tuple]:
        """
        Performs the forward pass using the selected pooling strategy.
        """
        outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        if self.pooling_strategy == 'cls':
            pooled_output = outputs.pooler_output
        elif self.pooling_strategy == 'mean':
            pooled_output = outputs.last_hidden_state.mean(dim=1)
        elif self.pooling_strategy == 'max':
            pooled_output = outputs.last_hidden_state.max(dim=1).values
        else:
            raise ValueError("pooling_strategy must be 'cls', 'mean' or 'max'")

        preds = self.regressor(pooled_output).flatten()
        return preds, outputs.attentions


class PLMClassifier(nn.Module):
    """
    A PLM classifier with a switchable pooling strategy.

    Parameters
    ----------
    model_name : str
        Hugging Face model name (e.g., "facebook/esm2_t33_650M_UR50D").
    num_classes : int
        The number of output classes.
    pooling_strategy : str, optional
        The pooling strategy to use: 'cls' or 'mean'. Default is 'mean'.
    head_hidden_dim : int or None, optional
        Hidden dimension for an optional MLP head. If 0 or None, uses a
        simple linear head. Default is 512.
    dropout : float, optional
        Dropout probability for the prediction head. Default is 0.2.
    """
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pooling_strategy: str = 'cls',
        head_hidden_dim: Optional[int] = 512,
        dropout: float = 0.2
    ):
        super().__init__()
        self.pooling_strategy = pooling_strategy
        self.plm = AutoModel.from_pretrained(model_name, output_attentions=True)
        
        # Build the classifier head (linear or MLP)
        if head_hidden_dim is not None and head_hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(self.plm.config.hidden_size, head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.plm.config.hidden_size, num_classes)
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, tuple]:
        """
        Performs the forward pass using the selected pooling strategy.
        """
        outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        if self.pooling_strategy == 'cls':
            pooled_output = outputs.pooler_output
        elif self.pooling_strategy == 'mean':
            pooled_output = outputs.last_hidden_state.mean(dim=1)
        elif self.pooling_strategy == 'max':
            pooled_output = outputs.last_hidden_state.max(dim=1).values
        else:
            raise ValueError("pooling_strategy must be 'cls', 'mean' or 'max'")

        logits = self.classifier(pooled_output)
        return logits, outputs.attentions
