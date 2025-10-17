"""
model.py
---------
Contains model architectures for protein language model (PLM) tasks.
"""

import torch
from torch import nn
from transformers import AutoModel
from typing import Tuple

class PLMRegressor(nn.Module):
    """
    A regressor using a pre-trained language model (PLM) backbone with a
    simple linear head on top of the [CLS] token representation.

    This model is designed for sequence-level regression tasks. It uses the
    PLM's dedicated 'pooler_output' as the feature vector for the entire
    protein sequence.

    Parameters
    ----------
    model_name : str
        Hugging Face model name (e.g., "facebook/esm2_t33_650M_UR50D").
    dropout : float, optional
        The dropout probability before the final regression head. Default is 0.2.
    """
    def __init__(self, model_name: str, dropout: float = 0.2):
        super().__init__()
        # Load the pre-trained PLM and configure it to output attentions
        self.plm = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.dropout = nn.Dropout(dropout)
        
        # The regression head takes the PLM's hidden dimension as input
        self.regressor = nn.Linear(self.plm.config.hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, tuple]:
        """
        Performs the forward pass using CLS token pooling.

        Returns
        -------
        torch.Tensor
            The flattened regression output. Shape: (batch_size,).
        tuple
            A tuple of attention tensors from all layers of the base PLM.
        """
        # Get the outputs from the base PLM.
        # `return_dict=True` provides a structured output object.
        outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        # --- CLS Pooling ---
        # The `pooler_output` is a dedicated representation of the [CLS] token,
        # often passed through an additional linear layer and activation function.
        # It is designed for sequence-level tasks.
        # Shape: [Batch, Hidden_Dim]
        pooled_output = outputs.pooler_output
        
        # --- End Pooling ---

        # Apply dropout and the final regression head.
        x = self.dropout(pooled_output)
        preds = self.regressor(x).flatten()
        
        return preds, outputs.attentions

class PLMClassifier(nn.Module):
    """
    A PLM classifier with a flexible prediction head.

    The architecture of the prediction head is determined by the `head_hidden_dim`
    parameter. If an integer is provided, a Multi-Layer Perceptron (MLP) head
    is used. If set to None, a simpler single linear layer is used.

    Parameters
    ----------
    model_name : str
        Hugging Face model name (e.g., "facebook/esm2_t33_650M_UR50D").
    num_classes : int
        The number of output classes for the classification task.
    head_hidden_dim : int, optional
        The size of the hidden layer within the MLP head. If set to 0,
        the model will use a single linear layer instead of an MLP.
        Default is 512.
    dropout : float, optional
        The dropout rate for the prediction head. Default is 0.2.
    """
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        head_hidden_dim: int = 512,
        dropout: float = 0.2
    ):
        super().__init__()
        self.plm = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.dropout = nn.Dropout(dropout)

        if head_hidden_dim == 0:
            # Build the linear head
            self.classifier = nn.Sequential(
                self.dropout,
                nn.Linear(self.plm.config.hidden_size, num_classes)
            )
        else:
            # Build the MLP head
            self.classifier = nn.Sequential(
                nn.Linear(self.plm.config.hidden_size, head_hidden_dim),
                nn.ReLU(),
                self.dropout,
                nn.Linear(head_hidden_dim, num_classes)
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, tuple]:
        """
        Performs the forward pass using CLS token pooling.
        """
        outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        # Use the standard CLS token's pooler_output for sequence representation
        pooled_output = outputs.pooler_output

        # Pass the pooled output through the selected classifier head
        logits = self.classifier(pooled_output)
        
        return logits, outputs.attentions