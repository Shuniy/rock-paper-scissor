import torch
import torch.nn as nn
from src.constants import SEED

torch.manual_seed(SEED)  # For reproducibility

class SimpleOpponentPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(3, 8)  # 3 actions, embedding size 8
        self.network = nn.Sequential(
            nn.Linear(8 * 5, 32),  # history_length = 5
            nn.ReLU(),
            nn.Linear(32, 3)  # 3 actions
        )

    def forward(self, history):
        """
        Forward pass of the model.
        :param history: Tensor of shape (batch_size, history_length) containing the opponent's past moves.
        :return: Tensor of shape (batch_size, 3) containing the predicted probabilities for each action.
        """
        x = self.embedding(history)  # shape: (batch_size, history_length, embedding_dim)
        x = x.flatten(start_dim=1)  # shape: (batch_size, history_length * embedding_dim)
        return self.network(x)  # shape: (batch_size, 3)
