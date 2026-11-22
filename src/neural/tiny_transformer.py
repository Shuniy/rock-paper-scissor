import torch
import torch.nn as nn

class TinyTransformer(nn.Module):
    def __init__(self, sequence_length=5, d_model=16):
        super().__init__()

        # embed rock, paper and scissors into a d_model-dimensional space
        self.token_embedding = nn.Embedding(num_embeddings=3, embedding_dim=d_model)

        # learn the position of each element in the sequence
        self.position_embedding = nn.Embedding(num_embeddings=sequence_length, embedding_dim=d_model)

        # one transformer encoder layer
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=4,
            dim_feedforward=64,
            dropout=0.0,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor = False)
        self.output = nn.Linear(d_model, 3)  # output logits for rock, paper, scissors


    def forward(self, history):
        batch_size, seq_len = history.size()
        # create position indices
        positions = torch.arange(seq_len, device=history.device)
        # combine the action and position embeddings
        x = self.token_embedding(history) + self.position_embedding(positions)

        # prevent attention to future tokens
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=history.device, dtype=torch.bool), diagonal=1)
        # process sequence through the transformer
        h = self.transformer(x, mask=causal_mask)
        # take the final position to predict the next action
        logits = self.output(h[:, -1, :])
        return logits

