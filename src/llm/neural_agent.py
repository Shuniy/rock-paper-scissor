from typing import Literal

import torch
import torch.nn as nn

from src.llm.tiny_transformer import TinyTransformer
from src.llm.batch_generator import make_batch
from src.llm.simple_neural_train_eval_predict import evaluate_model, predict, train_model
from src.constants import SEED
from src.llm.simple_opponent_predictor import SimpleOpponentPredictor

torch.manual_seed(SEED)  # For reproducibility

class SimpleOpponentPredictorGame(nn.Module):
    def __init__(self, type: Literal["MLP", "Transformer"] = "MLP"):
        super().__init__()
        # initialize the model
        if type == "MLP":
            self.model = SimpleOpponentPredictor()
        elif type == "Transformer":
            self.model = TinyTransformer()

        #optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

        self.train_data = make_batch(n=256, history_length=5)
        self.test_data = make_batch(n=256, history_length=5)

    def train(self, epochs=1000):
        self.model = train_model(self.model, self.train_data, self.optimizer, self.criterion, epochs=epochs)

    def evaluate(self):
        results = evaluate_model(self.model, self.test_data)
        return results

    def train_and_evaluate(self, rounds=100):
        self.train(epochs=rounds)
        stats = self.evaluate()
        print(f"Results after {rounds} rounds:")
        for metric, value in stats.items():
            print(f"  {metric}: {value}")

    def play(self, history, rounds=100):
        self.train_and_evaluate(rounds=rounds)
        # check what the model predicts for simple histories
        return predict(self.model, history)