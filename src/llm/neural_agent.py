import torch
import torch.nn as nn

from src.llm.batch_generator import make_batch
from src.llm.neural_train_eval import evaluate_model, train_model
from src.constants import ACTIONS, PAYOFF_TENSOR, SEED
from src.llm.SimpleOpponentPredictor import SimpleOpponentPredictor

torch.manual_seed(SEED)  # For reproducibility

class SimpleOpponentPredictorGame(nn.Module):
    def __init__(self):
        super().__init__()
        # initialize the model
        self.model = SimpleOpponentPredictor()

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

    def play(self, rounds=100):
        self.train(epochs=1000)
        stats = self.evaluate()
        print(f"Results after {rounds} rounds:")
        for metric, value in stats.items():
            print(f"  {metric}: {value}")

    def example(self):
        # check what the model predicts for simple histories
        examples = torch.tensor([
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2]
        ])

        with torch.no_grad():
            probabilities_b = self.model(examples).softmax(dim=1)
            # Expected reward for each possible A action
            expected_rewards_a = probabilities_b @ PAYOFF_TENSOR.T
            best_actions_a = expected_rewards_a.argmax(dim=1)
            for i in range(3):
                print(
                    f"\nB history: {ACTIONS[i]} x5"
                )
                print(
                    "Predicted B probabilities:",
                    probabilities_b[i].numpy().round(3)
                )
                print(
                    "Best action for A:",
                    ACTIONS[best_actions_a[i].item()]
                )
