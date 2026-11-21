import torch

from src.constants import ACTIONS, PAYOFF_TENSOR

def train_model(model, train_data, optimizer, criterion, epochs=1000):
    """
    Train the opponent predictor model.
    :param model: The opponent predictor model to be trained.
    :param train_data: Tuple of (history, target) tensors for training.
    :param optimizer: Optimizer for updating model parameters.
    :param criterion: Loss function for calculating prediction error.
    :param epochs: Number of training epochs.
    """
    model.train()
    for epoch in range(epochs):
        history, action_b = train_data

        # predict opponent action
        logits = model(history)

        # convert to a's action probabilities using softmax
        probabilities_a = torch.softmax(logits, dim=1)

        # rewards for each possible action a given b's action
        expected_rewards = PAYOFF_TENSOR[:, action_b].T

        # expected reward for each possible a action
        expected_rewards = (probabilities_a * expected_rewards).sum(dim=1).mean()

        # maximise reward = minimise negative reward
        loss = -expected_rewards

        # optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    return model


def evaluate_model(model, test_data):
    """
    Evaluate the opponent predictor model on test data.
    :param model: The trained opponent predictor model.
    :param test_data: Tuple of (history, target) tensors for testing.
    :return: Dictionary containing evaluation metrics.
    """
    model.eval()
    with torch.no_grad():
        history, action_b = test_data
        logits = model(history)
        probabilities_a = torch.softmax(logits, dim=1)
        rewards = PAYOFF_TENSOR[:, action_b].T
        expected_rewards = (probabilities_a * rewards).sum(dim=1).mean()
        print(f'Expected Reward: {expected_rewards.item():.4f}')
    return {
        "expected_reward": expected_rewards.item(),
        "predictions": probabilities_a,
        "targets": action_b
    }

def predict(model, history):
    """
    Predict the opponent's next action based on the given history.
    :param model: Trained opponent predictor model.
    :param history: Tensor of shape (batch_size, history_length) representing the history of opponent moves.
    :return: Tensor of shape (batch_size,) representing the predicted next actions.
    """
    result = {}
    model.eval()
    with torch.no_grad():
        logits = model(history)
        probabilities_a = torch.softmax(logits, dim=1)
        for i in range(3):
            print(f"\nB history: {ACTIONS[i]} x5")
            print("A's strategy:", probabilities_a[i].numpy().round(3))
            best_action = probabilities_a[i].argmax().item()
            print("A prefers:", ACTIONS[int(best_action)])
            result[ACTIONS[i]] = {
                "probabilities": probabilities_a[i],
                "best_action": ACTIONS[int(best_action)]
            }
    return result