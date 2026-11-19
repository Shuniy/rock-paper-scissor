import torch
from src.constants import ACTIONS, PAYOFF_TENSOR, SEED

torch.manual_seed(SEED)  # For reproducibility

# train the model
def train_model(model, train_data, optimizer, criterion, epochs=1000, batch_size=256, history_length=5):
    """
    Train the opponent predictor model.
    :param epochs: Number of training epochs.
    :param batch_size: Number of samples per batch.
    :param history_length: Length of the history to generate.
    :return: Trained model.
    """
    for epoch in range(epochs):
        history, target = train_data

        # predict opponent action
        logits = model(history)

        # calculate prediction error
        loss = criterion(logits, target)

        # update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    return model

def evaluate_model(model, test_data):
    # evaluate on new examples
    model.eval()
    with torch.no_grad():
        history, target = test_data
        logits = model(history)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == target).float().mean().item()
        print(f'Accuracy: {accuracy * 100:.2f}%')
    return {
        "accuracy": accuracy,
        "predictions": predictions,
        "targets": target
    }

def predict(model, history):
    """
    Predict the opponent's next action based on the given history.
    :param model: Trained opponent predictor model.
    :param history: Tensor of shape (batch_size, history_length) representing the history of opponent moves.
    :return: Tensor of shape (batch_size,) representing the predicted next actions.
    """
    result = {}
    with torch.no_grad():
        probabilities = model(history).softmax(dim=1)
        # Expected reward for each possible A action
        expected_rewards = probabilities @ PAYOFF_TENSOR.T
        best_actions = expected_rewards.argmax(dim=1)
        for i in range(3):
            print(f"\nB history: {ACTIONS[i]} x5")
            print("Predicted B probabilities:", probabilities[i].numpy().round(3))
            print("Best action for A:", ACTIONS[best_actions[i].item()])

        result["best_actions"] = best_actions
        result["probabilities"] = probabilities
    return result

