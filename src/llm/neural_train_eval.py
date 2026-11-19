import torch
import torch.nn as nn
from src.constants import SEED

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
