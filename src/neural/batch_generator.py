import torch


def make_batch(n=256, history_length=5, generator=None):
    """
    Generate a batch of opponent moves for training.
    Each opponent has a preferred action, and they follow it with 85% probability.
    :param n: Number of opponents in the batch.
    :param history_length: Length of the history to generate.
    :return: Tuple of (history, target) where history is a tensor of shape (n, history_length) and target is a tensor of shape (n,).
    """
    # Each opponent has a preferred action
    favorite = torch.randint(0, 3, (n, 1), generator=generator)

    #  generate random actions
    random_moves = torch.randint(0, 3, (n, history_length + 1), generator=generator)

    # Follow preferred action 85% of the time
    follow_favorite = torch.rand(n, history_length + 1, generator=generator) < 0.85

    # follow preferred action with 85% probability, otherwise random
    moves = torch.where(
        follow_favorite,
        favorite,
        random_moves
    )
    # return the first history_length moves as input and the last move as target
    return moves[:, :-1], moves[:, -1]
