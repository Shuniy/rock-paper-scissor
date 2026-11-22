import torch
import torch.nn as nn
from itertools import product
from src.constants import PAYOFF_TENSOR, SEED

torch.manual_seed(SEED)  # For reproducibility

H = 5       # Previous actions visible to the model
T = 30      # Rounds per episode

rng = torch.Generator().manual_seed(SEED)  # For reproducibility

# STEP 1: Generate complete game episodes
def generate_episodes(batch, generator=None):
    if generator is None:
        generator = rng
    length = H + T
    # Generate random moves for both players
    random_moves = torch.randint(0, 3, (batch, length), generator=generator)
    # Generate random favorite actions for player A
    favorite = torch.randint(0, 3, (batch, 1), generator=generator)
    # Generate random phases for each episode
    phase = torch.randint(0, 3, (batch, 1), generator=generator)  # 0 for phase 1, 1 for phase 2
    t = torch.arange(length)[None, :]
    # Four opponent behaviours
    mode = (torch.arange(batch) % 4)[:, None]
    fixed = favorite.expand(batch, length)
    switched = torch.where(
        t < H + T // 2,
        favorite,
        (favorite + 1) % 3
    )
    cycling = (t + phase) % 3
    preferred = torch.where(
        mode == 0,
        fixed,
        torch.where(mode == 1, switched, cycling)
    )
    # Follow the pattern 85% of the time.
    # Mode 3 uses uniformly random actions.
    follow = (
        (torch.rand(batch, length, generator=generator) < 0.85)
        & (mode != 3)
    )

    moves = torch.where(follow, preferred, random_moves)

    # Each decision uses only the previous H moves
    histories = torch.stack(
        [moves[:, i:i + H] for i in range(T)],
        dim=1
    )

    opponents = moves[:, H:]
    return histories, opponents

# STEP 2: Transformer policy
class TinyTransformer(nn.Module):

    def __init__(self):
        super().__init__()

        d_model = 16

        self.token_embedding = nn.Embedding(3, d_model)
        self.position_embedding = nn.Embedding(H, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=64,
            dropout=0.0,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=1,
            enable_nested_tensor=False
        )

        self.output = nn.Linear(d_model, 3)

    def forward(self, history):

        length = history.size(1)

        positions = torch.arange(
            length, device=history.device
        )

        x = (
            self.token_embedding(history)
            + self.position_embedding(positions)
        )

        mask = torch.triu(
            torch.ones(
                length, length,
                dtype=torch.bool,
                device=history.device
            ),
            diagonal=1
        )

        h = self.transformer(x, mask=mask)

        return self.output(h[:, -1, :])


# STEP 3: Calculate reward and swap-regret loss
def calculate_metrics(probabilities, opponent_actions):

    # Shape: [batch, rounds, 3]
    rewards = PAYOFF_TENSOR.T[opponent_actions]

    # Expected reward per episode
    expected_reward = (
        probabilities * rewards
    ).sum(dim=-1).mean(dim=-1)

    # Pairwise replacement gains:
    # delta[source, destination]
    delta = (
        rewards[:, :, None, :]
        - rewards[:, :, :, None]
    )

    # Expected swap gains per episode
    swap_matrix = (
        probabilities[:, :, :, None] * delta
    ).mean(dim=1)

    # Find best replacement for each original action
    swap_regret = (
        swap_matrix.max(dim=-1).values
    ).sum(dim=-1)

    return expected_reward.mean(), swap_regret.mean()


# STEP 4: Train one model
def train_model(lambda_swap):

    torch.manual_seed(42)
    training_rng = torch.Generator().manual_seed(SEED)

    model = TinyTransformer()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.003
    )

    for step in range(1, 151):

        model.train()

        histories, opponent_actions = generate_episodes(batch=32, generator=training_rng)

        # Shape [32, 30, 5] -> [960, 5]
        x = histories.reshape(-1, H)

        # A's policy for every round
        probabilities = model(x).softmax(dim=-1)

        # Shape [32, 30, 3]
        probabilities = probabilities.reshape(32, T, 3)

        reward, swap = calculate_metrics(
            probabilities, opponent_actions
        )

        # Combined training objective
        loss = -reward + lambda_swap * swap

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step in [1, 50, 100, 150]:
            print(
                f"Lambda={lambda_swap}, Step={step}: "
                f"Reward={reward.item():.3f}, "
                f"Swap={swap.item():.3f}, "
                f"Loss={loss.item():.3f}"
            )

    return model


# STEP 5: Evaluate on unseen episodes
def evaluate(model):

    model.eval()

    with torch.no_grad():

        evaluation_rng = torch.Generator().manual_seed(SEED + 1)
        histories, opponent_actions = generate_episodes(batch=200, generator=evaluation_rng)

        probabilities = model(
            histories.reshape(-1, H)
        ).softmax(dim=-1).reshape(200, T, 3)

        reward, swap = calculate_metrics(
            probabilities, opponent_actions
        )

    return reward.item(), swap.item()


def evaluate_realized(probabilities, opponents, random_values):

    # probabilities: [batch, rounds, 3]
    # opponents: [batch, rounds]

    batch, rounds, _ = probabilities.shape

    # Sample actions using fixed random values
    cumulative = probabilities.cumsum(dim=-1)

    actions = (
        random_values.unsqueeze(-1) > cumulative
    ).sum(dim=-1).clamp(max=2)

    # Rewards for every possible action
    # Shape: [batch, rounds, 3]
    rewards = PAYOFF_TENSOR.T[opponents]

    # Actual rewards of sampled actions
    actual_rewards = rewards.gather(
        dim=-1,
        index=actions.unsqueeze(-1)
    ).squeeze(-1)

    total_rewards = actual_rewards.sum(dim=1)

    # Best fixed action per episode
    fixed_rewards = rewards.sum(dim=1)

    best_fixed_rewards = fixed_rewards.max(
        dim=-1
    ).values

    # External regret per round
    external_regret = (
        best_fixed_rewards - total_rewards
    ) / rounds

    # One-hot encoding of chosen actions
    chosen = nn.functional.one_hot(
        actions, num_classes=3
    ).float()

    # Replacement gains:
    # [batch, rounds, source, destination]
    gains = (
        rewards[:, :, None, :]
        - rewards[:, :, :, None]
    )

    # Only include rounds when source was selected
    swap_matrix = (
        chosen.unsqueeze(-1) * gains
    ).sum(dim=1)

    # Best replacement for each original action
    swap_regret = (
        swap_matrix.max(dim=-1).values
    ).sum(dim=-1) / rounds

    # Validate calculations
    assert torch.all(
        swap_regret >= external_regret - 1e-5
    )

    return {
        "realized_reward": total_rewards.mean().item() / rounds,
        "external_regret": external_regret.mean().item(),
        "swap_regret": swap_regret.mean().item()
    }


def experiment_swap_a():
    # STEP 6: Compare two training objectives
    for lambda_swap in [0.0, 1.0]:

        model = train_model(lambda_swap)

        reward, swap = evaluate(model)

        print(f"\nTEST: lambda={lambda_swap}")
        print(f"Expected reward/round: {reward:.4f}")
        print(f"Expected swap surrogate/round: {swap:.4f}")

@torch.no_grad()
def choose_b(model, history, method, horizon=3):

    # Strategy 1: maximize immediate reward
    if method == "myopic":

        p = model(history.unsqueeze(0)).softmax(-1)[0]

        # B's expected rewards for Rock, Paper, Scissors
        b_rewards = -(p @ PAYOFF_TENSOR)

        return int(b_rewards.argmax())

    # Strategy 2: plan multiple rounds ahead
    plans = torch.tensor(
        list(product(range(3), repeat=horizon))
    )

    # Simulate each possible sequence of B's actions
    h = history.unsqueeze(0).repeat(len(plans), 1)

    b_total = torch.zeros(len(plans))

    for k in range(horizon):

        # Predict A's response to current simulated history
        p = model(h).softmax(-1)

        # B's action in each candidate plan
        b = plans[:, k]

        # Expected reward for B
        b_total += -(p * PAYOFF_TENSOR[:, b].T).sum(-1)

        # Add B's action to the history
        h = torch.cat([h[:, 1:], b[:, None]], dim=1)

    # Choose the first action of the best plan
    best_plan = b_total.argmax()

    return int(plans[best_plan, 0])


@torch.no_grad()
def play_episode(model, method, seed, rounds=30):

    rng = torch.Generator().manual_seed(seed)

    history = torch.randint(
        0, 3, (H,), generator=rng
    )

    # Same randomness for both opponent strategies
    draws = torch.rand(rounds, generator=rng)

    expected_b_reward = 0.0
    actual_b_reward = 0.0

    for t in range(rounds):

        # A's current strategy
        p = model(history.unsqueeze(0)).softmax(-1)[0]

        # B chooses strategically
        b = choose_b(model, history, method)

        # Sample A's action
        a = int(
            torch.searchsorted(
                p.cumsum(0), draws[t]
            ).clamp(max=2)
        )

        # B receives the opposite reward to A
        expected_b_reward += float(
            -(p * PAYOFF_TENSOR[:, b]).sum()
        )

        actual_b_reward += float(-PAYOFF_TENSOR[a, b])

        # Update history observed by A
        history = torch.cat([
            history[1:],
            torch.tensor([b])
        ])

    return (
        expected_b_reward / rounds,
        actual_b_reward / rounds
    )

def experiment_swap_b():
        
    # Train both models
    reward_model = train_model(lambda_swap=0.0)
    swap_model = train_model(lambda_swap=1.0)

    evaluation_rng = torch.Generator().manual_seed(SEED + 1)
    histories, opponents = generate_episodes(batch=200, generator=evaluation_rng)

    # Shared random numbers for action sampling
    sample_rng = torch.Generator().manual_seed(123)

    random_values = torch.rand(
        (200, T),
        generator=sample_rng
    )

    for name, model in [
        ("Reward only", reward_model),
        ("Reward + Swap", swap_model)
    ]:

        model.eval()

        with torch.no_grad():

            probabilities = model(
                histories.reshape(-1, H)
            ).softmax(dim=-1).reshape(200, T, 3)

            # Expected-policy metrics
            reward, swap_surrogate = calculate_metrics(
                probabilities,
                opponents
            )

            # Realized metrics
            results = evaluate_realized(
                probabilities,
                opponents,
                random_values
            )

            print(f"\n{name}")
            print(f"Expected reward: {reward.item():.4f}")
            print(
                f"Swap surrogate: "
                f"{swap_surrogate.item():.4f}"
            )

            print(
                f"Realized reward: "
                f"{results['realized_reward']:.4f}"
            )

            print(
                f"External regret: "
                f"{results['external_regret']:.4f}"
            )

            print(
                f"Realized swap regret: "
                f"{results['swap_regret']:.4f}"
            )

def experiment_c():
    # Train both models
    reward_model = train_model(lambda_swap=0.0)
    swap_model = train_model(lambda_swap=1.0)
    for name, model in [
        ("Reward only", reward_model),
        ("Reward + Swap", swap_model)
    ]:

        model.eval()

        print(f"\n{name}")

        for method in ["myopic", "planner"]:

            results = [
                play_episode(
                    model,
                    method,
                    seed=100 + i,
                    rounds=30
                )
                for i in range(10)
            ]

            expected = sum(r[0] for r in results) / len(results)
            realized = sum(r[1] for r in results) / len(results)

            print(
                f"{method}: "
                f"B expected reward={expected:.4f}, "
                f"B realized reward={realized:.4f}"
            )
