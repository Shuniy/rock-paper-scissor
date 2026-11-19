from src.llm.neural_agent import SimpleOpponentPredictorGame
from src.regret_learning.zero_sum_game import ZeroSumGame
from src.regret_learning.pure_strategy import PureStrategyGame
from src.regret_learning.mixed_strategy import MixedStrategyGame
from src.regret_learning.hedge import HedgeGame
from src.regret_learning.swap_regret import SwapRegretGame
from src.regret_learning.swap_learner import SwapLearnerGame
from src.regret_learning.compare_learners import CompareLearners
from src.constants import ACTION
import torch

def checking_regret_learning():
    # normal game
    print("Normal Game:")
    zero_sum_game = ZeroSumGame()
    zero_sum_game.play(rounds=100)
    
    # pure strategy game
    print("\nPure Strategy Game:")
    pure_strategy_game = PureStrategyGame()
    pure_strategy_game.play(rounds=100)
    
    # mixed strategy game
    print("\nMixed Strategy Game with Player B playing uniformly:")
    mixed_strategy_game = MixedStrategyGame()
    mixed_strategy_game.play_b_uniformly(rounds=100)
    
    print("\nMixed Strategy Game with Player B playing optimally:")
    mixed_strategy_game = MixedStrategyGame()
    mixed_strategy_game.play(rounds=100)
    
    # hedge game
    print("\nHedge Game with Player B playing fixed action paper for 100 rounds:")
    hedge_game = HedgeGame()
    hedge_game.play_fixed_b(action = ACTION.PAPER, rounds=100)
    
    print("\nHedge Game with Player B playing fixed action paper for 500 rounds:")
    hedge_game = HedgeGame()
    hedge_game.play_fixed_b(action = ACTION.PAPER, rounds=500)
    
    print("\nHedge Game with Player B playing fixed action paper for 1000 rounds:")
    hedge_game = HedgeGame()
    hedge_game.play_fixed_b(action = ACTION.PAPER, rounds=1000, verbose=False)
    
    print("\nHedge Game with Player B playing fixed action paper for 50 rounds and then switching to rock for 50 rounds:")
    hedge_game = HedgeGame()
    hedge_game.experiment_a(first_action=ACTION.PAPER, second_action=ACTION.ROCK, rounds=100, verbose=False, change_action_after=50)
    
    print("\nHedge Game with Player B playing fixed action paper for 50 rounds and then switching to rock for 50 rounds:")
    hedge_game = HedgeGame()
    hedge_game.experiment_b(first_action=ACTION.PAPER, second_action=ACTION.ROCK, rounds=100, verbose=False, change_action_after=50)
    
 
    print("\nHedge Game with Player a and b uses hedge algorithm:")
    hedge_game = HedgeGame()
    hedge_game.play(rounds=100, verbose=False)
    
    print("\nSwap Regret Game with Player a and b:")
    swap_regret_game = SwapRegretGame()
    swap_regret_game.play_history(rounds=100, verbose=False)
    
    print("\nSwap Learner Game with Player a and b using paper 1 round:")
    swap_learner_game = SwapLearnerGame()
    swap_learner_game.experiment(rounds=100, verbose=False)
    
    print("\nSwap Learner Game with Player a and b using paper 5 rounds:")
    swap_learner_game = SwapLearnerGame()
    swap_learner_game.experiment_5_round()
    
    print("\nSwap Learner Game with Player a and b using fixed action paper:")
    swap_learner_game = SwapLearnerGame()
    swap_learner_game.play_fixed_b(action=ACTION.PAPER, rounds=100, verbose=False)
    
    print("\nSwap Learner Game with Player a and b:")
    swap_learner_game = SwapLearnerGame()
    swap_learner_game.play(rounds=100, verbose=False)
    
    print("\nSwap Learner Game vs Hedge Game with Player a and b:")
    comparison_result = CompareLearners.results(rounds=1000, verbose=False)
    print("\nComparison Result:")
    for player, stats in comparison_result.items():
        print(f"Player {player}:")
        for metric, value in stats.items():
            print(f"  {metric}: {value}")

def check_neural():
    # simple neural network opponent predictor
    examples = torch.tensor([
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2]
        ])
    print("\nSimple Neural Network Opponent Predictor Game:")
    neural_game = SimpleOpponentPredictorGame()
    neural_game.play(examples, rounds=100)

    print("\nTransformer Neural Network Opponent Predictor Game:")
    transformer_game = SimpleOpponentPredictorGame(type="Transformer")
    transformer_game.play(examples, rounds=100)

def main():
    # checking_regret_learning()
    check_neural()

if __name__ == "__main__":
    main()