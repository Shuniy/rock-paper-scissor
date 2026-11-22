from src.neural.rps_blackbox_expected_rollouts import rps_blackbox_expected_rollouts
from src.neural.rps_blackbox_dynamics import rps_blackbox_dynamics
from src.neural.blackbox_opponent import blackbox_experiment
from src.neural.online_transformer_rps_stationary_benchmark import online_transformer_rps_stationary_benchmark
from src.neural.online_transformer_rps_model_mismatch import online_transformer_rps_model_mismatch
from src.neural.horizon_strategic_online_transformer_planner import experiment_horizon_strategic_online_transformer_planner
from src.neural.strategic_online_transformer_planner import experiment_strategic_online_transformer_planner
from src.neural.online_transformer_planner import experiment_online_transformer_planner
from src.neural.online_neural_learning import play_experiment_online_learning_aware
from src.neural.swap_regret_transformer import experiment_swap_a, experiment_swap_b
from src.neural.neural_agent import SimpleOpponentPredictorGame
from src.neural.regret_neural_agent import RegretOpponentPredictorGame
from src.regret_learning.zero_sum_game import ZeroSumGame
from src.regret_learning.pure_strategy import PureStrategyGame
from src.regret_learning.mixed_strategy import MixedStrategyGame
from src.regret_learning.hedge import HedgeGame
from src.regret_learning.swap_regret import SwapRegretGame
from src.regret_learning.swap_learner import SwapLearnerGame
from src.regret_learning.compare_learners import CompareLearners
from src.neural.strategic_swap_regret_transformer import experiment_c
from src.neural.learning_aware_opponent import play_experiment
from src.neural.rps_learning_update_ablation import rps_learning_update_ablation
from src.neural.neural_swap_validation import neural_swap_validation

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
    swap_regret_game.play_history(verbose=False)
    
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
    for algorithm, stats in comparison_result.items():
        print(f"{algorithm}:")
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

    print("\nTransformer Neural Network Opponent Predictor Game with more rounds:")
    transformer_game = RegretOpponentPredictorGame(type="Transformer")
    transformer_game.play(examples, rounds=500)

    print("\nSwap Regret transformer experiment a:")
    experiment_swap_a()

    print("\nSwap Regret transformer experiment b:")
    experiment_swap_b()

def check_exploitability():
    print("\nNeural with Learning Agent")
    experiment_c()

    print("\nLearning aware opponent with neural agent")
    play_experiment()

    print("\nOnline learning aware opponent with neural agent")
    play_experiment_online_learning_aware()

    print("\nOnline transformer planner with neural agent")
    experiment_online_transformer_planner()

    print("\nStrategic online transformer planner with neural agent")
    experiment_strategic_online_transformer_planner()

    print("\nHorizon strategic online transformer planner with neural agent")
    experiment_horizon_strategic_online_transformer_planner()

    print("\nOnline transformer RPS model mismatch with neural agent")
    online_transformer_rps_model_mismatch()

    print("\nOnline transformer RPS stationary benchmark with neural agent")
    online_transformer_rps_stationary_benchmark()

    print("\nRPS learning update ablation with neural agent")
    rps_learning_update_ablation()

def blackbox():
    print("\nBlack-box opponent experiment with neural agent")
    blackbox_experiment()

    print("\nRPS black-box dynamics with neural agent")
    rps_blackbox_dynamics()

    print("\nRPS black-box expected rollouts with neural agent")
    rps_blackbox_expected_rollouts()

def check_neural_swap_validation():
    print("\nNeural swap validation")
    neural_swap_validation()

def main():
    # checking_regret_learning()
    # check_neural()
    # check_exploitability()
    # blackbox()
    check_neural_swap_validation()

if __name__ == "__main__":
    main()
