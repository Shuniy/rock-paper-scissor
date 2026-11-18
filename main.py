from src.regret_learning.zero_sum_game import ZeroSumGame
from src.regret_learning.pure_strategy import PureStrategyGame
from src.regret_learning.mixed_strategy import MixedStrategyGame
from src.regret_learning.hedge import HedgeGame
from src.regret_learning.swap_regret import SwapRegretGame
from src.constants import ACTION

def main():
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
    swap_regret_game.play(rounds=100, verbose=False)
    

if __name__ == "__main__":
    main()