import sys
import os
from typing import Callable

from src.rps_game import Move, RockPaperScissorsGame

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from benchmark_utils import (
    preprocess_flare_df,
    preprocess_life_df,
    preprocess_video_games_df,
    preprocess_pulsar_df,
    preprocess_heart_df,
    gaussian_quantiles,
    moons,
    breast_cancer,
    challenge1,
    challenge2,
    extra_challenge,
)
# from src.main import Solution
from src.agents import Agent
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import timeit


def run_benchmarks(create_agent: Callable[[str], Agent]) -> dict:
    SEED = 42

    np.random.seed(SEED)
    results = {}
    game = RockPaperScissorsGame()
    agent_ids = ["agent" + i for i in range(1, 4)]
    for agent_id in agent_ids:
        agent = create_agent(agent_id)
        test_df = pd.read_csv(agent_id + "_df_test.csv")
        wins = 0
        draws = 0
        count = 0
        last_result = None
        for _, row in test_df.iterrows():
            opponent_move = Move.from_str(row["agent_choice"])
            player_move = agent.move(last_result)
            result = game.get_winner(player_move, opponent_move)
            wins += result == 1
            draws += results == 0

        results[agent_id] = {
            "wld_score": (3 * wins + draws)
            / (count * 3)  # 3 points for win, 1 point for draw, 0 for loss
        }
    return results
