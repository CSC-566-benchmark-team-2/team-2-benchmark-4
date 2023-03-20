import os
import sys
import random
from typing import Dict

from rps_game import RockPaperScissorsGame, Move
from agents import *

# from send_results import

num_games = 2000  # number of rps games to play


def create_agent(requested_agent: str) -> Agent:
    """
    Initialization function that creates instances of the agent's you trained on the provided datasets.

    Args:
        `requested_agent`: A string identifier of the opponent that your agent will run against: 'agent1' | 'agent2' | 'agent3'
    Returns:
        Your instantiated agent initialized for benchmarking

    """
    # Your code here.

    if requested_agent == "agent_1":
        return None
    elif requested_agent == "agent_2":
        return None
    elif requested_agent == "agent_3":
        return None


def set_computer_agent(agent: int, seed: int = 42):
    if agent == 1:
        return Agent1(seed=seed)
    if agent == 2:
        return Agent2()
    if agent == 3:
        return CyclicalAgent(seed=seed)


# sample agents
def get_computer_agent_input(agent, last_result: int):
    return agent.move(last_result)


def get_your_agent_input(last_result: int):
    # last_result:
    #   None means no previous game was played
    #   0 means the last game was a draw,
    #   1 means that your agent won the last game
    #   2 means that your agent lost the last game

    choice = None  # options: [0-2], Rock = 0, Paper = 1, Scissors = 2
    move = Move(choice)
    # YOUR CODE HERE

    return move


def get_random_agent_input(last_result: int):

    return Move(random.choice([0, 1, 2]))
    # return random.choice([0, 1, 2])


if __name__ == "__main__":

    rps = RockPaperScissorsGame()
    agent_selected = 3  # which agent to test against, options: [1-3]
    agent = set_computer_agent(agent_selected, seed=42)
    random_agent = Agent1(seed=75)
    last_result = None
    generating_data = True
    test_functionality = False

    import pandas as pd

    output_df = pd.DataFrame(columns=["agents_choice", "result"])

    for i in range(num_games):

        if test_functionality:
            player_one_choice = get_random_agent_input(last_result)
            player_two_choice = get_random_agent_input(last_result)
            last_result = rps.get_winner(player_one_choice, player_two_choice)
            continue

        if generating_data:  # TODO: get rid of this after generating data
            # player1: our agent, player2: random agent
            player_two_choice = get_computer_agent_input(random_agent, last_result)
            player_one_choice = get_computer_agent_input(agent, last_result)
            last_result = rps.get_winner(player_one_choice, player_two_choice)

            str_result = "D"
            if last_result == 1:
                str_result = "L"
            elif last_result == 2:
                str_result = "W"

            player_one_choice = str(player_one_choice).split(".")[
                1
            ]  # take off prefix from "Move.[Rock|Paper|Scissors]"

            output_df = pd.concat(
                [
                    output_df,
                    pd.DataFrame(
                        [{"agents_choice": player_one_choice, "result": str_result}]
                    ),
                ],
                ignore_index=True,
                axis=0,
            )
            continue
        else:
            # player1: your agent, player2: computer agent
            player_two_choice = get_computer_agent_input(agent, last_result)
            player_one_choice = get_your_agent_input(last_result)
        last_result = rps.get_winner(player_one_choice, player_two_choice)

    print(rps.get_result_percentages())  # DEBUG

    output_df.to_csv(f"agent{agent_selected}_df.csv")

    # results = rps.get_result_percentages()
    # results_metrics = {
    #     "agent": agent_selected,
    #     "win": results["player_1"],
    #     "loss": results["player_2"],
    #     "draw": results["draw"],
    # }

    sys.exit()
