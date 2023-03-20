# Team-2 Benchmark-4 Rock Paper Scissors

## Repository Structure

- main: The gym environment used for building an agent and playing rock paper scissors games.
- rps_game: Infrastructure to play rock, paper, scissors. Stores game history of draws, player 1 wins, and player 2 wins.

## Idea Behind Benchmark 4

The original idea behind benchmark 4 was to have teams create agents to play Atari games using OpenAI's Gym framework. While this is more of an AI task, we reframe the problem by creating training data and ask teams to create neural nets to predict moves and play a variety of agents we created. So, for this benchmark we have created our own version of an AI gym, created training data from Agents who take actions in a certain manner, and have given you a framework to create your own and play the game.

## Training Data

- 1000 games
- Uniform Random Agent (player 1) vs Computer Agent (player 2)
  - Computer agent is one of 3 unspecified agents
- Features: choice p1, choice p2, round, winner
- Target: Winning choice

## Results Metrics

- Win percentage of your agent (including draws)

## Submission Process

1. Fill out the code to create your agent
2. Same as usual, push to submit and check the [leaderboard](https://csc-566-benchmark-results.netlify.app/)!

## Setup a virtual env

From the command line in the root directory run:

1. `python3.9 -m venv .venv`
2. `source .venv/bin/activate`
3. ` pip install -r requirements.txt`
