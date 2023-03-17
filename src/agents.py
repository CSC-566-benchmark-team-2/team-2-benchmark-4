import random
from rps_game import Move
from abc import ABC


class Agent(ABC):
    def move(self, last_result: int) -> Move:
        """
        Args:
            `last_result`: who won the last round - `1` for player 1 or `2` for player 2
        Returns:
            Move against opponent
        """
        pass


class Agent1(Agent):
    def __init__(self, seed) -> None:
        self.seed = seed
        self.rand = random.Random(seed)

    def move(self, last_result: int):
        return Move(self.rand.randint(0, 3))


class Agent2(Agent):
    def __init__(self, move=1) -> None:
        self.move = move
        pass

    def move(self, last_result: int):
        return Move(self.move)


class ProbabilisticAgent(Agent):
    """
    Agent1
    Chooses the move with the highest probability of victory based on a window of games
    """

    def __init__(self) -> None:
        self.move_stats = {
            Move.Rock: {"count": 0, "wins": 0},
            Move.Paper: {"count": 0, "wins": 0},
            Move.Scissors: {"count": 0, "wins": 0},
        }
        self.last_move: Move = None

    def move(self, last_result: int):
        if last_result == 2:
            self.move_stats[self.last_move]["wins"] += 1

        best_move = max(self.move_stats.keys(), key=lambda k: self.move_stats[k])
        move_stats = self.move_stats[best_move]
        win_rate = move_stats["wins"] / move_stats["count"]
        if win_rate < 0.5:
            best_move = random.choice(Move.domain())
        self.move_stats[best_move]["count"] += 1
        return best_move
