import enum


class Move(enum.Enum):
    Rock = 0
    Paper = 1
    Scissors = 2

    def wins_against(self, other):
        return (
            self == Move.Rock
            and other == Move.Scissors
            or self == Move.Paper
            and other == Move.Rock
            or self == Move.Scissors
            and other == Move.Paper
        )

    def draws(self, other):
        return self == other

    def loses_to(self, other):
        return not self.wins_against(other) and not self.draws(other)

    @staticmethod
    def domain() -> list["Move"]:
        return [Move.Rock, Move.Paper, Move.Scissors]


class RockPaperScissorsGame:
    victories = {
        Move.Rock: Move.Scissors,  # Rock beats scissors
        Move.Paper: Move.Rock,  # Paper beats rock
        Move.Scissors: Move.Paper,  # Scissors beats paper
    }

    def __init__(self) -> None:
        self.game_count = 0
        self.result_history = {
            "draw": 0,
            "player_1": 0,
            "player_2": 0,
        }  # 0th element = draw count, 1st element = player 1 win count, 2nd element = player 2 win count
        pass

    def get_winner(self, choice_player1: Move, choice_player2: Move) -> int:
        # returns Player who won as integer, 0 if draw

        # in case the user passed in an integer instead of a Move
        if isinstance(choice_player1, Move) is False:
            choice_player1 = Move(choice_player1)
        if isinstance(choice_player2, Move) is False:
            choice_player2 = Move(choice_player2)

        self.game_count += 1
        if choice_player1.draws(choice_player2):
            self.result_history["draw"] += 1
            return 0
        elif choice_player1.wins_against(choice_player2):
            self.result_history["player_1"] += 1
            return 1
        else:
            self.result_history["player_2"] += 1
            return 2

    def get_result_percentages(self):
        return {
            result: count / self.game_count
            for result, count in self.result_history.items()
        }
