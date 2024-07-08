"""
This module implements a Rock, Paper, Scissors game where the player plays against the computer.
The computer's choices are guided by a predictive model based on the player's past choices,
utilizing a Markov Chain to increase its chances of winning.

The source links for Markov Chain:
Markov chain(theory):
    https://en.wikipedia.org/wiki/Markov_chain
    https://towardsdatascience.com/brief-introduction-to-markov-chains-2c8cab9c98ab
Markov chain(code):
    https://towardsdatascience.com/how-to-win-over-70-matches-in-rock-paper-scissors-3e17e67e0dab
"""

from __future__ import division
import json
import os
import random
from typing import Dict, Optional

# BEAT maps each choice to the choice it defeats in the game
BEAT = {'R': 'P', 'P': 'S', 'S': 'R'}

# STATE_CODES are used to encode the state of the game based on the outcome and the move.
STATE_CODES = {'VA': 'P', 'VT': 'R', 'VC': 'S', 'LA': 'P', 'LT': 'R', 'LC': 'S'}

# VALID_CHOICES defines the possible moves a player can make in the game.
VALID_CHOICES = ['R', 'P', 'S']

# MAX_ROUNDS sets the maximum number of rounds to be played in a game.
MAX_ROUNDS = 30

# WIN_SCORE defines the score a player needs to reach to win the game.
WIN_SCORE = 10

# FILE_NAME specifies the name of the file where the Markov Chain matrix is saved.
FILE_NAME = 'matrix.json'


class MarkovChain:
    """
        Class representing a Markov Chain model for predicting player's next move in a game.

        Attributes:
            filename (str): The name of the file where the Markov Chain matrix is stored.
            matrix (dict): The current state of the Markov Chain matrix used for predictions.
            increase (float): The amount to increase the probability for the observed transition.
            decrease (float): The amount to decrease the probabilities for all other transitions
                from the current state.

        Methods:
            load_or_create_matrix(): Loads the Markov Chain matrix from a file,
                or creates a new one if the file doesn't exist.

            update_matrix(prev, next): Updates the matrix based on the previous
                and next states observed.

            predict_move(prev_state): Predicts the next move based on the previous state
                or overall statistics if no previous state.

            save_matrix(): Saves the current state of the Markov Chain matrix to a file.
    """

    def __init__(self, filename: str = FILE_NAME, increase: float = 0.05,
                 decrease: float = 0.01) -> None:
        self.filename = filename
        self.increase = increase
        self.decrease = decrease
        self.matrix: Dict[str, Dict[str, float]] = self.load_or_create_matrix()

    def load_or_create_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Loads the Markov Chain matrix from a file if it exists, otherwise initializes
        a new matrix.

        This method attempts to load the matrix from a JSON file specified by the
        filename attribute. If the file does not exist or there is an error reading
        the file, a new matrix is created with initial probabilities. Each state in
        the new matrix is initialized with equal probabilities for transitioning to
        any next state, based on the predefined STATE_CODES.

        Returns:
            Dict[str, Dict[str, float]]: A dictionary representing the Markov Chain matrix,
            where keys are current states and values are dictionaries mapping next states to
            their transition probabilities.
        """
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r', encoding='utf-8') as file:
                    return json.load(file)
            except json.JSONDecodeError:
                print("Error reading the matrix file. A new matrix will be created.")
        return {
            state: {
                next_state: 1 / len(STATE_CODES)
                for next_state in STATE_CODES
            }
            for state in STATE_CODES
        }

    def update_matrix(self, prev_element: str, next_element: str) -> None:
        """
        Updates the Markov Chain matrix based on the transition from the previous
        element (state) to the next.

        This method increments the probability for the observed transition(prev_element
        to next_element) by a predefined increase amount. For all other transitions from
        the prev_element, it decrements thei probabilities by a predefined decrease amount,
        ensuring the total probability from prev_element remains normalized.

        Args:
            prev_element (str): The previous state in the sequence of player moves.
            next_element (str): The next state in the sequence, following the prev_element.

        Returns:
            None
        """
        increase_probability = self.matrix[prev_element][next_element] + self.increase
        self.matrix[prev_element][next_element] = min(1.0, increase_probability)
        for state in STATE_CODES:
            if state != next_element:
                decrease_probability = self.matrix[prev_element][state] - self.decrease
                self.matrix[prev_element][state] = max(0.0, decrease_probability)

    def predict_move(self, prev_state: Optional[str] = None) -> str:
        """
        Predicts the computer's next move in a Rock game based on the Markov Chain model.

        If a previous state is provided and exists in the matrix, the method uses the
        probabilities from the matrix to predict the next move. If no previous state
        is provided, or it does not exist in the matrix, the method calculates the
        next move based on average probabilities across all states.

        Args:
            prev_state (Optional[str]): The previous state of the game, used to look up
            the next predicted move in the matrix.

        Returns:
            str: The predicted next move by the computer ('R', 'P', or 'S').
        """
        if prev_state is not None and prev_state in self.matrix:
            probs: Dict[str, float] = {move: 0.0 for move in VALID_CHOICES}
            for state, prob in self.matrix[prev_state].items():
                probs[STATE_CODES[state]] += prob
            return BEAT[max(probs, key=lambda k: probs[k])]

        avg_probs: Dict[str, float] = {move: 0.0 for move in VALID_CHOICES}
        total_states = len(self.matrix)
        for state_probs in self.matrix.values():
            for sub_state, prob in state_probs.items():
                move = STATE_CODES[sub_state]
                avg_probs[move] += prob / (total_states * len(state_probs))
        return BEAT[max(avg_probs, key=lambda k: avg_probs[k])]

    def save_matrix(self) -> None:
        """
        Saves the current state of the Markov Chain matrix to a file specified
        by the filename attribute.

        This method writes the Markov Chain matrix, which is maintained as a dictionary
        in memory, to a JSON file. The matrix contains the probabilities of transitioning
        from one state to another based on the player's move history.
        If an IOError occurs during file writing, an error message is printed indicating that
        the matrix could not be saved.

            Returns:
                None
        """
        try:
            with open(self.filename, 'w', encoding='utf-8') as file:
                json.dump(self.matrix, file)
        except IOError:
            print("Error saving the matrix file.")


def determine_winner(player: str, computer: str) -> int:
    """
    Determines the winner of a round in the Rock, Paper, Scissors game.

    Compares the player's choice against the computer's choice to decide the round's outcome.
    The game logic is based on the classic rules: Rock beats Scissors ('R' beats 'S'),
    Scissors beats Paper ('S' beats 'P'), and Paper beats Rock ('P' beats 'R').

    Args:
        player (str): The player's choice for the round, represented as 'R' for Rock,
                      'P' for Paper, or 'S' for Scissors.
        computer (str): The computer's choice for the round, also represented as 'R', 'P', or 'S'.

    Returns:
        int: The result of the round. Returns 0 for a tie, 1 if the player wins,
             or -1 if the computer wins.
    """
    if player == computer:
        return 0
    return 1 if BEAT[player] == computer else -1


def get_state(choice: str, winner: int) -> str:
    """
    Derives a state code from the player's move and the outcome of the round.

    This function creates a state code that encapsulates the player's choice ('R', 'P', or 'S')
    and the result of the round (win or lose). The state code is prefixed with 'V' for a victory
    and 'L' for a loss, followed by the corresponding code for the player's move as defined in
    STATE_CODES.

    Args:
        choice (str): The player's move in the round.
        winner (int): The result of the round, where 1 indicates the player won, and -1 indicates
            a loss.

    Returns:
        str: A state code representing the outcome and the player's move.
    """
    prefix = 'V' if winner == 1 else 'L'
    return prefix + next((k[1] for k, v in STATE_CODES.items() if v == choice), '')


def display_round_result(winner: int, player_choice: str, computer_choice: str, player_score: int,
                         computer_score: int) -> None:
    """
    Prints the outcome of a round, including the choices and current scores.

    This function displays the round's result with descriptive messages indicating whether
    the player or the computer won, or if there was a tie. It also shows the choices made
    by both the player and the computer, along with their scores up to the current round.

    Args:
        winner (int): The result of the round; 0 for a tie, 1 for a player win, -1 for
            a computer win.
        player_choice (str): The player's move in the round.
        computer_choice (str): The computer's predicted move in the round.
        player_score (int): The player's total score up to the current round.
        computer_score (int): The computer's total score up to the current round.

    Returns:
        None
    """
    result_messages = ["Computer wins the round!", "It's a tie!", "Player wins the round!"]
    print(
        f"Player's choice: {player_choice}, Computer's choice: {computer_choice}")
    print(result_messages[winner + 1])
    print(f"Player Score: {player_score}, Computer Score: {computer_score}")
    print("-" * 80)


def play_game() -> None:
    """
    Initiates and manages a game of Rock, Paper, Scissors against the computer.

    This function orchestrates the game flow, including making predictions for the
    computer's moves using a Markov Chain model, capturing the player's choices,
    determining round winners, and updating scores. The game continues until a maximum
    number of rounds or a win score is reached.

    Returns:
        None
    """
    player_score, computer_score, rounds = 0, 0, 0
    prev_state = None
    markov = MarkovChain()

    while rounds < MAX_ROUNDS and max(player_score, computer_score) < WIN_SCORE:
        if prev_state:
            computer_choice = markov.predict_move(prev_state)
        else:
            computer_choice = random.choice(VALID_CHOICES)
        player_choice = input("Enter your choice (R/P/S): ").upper()
        if player_choice not in VALID_CHOICES:
            print("Invalid choice. Choose again.")
            continue

        winner = determine_winner(player_choice, computer_choice)
        player_score += winner == 1
        computer_score += winner == -1

        display_round_result(winner, player_choice, computer_choice, player_score, computer_score)

        state = get_state(player_choice if winner == -1 else computer_choice, winner)
        if prev_state and state:
            markov.update_matrix(prev_state, state)
        prev_state = state
        rounds += 1

    print(
        [
            "The game ends in a tie.",
            "Congratulations! You win the game!",
            "The computer wins the game."
        ]
        [sign(player_score - computer_score)]
    )

    markov.save_matrix()


def sign(x: int) -> int:
    """
    Determines the mathematical sign of a given integer.

    This function calculates the sign of an integer 'x' and returns:
    - 1 if 'x' is positive,
    - -1 if 'x' is negative,
    - 0 if 'x' is zero.

    The sign function is useful in various mathematical and game logic contexts where
    the distinction between positive, negative, and zero values impacts decision-making.

    Args:
        x (int): The integer whose sign is to be determined.

    Returns:
        int: The sign of 'x', indicated as 1, -1, or 0.
    """
    return (x > 0) - (x < 0)


if __name__ == '__main__':
    play_game()
