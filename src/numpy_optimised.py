from itertools import product

import numpy as np

MAX_TURNS = 12
SIZE = 5
NUM_COLOURS = 6
NUM_PERMUTATIONS = NUM_COLOURS ** SIZE


def generate_colour_permutations():
    """Generate all unique pairs of colour combinations."""
    return np.array(list(product(np.arange(NUM_COLOURS), repeat=SIZE)))


all_permutations = generate_colour_permutations()


def get_scores_fully_vectorized(choices, codes):
    """
    Fully vectorized implementation with no Python loops.

    Parameters:
    - choices: np.array of shape (M, SIZE)
    - codes: np.array of shape (N, SIZE)

    Returns:
    - scores: np.array of shape (M, N, 2) where scores[i,j] = [correct_positions, correct_colours]
    """
    M, SIZE = choices.shape
    N = codes.shape[0]

    # Correct positions
    correct_positions = np.sum(
        choices[:, np.newaxis, :] == codes[np.newaxis, :, :],
        axis=2
    )

    # Create one-hot encoding for color counts
    choices_onehot = np.zeros((M, SIZE, NUM_COLOURS), dtype=int)
    codes_onehot = np.zeros((N, SIZE, NUM_COLOURS), dtype=int)

    for i in range(SIZE):
        choices_onehot[np.arange(M), i, choices[:, i]] = 1
        codes_onehot[np.arange(N), i, codes[:, i]] = 1

    # Sum to get counts
    choices_counts = choices_onehot.sum(axis=1)  # (M, NUM_COLOURS)
    codes_counts = codes_onehot.sum(axis=1)  # (N, NUM_COLOURS)

    # Compute pairwise minimums and sum
    total_matches = np.sum(
        np.minimum(
            choices_counts[:, np.newaxis, :],
            codes_counts[np.newaxis, :, :]
        ),
        axis=2
    )

    correct_colours = total_matches - correct_positions

    # Stack into single array with shape (M, N, 2)
    scores = np.stack([correct_positions, correct_colours], axis=2)

    return scores


def information_gain_fully_vectorized(remaining_possible_solutions, set_of_choices):
    """
    Fully vectorized information gain calculation with minimal Python loops.

    Parameters:
    - remaining_possible_solutions: np.array of shape (S, SIZE)
    - set_of_choices: np.array of shape (C, SIZE)

    Returns:
    - info_gain: np.array of shape (C,) with entropy for each choice
    """
    S = len(remaining_possible_solutions)
    C = len(set_of_choices)

    # Get all pairwise scores
    all_scores = get_scores_fully_vectorized(set_of_choices, remaining_possible_solutions)

    # Convert to unique score IDs
    score_ids = all_scores[:, :, 0] * (SIZE + 1) + all_scores[:, :, 1]

    # Vectorized entropy calculation
    entropies = np.zeros(C)

    # Process all choices at once using apply_along_axis
    def calc_entropy(scores):
        _, counts = np.unique(scores, return_counts=True)
        probs = counts / S
        return -np.sum(probs * np.log2(probs))

    entropies = np.apply_along_axis(calc_entropy, 1, score_ids)

    return entropies


def make_choice(strategy, remaining_possible_solutions):
    """
    Make a choice based on the given strategy.

    Parameters:
    - strategy: string specifying the strategy
    - remaining_possible_solutions: np.array of shape (N, SIZE)

    Returns:
    - choice: np.array of shape (1, SIZE)
    """
    # Special case: if only one solution remains, just return it
    if len(remaining_possible_solutions) == 1:
        return remaining_possible_solutions[0:1]

    if strategy == "random":
        idx = np.random.randint(len(all_permutations))
        return all_permutations[idx:idx + 1]

    elif strategy == "random_possible":
        idx = np.random.randint(len(remaining_possible_solutions))
        return remaining_possible_solutions[idx:idx + 1]

    elif strategy == "info_gain" or (strategy == "info_gain_hybrid" and len(remaining_possible_solutions) <= 100):
        # Calculate entropy for all remaining possible solutions
        entropies = information_gain_fully_vectorized(
            remaining_possible_solutions,
            remaining_possible_solutions
        )
        # Find the choice with maximum entropy
        best_idx = np.argmax(entropies)
        return remaining_possible_solutions[best_idx:best_idx + 1]

    elif strategy == "info_gain_all" or (strategy == "info_gain_hybrid" and len(remaining_possible_solutions) > 100):
        # Calculate entropy for all possible codes
        entropies = information_gain_fully_vectorized(
            remaining_possible_solutions,
            all_permutations
        )
        # Find the choice with maximum entropy
        best_idx = np.argmax(entropies)
        return all_permutations[best_idx:best_idx + 1]

    else:
        raise ValueError("Unknown strategy")


def play_game(code, strategy):
    """
    Play a single game of Mastermind.

    Parameters:
    - code: np.array of shape (1, SIZE) or (SIZE,)
    - strategy: string specifying the strategy

    Returns:
    - num_turns: int, number of turns taken (MAX_TURNS + 1 if not solved)
    """
    if code.ndim == 1:
        code = code.reshape(1, -1)

    remaining_possible_solutions = all_permutations.copy()

    for turn in range(1, MAX_TURNS + 1):
        # Special case: only one solution left
        if len(remaining_possible_solutions) == 1:
            return turn

        # Make choice
        choice = make_choice(strategy, remaining_possible_solutions)

        # Get score
        score = get_scores_fully_vectorized(choice, code)[0, 0]  # shape: (2,)

        # Check if solved
        if score[0] == SIZE:
            return turn

        # Prune remaining solutions
        all_scores = get_scores_fully_vectorized(
            choice,
            remaining_possible_solutions
        )[0]  # shape: (N, 2)

        # Compare each score pair
        matching = (all_scores[:, 0] == score[0]) & (all_scores[:, 1] == score[1])
        remaining_possible_solutions = remaining_possible_solutions[matching]

    # Failed to solve
    return MAX_TURNS + 1


def play_game_wrapper(args):
    code, strategy = args
    return strategy, play_game(code, strategy)