from itertools import combinations, permutations, product
import random
import numpy as np


MAX_TURNS = 12
SIZE = 4
COLOURS = [
    "red", "blue", "green", "yellow", "orange", "purple"
]


def generate_colour_permutations(colours, n):
    """Generate all unique pairs of colour combinations."""
    return list(product(colours, repeat=n))


colour_permutations = generate_colour_permutations(COLOURS, SIZE)


def get_score(choice, code):
    """Score a choice against the code.

    Returns a tuple (correct_position, correct_colour).
    """
    correct_position = sum(c1 == c2 for c1, c2 in zip(choice, code))
    correct_colour = sum(min(choice.count(c), code.count(c)) for c in set(choice)) - correct_position
    return correct_position, correct_colour


def information_gain(possible_choices, pool_of_choices):
    """Calculate the information gain for each possible choice."""
    info_gain = {}
    total_possibilities = len(possible_choices)
    for choice in pool_of_choices:
        score_distribution = {}
        for code in possible_choices:
            score = get_score(choice, code)
            if score not in score_distribution:
                score_distribution[score] = 0
            score_distribution[score] += 1
        entropy = 0
        for count in score_distribution.values():
            probability = count / total_possibilities
            entropy -= (probability * np.log2(probability))
        info_gain[choice] = entropy
    return info_gain


def make_choice(strategy, possible_choices):
    if strategy == "random":
        return random.choice(colour_permutations)
    elif strategy == "random_possible":
        return random.choice(possible_choices)
    elif strategy == "info_gain":
        info_gain = information_gain(possible_choices, possible_choices)
        return max(info_gain, key=info_gain.get)
    elif strategy == "info_gain_all":
        info_gain = information_gain(possible_choices, colour_permutations)
        return max(info_gain, key=info_gain.get)
    else:
        raise ValueError("Unknown strategy")


def play_game(code, strategy, verbose=True):
    possible_choices = colour_permutations.copy()
    turn = 0
    possible_choices_history = []
    while True:
        turn += 1
        possible_choices_history.append(len(possible_choices))
        if verbose:
            print(f"Possible choices remaining: {len(possible_choices)}")
        choice = make_choice(strategy, possible_choices)
        score = get_score(choice, code)
        if verbose:
            print(f"Turn {turn}: Choice: {choice}, Score: {score}")
        if score[0] == SIZE:
            if verbose:
                print(f"Solved the code {code} in {turn} turns!")
            return True, turn, possible_choices_history
        elif turn == MAX_TURNS:
            if verbose:
                print(f"Failed to solve the code {code} in {MAX_TURNS} turns.")
            return False, None, possible_choices_history

        if choice in possible_choices:
            possible_choices.remove(choice)
        for possible_choice in colour_permutations:
            if possible_choice not in possible_choices:
                continue
            if get_score(possible_choice, choice) != score:
                possible_choices.remove(possible_choice)


if __name__ == "__main__":
    verbose = False
    results = []
    for i in range(100):
        print(f"=== Game {i+1} ===")
        code = random.choice(colour_permutations)
        result1 = play_game(code, strategy="info_gain", verbose=verbose)
        result2 = play_game(code, strategy="random_possible", verbose=verbose)
        # result3 = play_game(code, strategy="random", verbose=verbose)
        result4 = play_game(code, strategy="info_gain_all", verbose=verbose)
        # information_gain = [float(-np.log2(result1[2][i+1]/result1[2][i])) for i in range(len(result1)-1)]
        # avg_information_gain = sum(information_gain) / len(information_gain)
        results.append({
            "info_gain": result1[1],
            "random_possible": result2[1],
            # "random": result3[1],
            "info_gain_all": result4[1],
        })

    import pandas as pd
    df_results = pd.DataFrame(results).replace({None: np.nan})
    df_results = df_results.melt(var_name="strategy", value_name="turns")
    df_results_agg = df_results.groupby("strategy").agg(
        games_played=("turns", "count"),
        solved_games=("turns", lambda x: x.notna().sum()),
        avg_turns=("turns", "mean"),
        std_turns=("turns", "std"),
    )