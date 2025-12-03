from itertools import combinations, permutations, product
import random
import numpy as np
from functools import lru_cache


MAX_TURNS = 12
SIZE = 5
COLOURS = [
    "red", "blue", "green", "yellow", "orange", "purple"
]


def generate_colour_permutations(colours, n):
    """Generate all unique pairs of colour combinations."""
    return tuple(list(product(colours, repeat=n)))


all_permutations = generate_colour_permutations(COLOURS, SIZE)


@lru_cache(maxsize=None)
def get_score(choice, code):
    correct_position = sum(c1 == c2 for c1, c2 in zip(choice, code))
    correct_colour = sum(min(choice.count(c), code.count(c)) for c in set(choice)) - correct_position
    return correct_position, correct_colour


@lru_cache(maxsize=None)
def information_gain(remaining_possible_solutions, set_of_choices):
    """Calculate the information gain for each possible choice."""
    if len(remaining_possible_solutions) == 1:
        return {remaining_possible_solutions[0]: 1}
    info_gain = {}
    S = len(remaining_possible_solutions)
    for choice in set_of_choices:
        score_counts = {}
        for possible_solution in remaining_possible_solutions:
            score = get_score(choice, possible_solution)
            if score not in score_counts:
                score_counts[score] = 0
            score_counts[score] += 1
        entropy = 0
        for count in score_counts.values():
            probability = count / S
            entropy -= (probability * np.log2(probability))
        info_gain[choice] = entropy
    return info_gain


def make_choice(strategy, remaining_possible_solutions):
    if strategy == "random":
        return random.choice(all_permutations)
    elif strategy == "random_possible":
        return random.choice(remaining_possible_solutions)
    elif strategy == "info_gain":
        info_gain = information_gain(tuple(remaining_possible_solutions), tuple(remaining_possible_solutions))
        return max(info_gain, key=info_gain.get)
    elif strategy == "info_gain_all":
        info_gain = information_gain(tuple(remaining_possible_solutions), all_permutations)
        return max(info_gain, key=info_gain.get)
    else:
        raise ValueError("Unknown strategy")


def play_game(code, strategy, verbose=True):
    remaining_possible_solutions = list(all_permutations).copy()
    turn = 0
    remaining_possible_solutions_history = []
    while True:
        turn += 1
        remaining_possible_solutions_history.append(len(remaining_possible_solutions))
        if verbose:
            print(f"Remaining possible solutions: {len(remaining_possible_solutions)}")
        choice = make_choice(strategy, remaining_possible_solutions)
        score = get_score(choice, code)
        if verbose:
            print(f"Turn {turn}: Choice: {choice}, Score: {score}")
        if score[0] == SIZE:
            if verbose:
                print(f"Solved the code {code} in {turn} turns!")
            return True, turn, remaining_possible_solutions_history
        elif turn == MAX_TURNS:
            if verbose:
                print(f"Failed to solve the code {code} in {MAX_TURNS} turns.")
            return False, None, remaining_possible_solutions_history

        remaining_possible_solutions = [
            sol for sol in remaining_possible_solutions
            if sol != choice and get_score(choice, sol) == score
        ]


if __name__ == "__main__":
    verbose = False
    results = []
    for i in range(1000):
        print(f"=== Game {i+1} ===")
        code = random.choice(all_permutations)
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
    df_results["games_played"] = 1
    df_results_agg = df_results.groupby("strategy").agg(
        games_played=("games_played", "sum"),
        solved_games=("turns", lambda x: x.notna().sum()),
        avg_turns=("turns", "mean"),
        std_turns=("turns", "std"),
    )