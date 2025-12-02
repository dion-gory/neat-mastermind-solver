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


def score_choice_to_code(choice, code):
    """Score a choice against the code.

    Returns a tuple (correct_position, correct_colour).
    """
    correct_position = sum(c1 == c2 for c1, c2 in zip(choice, code))
    correct_colour = sum(min(choice.count(c), code.count(c)) for c in set(choice)) - correct_position
    return correct_position, correct_colour


def information_gain(possible_choices):
    """Calculate the information gain for each possible choice."""
    info_gain = {}
    for choice in possible_choices:
        score_distribution = {}
        for code in possible_choices:
            score = score_choice_to_code(choice, code)
            if score not in score_distribution:
                score_distribution[score] = 0
            score_distribution[score] += 1
        total_possibilities = len(possible_choices)
        entropy = 0
        for count in score_distribution.values():
            probability = count / total_possibilities
            entropy -= (probability * np.log2(probability))
        info_gain[choice] = entropy
    return info_gain


if __name__ == "__main__":
    colour_permutations = generate_colour_permutations(COLOURS, SIZE)
    code = random.choice(colour_permutations)

    possible_choices = colour_permutations.copy()
    turn = 0
    while True:
        turn += 1
        info_gain = information_gain(possible_choices)
        choice = max(info_gain, key=info_gain.get)
        choice_entropy = info_gain[choice]
        score = score_choice_to_code(choice, code)
        print(f"Turn {turn}: Choice: {choice}, Entropy: {choice_entropy:.4f}, Score: {score}")
        if score[0] == SIZE:
            print(f"Solved the code {code} in {turn} turns!")
            break
        for possible_choice in possible_choices:
            if score_choice_to_code(possible_choice, choice) != score:
                possible_choices.remove(possible_choice)
        if turn == MAX_TURNS:
            print(f"Failed to solve the code {code} in {MAX_TURNS} turns.")
            break