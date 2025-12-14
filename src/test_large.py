"""Test script for large Mastermind problems."""
import numpy as np
import numpy_optimised
from numpy_optimised import play_game, generate_random_codes, configure_problem

# Test with different problem sizes
test_configs = [
    {"SIZE": 5, "NUM_COLOURS": 6, "name": "Classic (5x6)"},
    {"SIZE": 6, "NUM_COLOURS": 8, "name": "Medium (6x8)"},
    {"SIZE": 8, "NUM_COLOURS": 10, "name": "Large (8x10)"},
    {"SIZE": 10, "NUM_COLOURS": 10, "name": "Very Large (10x10)"},
    {"SIZE": 12, "NUM_COLOURS": 12, "name": "Extreme (12x12)"},
]

def test_config(size, num_colours, name):
    """Test a specific configuration."""
    # Configure the problem
    configure_problem(size, num_colours)

    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Search space size: {num_colours}^{size} = {numpy_optimised.NUM_PERMUTATIONS:,}")
    print(f"{'='*60}")

    # Generate a random test code
    test_code = generate_random_codes(1)
    print(f"Test code: {test_code[0]}")

    # Test each strategy
    strategies = ["random_possible", "info_gain", "info_gain_all"]

    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        try:
            turns = play_game(test_code, strategy)
            if turns <= numpy_optimised.MAX_TURNS:
                print(f"  [OK] Solved in {turns} turns")
            else:
                print(f"  [FAIL] Failed to solve (>{numpy_optimised.MAX_TURNS} turns)")
        except Exception as e:
            print(f"  [ERROR] {e}")

if __name__ == "__main__":
    for config in test_configs:
        test_config(config["SIZE"], config["NUM_COLOURS"], config["name"])

    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}")
