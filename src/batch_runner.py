from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm  # pip install tqdm for progress bar

from numpy_optimised import NUM_PERMUTATIONS, MAX_TURNS, play_game_wrapper, generate_random_codes, SIZE, NUM_COLOURS


if __name__ == "__main__":
    n_games = 1
    # Generate random test codes instead of sampling from all_permutations
    test_codes = generate_random_codes(n_games)

    strategies = ["info_gain", "random_possible", "info_gain_all", "info_gain_hybrid"]

    # Create all (code, strategy) pairs
    all_args = []
    for strategy in strategies:
        for code in test_codes:
            all_args.append((code, strategy))

    total_games = len(all_args)
    print(f"Running {total_games} total games ({n_games} games × {len(strategies)} strategies)...")

    # Initialize results storage
    results = {strategy: [] for strategy in strategies}

    # Run all games in parallel with progress bar
    with Pool() as pool:
        # Use imap_unordered for better performance and real-time results
        with tqdm(total=total_games) as pbar:
            for strategy, turns in pool.imap_unordered(play_game_wrapper, all_args, chunksize=50):
                results[strategy].append(turns)
                pbar.update(1)

    # Compute statistics
    summary = {}
    for strategy in strategies:
        turns = np.array(results[strategy])
        solved = turns <= MAX_TURNS
        summary[strategy] = {
            'games': len(turns),
            'solved_rate': np.mean(solved) * 100,  # as percentage
            'avg_turns': np.mean(turns[solved]) if np.any(solved) else np.nan,
            'std_turns': np.std(turns[solved]) if np.any(solved) else np.nan,
            'min_turns': np.min(turns[solved]) if np.any(solved) else np.nan,
            'max_turns': np.max(turns[solved]) if np.any(solved) else np.nan
        }

    df_results = pd.DataFrame(summary).T
    print("\nResults:")
    print(df_results.round(2))