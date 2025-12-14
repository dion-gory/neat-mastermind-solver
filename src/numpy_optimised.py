import numpy as np

MAX_TURNS = 50
SIZE = 20
NUM_COLOURS = 100
NUM_PERMUTATIONS = NUM_COLOURS ** SIZE

# Sampling parameters for large search spaces
# These can be increased for better accuracy at the cost of speed
MAX_REMAINING_SAMPLE = 10000  # Max size to keep for remaining_possible_solutions
MAX_CANDIDATE_SAMPLE = 5000   # Max candidates to evaluate for info gain
MAX_INFO_GAIN_EVAL = 2000     # Max remaining solutions to use when computing info gain


def configure_problem(size, num_colours, max_turns=12):
    """
    Configure the problem parameters.
    Call this before running games to change the problem size.

    Parameters:
    - size: int, number of positions
    - num_colours: int, number of colors
    - max_turns: int, maximum number of turns allowed
    """
    global SIZE, NUM_COLOURS, NUM_PERMUTATIONS, MAX_TURNS
    SIZE = size
    NUM_COLOURS = num_colours
    NUM_PERMUTATIONS = num_colours ** size
    MAX_TURNS = max_turns


def configure_sampling(max_remaining=10000, max_candidates=5000, max_eval=2000):
    """
    Configure sampling parameters for large search spaces.

    Parameters:
    - max_remaining: Max size for remaining_possible_solutions
    - max_candidates: Max candidates to evaluate for info gain
    - max_eval: Max remaining solutions to use when computing info gain
    """
    global MAX_REMAINING_SAMPLE, MAX_CANDIDATE_SAMPLE, MAX_INFO_GAIN_EVAL
    MAX_REMAINING_SAMPLE = max_remaining
    MAX_CANDIDATE_SAMPLE = max_candidates
    MAX_INFO_GAIN_EVAL = max_eval


def generate_random_codes(n_samples):
    """Generate n random codes from the search space."""
    return np.random.randint(0, NUM_COLOURS, size=(n_samples, SIZE))


def sample_search_space(n_samples):
    """
    Sample n codes from the full search space.
    For small spaces, return all codes. For large spaces, sample uniformly.
    """
    if NUM_PERMUTATIONS <= n_samples:
        # Small enough to enumerate all
        from itertools import product
        return np.array(list(product(np.arange(NUM_COLOURS), repeat=SIZE)))
    else:
        # Too large, sample randomly
        return generate_random_codes(n_samples)


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


def information_gain_fully_vectorized(remaining_possible_solutions, set_of_choices, max_sample_size=None):
    """
    Fully vectorized information gain calculation with minimal Python loops.
    Now supports sampling for large search spaces.

    Parameters:
    - remaining_possible_solutions: np.array of shape (S, SIZE)
    - set_of_choices: np.array of shape (C, SIZE)
    - max_sample_size: int, maximum number of remaining solutions to use (None = use all)

    Returns:
    - info_gain: np.array of shape (C,) with entropy for each choice
    """
    # Sample remaining solutions if too large
    if max_sample_size is not None and len(remaining_possible_solutions) > max_sample_size:
        sample_indices = np.random.choice(len(remaining_possible_solutions), max_sample_size, replace=False)
        remaining_sample = remaining_possible_solutions[sample_indices]
    else:
        remaining_sample = remaining_possible_solutions

    S = len(remaining_sample)
    C = len(set_of_choices)

    # Get all pairwise scores
    all_scores = get_scores_fully_vectorized(set_of_choices, remaining_sample)

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
    Now supports sampling for large search spaces.

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
        # Random code from the entire search space
        return generate_random_codes(1)

    elif strategy == "random_possible":
        idx = np.random.randint(len(remaining_possible_solutions))
        return remaining_possible_solutions[idx:idx + 1]

    elif strategy == "info_gain":
        # Calculate entropy for remaining possible solutions (with sampling if needed)
        # Sample candidates if there are too many
        if len(remaining_possible_solutions) > MAX_CANDIDATE_SAMPLE:
            candidate_indices = np.random.choice(len(remaining_possible_solutions), MAX_CANDIDATE_SAMPLE, replace=False)
            candidates = remaining_possible_solutions[candidate_indices]
        else:
            candidates = remaining_possible_solutions
            candidate_indices = np.arange(len(remaining_possible_solutions))

        entropies = information_gain_fully_vectorized(
            remaining_possible_solutions,
            candidates,
            max_sample_size=MAX_INFO_GAIN_EVAL
        )
        # Find the choice with maximum entropy
        best_idx = np.argmax(entropies)

        # Map back to the original index if we sampled
        if len(remaining_possible_solutions) > MAX_CANDIDATE_SAMPLE:
            return candidates[best_idx:best_idx + 1]
        else:
            return remaining_possible_solutions[best_idx:best_idx + 1]

    elif strategy == "info_gain_all":
        # Calculate entropy for sampled codes from the entire search space
        all_candidates = sample_search_space(MAX_CANDIDATE_SAMPLE)

        entropies = information_gain_fully_vectorized(
            remaining_possible_solutions,
            all_candidates,
            max_sample_size=MAX_INFO_GAIN_EVAL
        )
        # Find the choice with maximum entropy
        best_idx = np.argmax(entropies)
        return all_candidates[best_idx:best_idx + 1]

    elif strategy == "info_gain_hybrid":
        # Use info_gain for small spaces, info_gain_all for large spaces
        if len(remaining_possible_solutions) <= 100:
            return make_choice("info_gain", remaining_possible_solutions)
        else:
            return make_choice("info_gain_all", remaining_possible_solutions)

    else:
        raise ValueError("Unknown strategy")


def play_game(code, strategy):
    """
    Play a single game of Mastermind.
    Now supports large search spaces through sampling and constraint tracking.

    Parameters:
    - code: np.array of shape (1, SIZE) or (SIZE,)
    - strategy: string specifying the strategy

    Returns:
    - num_turns: int, number of turns taken (MAX_TURNS + 1 if not solved)
    """
    if code.ndim == 1:
        code = code.reshape(1, -1)

    # Track all guesses and scores for constraint filtering
    guess_history = []
    score_history = []

    # Initialize remaining possible solutions
    # For large spaces, we'll regenerate samples each turn
    use_sampling = NUM_PERMUTATIONS > MAX_REMAINING_SAMPLE

    if not use_sampling:
        # Small enough to enumerate all
        from itertools import product
        remaining_possible_solutions = np.array(list(product(np.arange(NUM_COLOURS), repeat=SIZE)))
    else:
        # For large spaces, we'll generate samples fresh each turn
        remaining_possible_solutions = None

    for turn in range(1, MAX_TURNS + 1):
        # For large spaces, regenerate sample and apply all constraints
        if use_sampling:
            # Generate a fresh sample
            remaining_possible_solutions = sample_search_space(MAX_REMAINING_SAMPLE)

            # Apply all previous constraints
            for guess, score in zip(guess_history, score_history):
                all_scores = get_scores_fully_vectorized(
                    guess.reshape(1, -1),
                    remaining_possible_solutions
                )[0]  # shape: (N, 2)

                matching = (all_scores[:, 0] == score[0]) & (all_scores[:, 1] == score[1])
                remaining_possible_solutions = remaining_possible_solutions[matching]

                if len(remaining_possible_solutions) == 0:
                    # No solutions in current sample, get a new one
                    remaining_possible_solutions = sample_search_space(MAX_REMAINING_SAMPLE)
                    # Re-apply all constraints
                    for g, s in zip(guess_history, score_history):
                        all_scores = get_scores_fully_vectorized(
                            g.reshape(1, -1),
                            remaining_possible_solutions
                        )[0]
                        matching = (all_scores[:, 0] == s[0]) & (all_scores[:, 1] == s[1])
                        remaining_possible_solutions = remaining_possible_solutions[matching]
                        if len(remaining_possible_solutions) == 0:
                            break

        # Special case: only one solution left
        if len(remaining_possible_solutions) == 1:
            return turn

        # If no solutions remain (shouldn't happen but check anyway)
        if len(remaining_possible_solutions) == 0:
            # Fallback: generate random guess
            choice = generate_random_codes(1)
        else:
            # Make choice
            choice = make_choice(strategy, remaining_possible_solutions)

        # Get score
        score = get_scores_fully_vectorized(choice, code)[0, 0]  # shape: (2,)

        # Record this guess and score
        guess_history.append(choice[0])
        score_history.append(score)

        # Check if solved
        if score[0] == SIZE:
            return turn

        # Prune remaining solutions (for non-sampling mode)
        if not use_sampling:
            all_scores = get_scores_fully_vectorized(
                choice,
                remaining_possible_solutions
            )[0]  # shape: (N, 2)

            matching = (all_scores[:, 0] == score[0]) & (all_scores[:, 1] == score[1])
            remaining_possible_solutions = remaining_possible_solutions[matching]

    # Failed to solve
    return MAX_TURNS + 1


def play_game_wrapper(args):
    code, strategy = args
    return strategy, play_game(code, strategy)