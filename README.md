# LSTM Mastermind Solver

Training recurrent neural networks (LSTM) to crack the Mastermind code-breaking game using reinforcement learning.

This project uses PyTorch LSTM networks to learn optimal Mastermind strategies. The neural network processes the complete game history sequentially and learns to make strategic guesses through policy gradient reinforcement learning.

## About Mastermind

Mastermind is a classic code-breaking game where:
- A secret code of 4 colors is hidden (colors 0-5)
- You have 12 attempts to guess the code
- After each guess, you receive feedback:
  - **Black pegs**: correct color in correct position
  - **White pegs**: correct color in wrong position

## Features

- Complete Mastermind game implementation
- Interactive game player for humans
- LSTM-based AI trained with policy gradients
- Sequential processing of game history
- Reinforcement learning training approach
- Model save/load functionality
- Comprehensive testing and evaluation tools

## Installation

1. Clone the repository
2. Create a virtual environment (recommended):
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
# source .venv/bin/activate  # On Mac/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Play Mastermind Yourself

```bash
python src/play_mastermind.py
```

Interactive game where you try to crack the code!

### Train an LSTM Agent

```bash
python src/train_rnn.py --episodes 1000 --games-per-episode 32
```

Options:
- `--episodes N`: Number of training episodes (default: 1000)
- `--games-per-episode N`: Games per training batch (default: 32)
- `--hidden-size N`: LSTM hidden state size (default: 128)
- `--num-layers N`: Number of LSTM layers (default: 2)
- `--learning-rate LR`: Learning rate (default: 0.001)
- `--temperature T`: Sampling temperature (default: 1.0, decays during training)
- `--device DEVICE`: cpu, cuda, or auto (default: auto)
- `--save-dir DIR`: Directory to save models (default: models)
- `--save-interval N`: Save checkpoint every N episodes (default: 50)

Training automatically:
- Saves checkpoints every 50 episodes to `models/`
- Saves best model based on win rate
- Logs training progress every 10 episodes
- Decays exploration temperature over time

### Test a Trained Model

```bash
python src/test_rnn.py models/best_model.pt --num-games 100 --watch 5
```

Options:
- `model_path`: Path to model checkpoint (required)
- `--num-games N`: Number of games to test (default: 100)
- `--watch N`: Watch N games with detailed output (default: 0)
- `--device DEVICE`: cpu, cuda, or auto (default: auto)
- `--verbose`: Show detailed game information

## How It Works

### LSTM Architecture

**Model Components:**
- **Input**: One-hot encoded guess (4 positions × 6 colors) + normalized feedback (2 values)
  - Input size: 26 features per time step
- **LSTM Layers**: 2 layers with 128 hidden units each
- **Output**: 24 values (4 positions × 6 colors) with softmax per position
- **Dropout**: 0.2 between layers for regularization

The LSTM processes the game history sequentially, maintaining hidden state to remember previous guesses and feedback.

### Training Approach

**Policy Gradient Reinforcement Learning:**
1. Model plays multiple games, sampling actions from its probability distribution
2. Games are scored with a reward function
3. Model parameters are updated to increase probability of high-reward actions
4. Temperature parameter controls exploration vs exploitation

**Reward Function:**

For wins:
- Base score: 100 points
- Efficiency bonus: (12 - attempts_used) × 10
- Example: Winning in 5 attempts = 100 + (12-5)×10 = 170 points

For losses:
- Partial credit based on best feedback achieved
- Black pegs: best_black × 15
- White pegs: best_white × 2

**Training Techniques:**
- Reward normalization across batches
- Gradient clipping (max norm = 1.0)
- Temperature annealing (exploration → exploitation)
- Adam optimizer with learning rate 0.001

### Why LSTM?

LSTM is well-suited for Mastermind because:
- Sequential nature matches the game's turn-based structure
- Hidden state remembers patterns from previous guesses
- Can learn to eliminate possibilities based on feedback history
- Handles variable-length sequences (games end at different attempts)

## Project Structure

```
neat-mastermind-solver/
├── src/
│   ├── mastermind.py          # Game engine
│   ├── play_mastermind.py     # Interactive game for humans
│   ├── rnn_model.py           # LSTM model architecture
│   ├── train_rnn.py           # Training script (policy gradient)
│   └── test_rnn.py            # Testing and evaluation
├── models/                     # Saved models (created during training)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Example Training Session

```bash
# Start training
python src/train_rnn.py --episodes 500 --games-per-episode 32

# Output shows:
# - Episode progress
# - Win rate, average reward, average attempts
# - Best model saves when win rate improves
# - Checkpoints every 50 episodes

# After training, test the best model:
python src/test_rnn.py models/best_model.pt --num-games 100 --watch 3

# Results show:
# - Win rate (e.g., 75/100 = 75%)
# - Average attempts for wins
# - Detailed statistics
```

## Tips for Better Results

1. **Train longer** - 1000+ episodes often needed for good convergence
2. **Increase batch size** - More games per episode (32-64) for stable gradients
3. **Adjust architecture** - Try larger hidden size (256) or more layers (3)
4. **Tune learning rate** - Lower rates (0.0005) for more stable training
5. **Monitor training** - Watch for increasing win rates and decreasing attempts
6. **Use GPU** - Significantly faster training with `--device cuda`

## Comparison with NEAT

This project previously used NEAT (NeuroEvolution of Augmenting Topologies). The LSTM approach offers:

**Advantages:**
- More stable gradient-based learning
- Better sample efficiency with backpropagation
- Explicit sequential modeling with LSTM
- Easier to scale and parallelize

**Trade-offs:**
- Requires more careful hyperparameter tuning
- Less automatic architecture search
- More computation per update (but fewer updates needed)

## Future Enhancements

- Attention mechanisms for better history utilization
- Multi-task learning (different code lengths/colors)
- Curriculum learning (easier games → harder games)
- Comparison with classic algorithms (Knuth's algorithm)
- Actor-Critic or PPO for more stable training
- Web interface for watching trained agents

## License

MIT License

## Acknowledgments

- PyTorch for the deep learning framework
- Classic Mastermind game by Mordecai Meirowitz
