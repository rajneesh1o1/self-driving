# Self-Driving Car — Imitation Learning

A 2D self-driving car that learns to navigate through obstacles by imitating an A* pathfinding expert.

The car sees a tiny 6x6 grid around itself (2 cells in each direction) and knows the direction to its goal. A neural network trained on expert demonstrations decides which of 8 directions to move each step.

## How It Works

### The World

An infinite procedural 2D world generated in chunks. Obstacles of various sizes are scattered randomly. The car (2x2 cells) must navigate to goal flags placed at random positions.

### The Expert (A*)

An A* pathfinding navigator that plans optimal routes using only what the car can see. It builds a known-map incrementally as the car explores, replans when new obstacles appear, and always finds a path if one exists.

### The Model

**Imitation learning (behavioral cloning):** a small CNN+MLP trained to copy the A* expert's decisions.

```
Input:   6x6 FOV grid (obstacles, empty, car)  +  2D goal direction
           |                                          |
         [CNN: 2x Conv2d + Pool]              [normalised vector]
           |                                          |
           +------------------+------ ----------------+
                              |
                       [MLP: 3 layers]
                              |
Output:  8 logits (one per direction: N, NE, E, SE, S, SW, W, NW)
```

At runtime, the highest-scoring direction wins, with two safety layers:
- **Obstacle check:** if a direction has a visible wall, it gets a heavy penalty
- **Visit memory:** cells visited before get penalised to prevent looping

### Pipeline

```
1. Play/observe A* expert  -->  2. Collect (FOV, goal_dir, action) data
                                         |
3. Train policy network    <-------------+
         |
4. Deploy: policy drives the car in real-time
```

## Project Structure

```
self-driving/
  game/
    car_game.py          # Game engine, chunk world, A* navigator
    renderer.py          # Pygame visual renderer
    play.py              # Interactive game (manual or A* auto-pilot)
    generate_data.py     # Collect training data from A* expert
    driving_data.h5      # Generated training data
  model/
    policy.py            # DrivingPolicy network (CNN+MLP, 8-class output)
    train_policy.py      # Training script (cross-entropy, 4x augmentation)
    train.py             # JEPA world model training (experimental)
    jepa.py              # JEPA architecture (experimental)
    module.py            # Transformer blocks, utilities
    checkpoints/         # Saved model weights
  drive.py               # Run the trained model in the game
  requirements.txt
```

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### 1. Play the game manually

```bash
cd self-driving/game
python play.py
```

Arrow keys to drive, SPACE to toggle A* auto-pilot, R to restart.

### 2. Watch the A* expert drive

```bash
cd self-driving/game
python play.py --auto
```

### 3. Generate training data

```bash
cd self-driving/game
python generate_data.py --episodes 60
```

This runs the A* expert for 60 episodes, capturing the 6x6 FOV, action taken, and goal direction at every step. Saves to `driving_data.h5`.

Options:
- `--episodes N` — number of episodes (default 10)
- `--seed N` — random seed
- `--append` — add to existing data instead of overwriting
- `--output NAME` — output file name (without .h5)

### 4. Train the policy

```bash
cd self-driving/model
python train_policy.py --data ../game/driving_data.h5 --epochs 30
```

Trains with 4x data augmentation (horizontal flip, vertical flip, 180-degree rotation) and cross-entropy loss. Saves best model to `checkpoints/policy_best.pt`.

Options:
- `--epochs N` — training epochs (default 50)
- `--lr RATE` — learning rate (default 3e-3)
- `--batch_size N` — batch size (default 64)
- `--hidden N` — hidden layer size (default 128)

### 5. Run the trained model

```bash
cd self-driving
python drive.py --checkpoint model/checkpoints/policy_best.pt
```

Watch the model drive. Compare side-by-side with the A* expert:

```bash
python drive.py --checkpoint model/checkpoints/policy_best.pt --expert-compare
```

Controls: R to restart, Q/ESC to quit.

## Training Data Format (HDF5)

| Dataset      | Shape    | Description                          |
|-------------|----------|--------------------------------------|
| `fov`       | (N,6,6)  | Local field of view per step         |
| `action`    | (N,2)    | Action taken (ax, ay) in {-1, 0, 1} |
| `goal_dir`  | (N,2)    | Normalised direction to goal (dy,dx) |
| `ep_len`    | (E,)     | Length of each episode               |
| `ep_offset` | (E,)     | Start index of each episode          |

## Key Constants

| Constant     | Value | Meaning                              |
|-------------|-------|--------------------------------------|
| `CAR_SIZE`  | 2     | Car is 2x2 cells                     |
| `VISIBILITY`| 2     | Car sees 2 cells from its outline    |
| `LOCAL_FOV` | 6     | Total FOV grid: 2*2+2 = 6x6         |
| `CHUNK_SIZE`| 32    | World generated in 32x32 chunks      |



(venv) rajneesh@Rajneeshs-MacBook-Air game % python play_road.py --auto python drive_road.py --checkpoint model/checkpoints/policy_best.pt


python drive_road.py --checkpoint model/checkpoints/policy_best.pt --expert-compare
