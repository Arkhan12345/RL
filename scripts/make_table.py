import sys
import os

# Optionally ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.append(project_root)

import json
import numpy as np

def make_convergence_table(log_path, output_path="data/results/convergence_table.txt"):
    with open(log_path, "r") as f:
        data = json.load(f)
    rewards = [d["reward"] for d in data]

    if len(rewards) < 100:
        print("Not enough episodes to compute last 100 stats. Using all episodes.")
        last_episodes = rewards
    else:
        last_episodes = rewards[-100:]

    mean_reward = np.mean(last_episodes)
    std_reward = np.std(last_episodes)
    max_reward = np.max(last_episodes)
    min_reward = np.min(last_episodes)

    lines = [
        "Convergence Table (Last 100 Episodes)\n",
        "-------------------------------------\n",
        f"Mean Reward: {mean_reward:.2f}\n",
        f"Std Dev Reward: {std_reward:.2f}\n",
        f"Max Reward: {max_reward}\n",
        f"Min Reward: {min_reward}\n"
    ]

    # Print to console
    for line in lines:
        print(line, end='')

    # Save to file
    with open(output_path, "w") as f:
        f.writelines(lines)

if __name__ == "__main__":
    make_convergence_table("data/logs/training_log.json")
