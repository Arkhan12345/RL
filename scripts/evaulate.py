import sys
import os

# Optionally ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.append(project_root)

import gymnasium as gym
import torch
#from src.env.wrappers import PreprocessingWrapper
from src.models.qr_dqn_model import QRDQNModel

def evaluate(checkpoint_path, episodes=10, render=False):
    env = gym.make("ALE/Breakout-v5", render_mode="human" if render else None)
    #env = PreprocessingWrapper(env)
    model = QRDQNModel(num_actions=env.action_space.n)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    total_rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            with torch.no_grad():
                state_t = torch.tensor(state).unsqueeze(0)
                quantiles = model(state_t)
                q_values = quantiles.mean(dim=2)
                action = q_values.argmax(dim=1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            ep_reward += reward
        total_rewards.append(ep_reward)
        print(f"Evaluation Episode {ep}: Reward = {ep_reward}")

    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"Average Reward over {episodes} episodes: {avg_reward}")

if __name__ == "__main__":
    evaluate("data/checkpoints/model_final.pth", episodes=10, render=False)
