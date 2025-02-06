import gymnasium as gym
import torch
import json
from src.models.qr_dqn_model import QRDQNModel  # Make sure it expects 3 channels
from src.agents.qr_dqn_agent import QRDQNAgent
from src.memory.replay_buffer import ReplayBuffer

def main():
    # No wrapper at all, raw 210x160x3 frames
    env = gym.make("ALE/Breakout-v5", render_mode=None)
    
    num_actions = env.action_space.n
    model = QRDQNModel(num_actions=num_actions)       # Must handle 3 channels
    target_model = QRDQNModel(num_actions=num_actions)
    target_model.load_state_dict(model.state_dict())

    agent = QRDQNAgent(model, target_model, num_actions=num_actions)
    buffer = ReplayBuffer(capacity=100000)

    episodes = 500
    batch_size = 32
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 50000
    epsilon = epsilon_start

    log_data = []

    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end)/epsilon_decay)
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(buffer) > batch_size:
                batch = buffer.sample(batch_size)
                loss = agent.update(batch)

            # Target network update (example)
            if len(buffer) % 1000 == 0:
                agent.update_target()

        log_data.append({"episode": ep, "reward": episode_reward})
        print(f"Episode {ep}: Reward = {episode_reward}, Epsilon = {epsilon:.3f}")

    # Save training logs and model
    torch.save(agent.model.state_dict(), "data/checkpoints/model_final.pth")
    with open("data/logs/training_log.json", "w") as f:
        json.dump(log_data, f)

if __name__ == "__main__":
    main()
