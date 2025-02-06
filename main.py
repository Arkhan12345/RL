import gymnasium as gym
import ale_py
import time
import csv
from src.env.wrappers import GrayScaleObservation, ResizeObservation
from src.agents.qr_dqn_agent import QRDQNAgent


def make_env():
    """
    Create Breakout environment with grayscale+resize to (84,84).
    """
    gym.register_envs(ale_py)
    env = gym.make("ALE/Breakout-v5", render_mode=None)  #change env to ALE/Et-v5 to run on complex
    print("env created!!!")
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=(84, 84))
    return env

def main():
    # parameters are adjusted depending on env
    total_timesteps = 1_000_000  # much higher for complex
    batch_size = 32
    learning_rate = 5e-3
    buffer_size = 200_000
    gamma = 0.99
    target_update_interval = 10_000

    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay_fraction = 0.05

    env = make_env()

    agent = QRDQNAgent(
        env=env,
        num_quantiles=51,
        gamma=gamma,
        lr=learning_rate,
        batch_size=batch_size,
        buffer_size=buffer_size,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay_fraction,
        target_update_interval=target_update_interval,
    )

    # CSV logging
    log_filename = "training_rewards-bo.csv"
    f = open(log_filename, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["Episode", "Timestep", "EpisodeReward", "Epsilon"])

    start_time = time.time()
    state, _ = env.reset()
    episode_reward = 0.0
    episode = 0

    for t in range(1, total_timesteps + 1):
        # epsilon decay
        fraction_done = t / (epsilon_decay_fraction * total_timesteps)
        agent.epsilon = max(
            agent.epsilon_end,
            agent.epsilon_start - (agent.epsilon_start - agent.epsilon_end) * fraction_done
        )

        # epsilon-greedy action
        action = agent.choose_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        done_flag = done or truncated

        agent.replay_buffer.push(state, action, reward, next_state, done_flag)
        state = next_state
        episode_reward += reward

        # Update
        agent.train_step()

        # Target net sync
        if t % agent.target_update_interval == 0:
            agent.update_target_network()

        if done_flag:
            episode += 1
            print(f"Step={t} | Episode={episode} | Reward={episode_reward:.2f} | Epsilon={agent.epsilon:.3f}")
            writer.writerow([episode, t, episode_reward, agent.epsilon]) # log
            f.flush()

            state, _ = env.reset()
            episode_reward = 0.0

    elapsed = time.time() - start_time
    print(f"Finished training {total_timesteps} timesteps in {elapsed:.2f} seconds.")

    f.close()
    env.close()
    print("DONE!!!")

if __name__ == "__main__":
    main()
