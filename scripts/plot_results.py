import pandas as pd
import matplotlib.pyplot as plt

def main():
    csv_file = "training_rewards-bo.csv"
    df = pd.read_csv(csv_file)

    # basic info
    print("\n===== CSV Info =====")
    print(df.info())

    # smoothing
    rolling_window = 100

    # average
    df["RollingReward"] = df["EpisodeReward"].rolling(rolling_window).mean()

    # plot rewards
    plt.figure(figsize=(10, 6))
    plt.plot(df["Episode"], df["EpisodeReward"], label="Episode Reward", color="blue", alpha=0.4)
    plt.plot(df["Episode"], df["RollingReward"], label=f"Rolling Avg (window={rolling_window})", color="orange")

    plt.title("Smoothed Episode Rewards Over Training")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # plt.show()
    plt.savefig("smoothed_rewards_plot-bo.png")

if __name__ == "__main__":
    main()