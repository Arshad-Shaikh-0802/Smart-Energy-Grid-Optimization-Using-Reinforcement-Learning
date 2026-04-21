import os
import csv
import time
import argparse
import numpy as np

from grid_env import SmartGridEnv
from dqn_agent import DQNAgent


RESULTS_DIR = "results"
LOG_FILE    = os.path.join(RESULTS_DIR, "training_log.csv")
BEST_MODEL  = os.path.join(RESULTS_DIR, "best_model.pth")
FINAL_MODEL = os.path.join(RESULTS_DIR, "final_model.pth")

LOG_FIELDS = [
    "episode", "total_reward", "renewable_ratio", "total_cost",
    "blackout_energy", "epsilon", "loss_mean", "eval_reward"
]


def evaluate(env, agent, n_episodes=3):
    """Run greedy evaluation episodes and return mean reward."""
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        for _ in range(24):
            action = agent.select_action(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        rewards.append(ep_reward)
    return float(np.mean(rewards))


def train(num_episodes=500, eval_every=25, noise_level=0.1):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    env      = SmartGridEnv(noise_level=noise_level)
    eval_env = SmartGridEnv(noise_level=0.0)  # deterministic for eval
    agent    = DQNAgent()

    best_eval_reward = float("-inf")
    start_time = time.time()

    print("=" * 60)
    print(" Smart Grid RL Training")
    print(f" Episodes: {num_episodes} | Eval every: {eval_every}")
    print("=" * 60)

    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()

        for episode in range(1, num_episodes + 1):
            state, _ = env.reset()
            ep_reward = 0.0
            losses = []

            for step in range(24):
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.store(state, action, reward, next_state, float(done))
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)

                state = next_state
                ep_reward += reward
                if done:
                    break

            agent.decay_epsilon()

            final_info = info
            renewable_ratio = final_info.get("renewable_ratio", 0)
            total_cost      = final_info.get("total_cost", 0)
            blackout_energy = final_info.get("blackout_energy", 0)
            loss_mean       = float(np.mean(losses)) if losses else 0.0

            # Evaluation
            eval_reward = None
            if episode % eval_every == 0:
                eval_reward = evaluate(eval_env, agent)
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    agent.save(BEST_MODEL)
                    print(f"  ★ New best eval reward: {eval_reward:.2f} → saved")

            # Logging
            row = {
                "episode":        episode,
                "total_reward":   round(ep_reward, 4),
                "renewable_ratio": round(renewable_ratio, 4),
                "total_cost":     round(total_cost, 4),
                "blackout_energy": round(blackout_energy, 4),
                "epsilon":        round(agent.epsilon, 4),
                "loss_mean":      round(loss_mean, 6),
                "eval_reward":    round(eval_reward, 4) if eval_reward is not None else "",
            }
            writer.writerow(row)
            f.flush()

            if episode % 50 == 0 or episode == 1:
                elapsed = time.time() - start_time
                print(
                    f"  Ep {episode:4d}/{num_episodes} | "
                    f"Reward: {ep_reward:7.2f} | "
                    f"Renewable: {renewable_ratio:.1%} | "
                    f"Cost: ${total_cost:.3f} | "
                    f"ε: {agent.epsilon:.3f} | "
                    f"Time: {elapsed:.0f}s"
                )

    agent.save(FINAL_MODEL)
    print("\n" + "=" * 60)
    print(f" Training complete! Best eval reward: {best_eval_reward:.2f}")
    print(f" Models saved in: {RESULTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Smart Grid DQN Agent")
    parser.add_argument("--episodes",   type=int,   default=500)
    parser.add_argument("--eval-every", type=int,   default=25)
    parser.add_argument("--noise",      type=float, default=0.1)
    args = parser.parse_args()
    train(num_episodes=args.episodes, eval_every=args.eval_every, noise_level=args.noise)
