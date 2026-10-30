import argparse
import csv
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Q-network
# ============================================================

class QNetwork(nn.Module):
    def __init__(self, env, layer_norm=False):
        super().__init__()

        input_dim = np.array(env.observation_space.shape).prod()
        output_dim = env.action_space.n

        if layer_norm:
            self.network = nn.Sequential(
                nn.Linear(input_dim, 120),
                nn.LayerNorm(120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.LayerNorm(84),
                nn.ReLU(),
                nn.Linear(84, output_dim),
            )

        else:
            self.network = nn.Sequential(
                nn.Linear(input_dim, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, output_dim),
            )

    def forward(self, x):
        return self.network(x)


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model",
    )

    parser.add_argument(
        "--env-id",
        type=str,
        default="LunarLander-v2",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="First evaluation seed",
    )

    parser.add_argument(
        "--layer-norm",
        action="store_true",
        help="Use LayerNorm architecture when loading the model",
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["DQN", "DDQN"],
        required=True,
        help="Algorithm used to train the model",
    )

    parser.add_argument(
        "--training-seed",
        type=int,
        required=True,
        help="Seed used to train the model",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CSV file",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    # --------------------------------------------------------
    # 1. Environment
    # --------------------------------------------------------

    env = gym.make(args.env_id)

    # --------------------------------------------------------
    # 2. Rebuild network architecture
    # --------------------------------------------------------

    model = QNetwork(
        env,
        layer_norm=args.layer_norm,
    )

    # --------------------------------------------------------
    # 3. Load trained weights
    # --------------------------------------------------------

    state_dict = torch.load(
        args.model_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    # Disable training-specific behaviour
    model.eval()

    # --------------------------------------------------------
    # 4. Containers
    # --------------------------------------------------------

    results = []

    episode_biases = []
    episode_returns = []

    terminated_biases = []
    truncated_biases = []

    terminated_episodes = 0
    truncated_episodes = 0

    # --------------------------------------------------------
    # 5. Evaluation episodes
    # --------------------------------------------------------

    for episode in range(args.episodes):

        eval_seed = args.seed + episode

        obs, _ = env.reset(
            seed=eval_seed
        )

        done = False

        q_values_episode = []
        rewards_episode = []

        terminated = False
        truncated = False

        # ----------------------------------------------------
        # Run one complete observed trajectory
        # ----------------------------------------------------

        while not done:

            obs_tensor = torch.tensor(
                obs,
                dtype=torch.float32,
            ).unsqueeze(0)

            # Q(s_t, a) for all actions
            with torch.no_grad():
                q_values = model(obs_tensor)

            # Greedy evaluation policy
            action = torch.argmax(
                q_values,
                dim=1,
            ).item()

            # Q(s_t, a_t)
            selected_q = q_values[
                0,
                action,
            ].item()

            # Environment transition
            (
                next_obs,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)

            q_values_episode.append(
                selected_q
            )

            rewards_episode.append(
                reward
            )

            obs = next_obs

            done = (
                terminated
                or truncated
            )

        # ----------------------------------------------------
        # 6. Episodic performance
        #
        # Undiscounted cumulative reward
        # ----------------------------------------------------

        total_return = sum(
            rewards_episode
        )

        episode_returns.append(
            total_return
        )

        # ----------------------------------------------------
        # 7. Monte Carlo return
        #
        # G_t =
        # r_t + gamma r_(t+1) + ...
        #
        # For naturally terminated episodes, this is the
        # complete observed return until the terminal state.
        #
        # For truncated episodes, this is the finite-horizon
        # return observed up to the time-limit cutoff.
        # ----------------------------------------------------

        mc_returns = []

        G = 0.0

        for reward in reversed(
            rewards_episode
        ):
            G = (
                reward
                + args.gamma * G
            )

            mc_returns.append(G)

        mc_returns.reverse()

        # ----------------------------------------------------
        # 8. Convert to NumPy
        # ----------------------------------------------------

        q_values_episode = np.array(
            q_values_episode,
            dtype=np.float64,
        )

        mc_returns = np.array(
            mc_returns,
            dtype=np.float64,
        )

        # ----------------------------------------------------
        # 9. Bias at every timestep
        #
        # b_t = Q(s_t, a_t) - G_t
        # ----------------------------------------------------

        biases = (
            q_values_episode
            - mc_returns
        )

        # ----------------------------------------------------
        # 10. Mean bias for this episode
        # ----------------------------------------------------

        mean_bias = float(
            biases.mean()
        )

        episode_biases.append(
            mean_bias
        )

        # ----------------------------------------------------
        # 11. Track termination type
        # ----------------------------------------------------

        if terminated:
            terminated_episodes += 1

            terminated_biases.append(
                mean_bias
            )

        if truncated:
            truncated_episodes += 1

            truncated_biases.append(
                mean_bias
            )

        # ----------------------------------------------------
        # 12. Save row
        # ----------------------------------------------------

        results.append(
            {
                "algorithm": args.algorithm,
                "layer_norm": args.layer_norm,
                "training_seed": args.training_seed,
                "eval_seed": eval_seed,
                "episode": episode + 1,
                "return": total_return,
                "mean_bias": mean_bias,
                "steps": len(rewards_episode),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

        # ----------------------------------------------------
        # 13. Print episode
        # ----------------------------------------------------

        termination_label = (
            "Truncated"
            if truncated
            else "Terminated"
        )

        print(
            f"Episode {episode + 1:02d} | "
            f"Return: {total_return:8.2f} | "
            f"Mean bias: {mean_bias:8.2f} | "
            f"Steps: {len(rewards_episode):4d} | "
            f"{termination_label}"
        )

    # --------------------------------------------------------
    # 14. Arrays
    # --------------------------------------------------------

    episode_returns = np.array(
        episode_returns,
        dtype=np.float64,
    )

    episode_biases = np.array(
        episode_biases,
        dtype=np.float64,
    )

    terminated_biases = np.array(
        terminated_biases,
        dtype=np.float64,
    )

    truncated_biases = np.array(
        truncated_biases,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # 15. Global summary
    # --------------------------------------------------------

    print()
    print("Evaluation summary")
    print("------------------")

    print(
        f"Episodes:             "
        f"{args.episodes}"
    )

    print(
        f"Terminated:           "
        f"{terminated_episodes}"
    )

    print(
        f"Truncated:            "
        f"{truncated_episodes}"
    )

    print()

    print(
        f"Mean return:          "
        f"{episode_returns.mean():.2f}"
    )

    if len(episode_returns) > 1:
        print(
            f"Std return:           "
            f"{episode_returns.std(ddof=1):.2f}"
        )

    else:
        print(
            "Std return:           N/A"
        )

    print()

    # Bias using ALL evaluation episodes
    print(
        f"Mean bias (all):      "
        f"{episode_biases.mean():.2f}"
    )

    if len(episode_biases) > 1:
        print(
            f"Std bias (all):       "
            f"{episode_biases.std(ddof=1):.2f}"
        )

    else:
        print(
            "Std bias (all):       N/A"
        )

    # Diagnostic:
    # naturally terminated episodes only
    if len(terminated_biases) > 0:

        print(
            f"Mean bias terminated: "
            f"{terminated_biases.mean():.2f}"
        )

    else:

        print(
            "Mean bias terminated: N/A"
        )

    # Diagnostic:
    # time-limit truncated episodes only
    if len(truncated_biases) > 0:

        print(
            f"Mean bias truncated:  "
            f"{truncated_biases.mean():.2f}"
        )

    else:

        print(
            "Mean bias truncated:  N/A"
        )

    # --------------------------------------------------------
    # 16. Output filename
    # --------------------------------------------------------

    if args.output is None:

        ln_label = (
            "ln"
            if args.layer_norm
            else "no_ln"
        )

        output_path = (
            f"results/"
            f"{args.algorithm.lower()}_"
            f"{ln_label}_"
            f"trainseed{args.training_seed}.csv"
        )

    else:

        output_path = args.output

    # --------------------------------------------------------
    # 17. Create directory
    # --------------------------------------------------------

    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # 18. Save CSV
    # --------------------------------------------------------

    fieldnames = [
        "algorithm",
        "layer_norm",
        "training_seed",
        "eval_seed",
        "episode",
        "return",
        "mean_bias",
        "steps",
        "terminated",
        "truncated",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print()
    print(
        f"Results saved to: "
        f"{output_path}"
    )

    # --------------------------------------------------------
    # 19. Cleanup
    # --------------------------------------------------------

    env.close()


if __name__ == "__main__":
    main()