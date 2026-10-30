import os
import subprocess
import sys
from pathlib import Path


# ============================================================
# Experiment configuration
# ============================================================

TOTAL_TIMESTEPS = 300_000

TRAINING_SEEDS = [0, 1, 2, 3, 4]

EVALUATION_EPISODES = 20

EVALUATION_SEED = 0


# ============================================================
# Paths
# ============================================================

# run_experiments.py is inside:
#
# DeepRL_TP/
# └── src/
#     └── run_experiments.py
#
# Therefore parent.parent = DeepRL_TP

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"

TRAIN_SCRIPT = SRC_DIR / "train_agent.py"

EVALUATE_SCRIPT = SRC_DIR / "evaluate_bias.py"


# ============================================================
# Experimental conditions
# ============================================================

EXPERIMENTS = [
    {
        "algorithm": "DQN",
        "layer_norm": False,
    },
    {
        "algorithm": "DQN",
        "layer_norm": True,
    },
    {
        "algorithm": "DDQN",
        "layer_norm": False,
    },
    {
        "algorithm": "DDQN",
        "layer_norm": True,
    },
]


# ============================================================
# Utility: run command
# ============================================================

def run_command(command):
    """
    Execute a command and print its output live.

    Also returns all stdout lines so that we can extract
    the saved model path after training.
    """

    print()
    print("============================================================")
    print("Executing:")
    print(" ".join(str(x) for x in command))
    print("============================================================")
    print()

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []

    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {process.returncode}"
        )

    return output_lines


# ============================================================
# Utility: extract model path
# ============================================================

def extract_model_path(output_lines):
    """
    train_agent.py prints:

    Model saved to: runs/...

    This function extracts that path.
    """

    prefix = "Model saved to:"

    for line in reversed(output_lines):

        stripped = line.strip()

        if stripped.startswith(prefix):

            model_path = stripped[len(prefix):].strip()

            return model_path

    raise RuntimeError(
        "Could not find saved model path in training output."
    )


# ============================================================
# Train one model
# ============================================================

def train_model(
    algorithm,
    layer_norm,
    training_seed,
):

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--algorithm",
        algorithm,
        "--env-id",
        "LunarLander-v2",
        "--total-timesteps",
        str(TOTAL_TIMESTEPS),
        "--seed",
        str(training_seed),
        "--save-model",
    ]

    if layer_norm:
        command.append("--layer-norm")

    output_lines = run_command(command)

    model_path = extract_model_path(
        output_lines
    )

    return model_path


# ============================================================
# Evaluate one model
# ============================================================

def evaluate_model(
    model_path,
    algorithm,
    layer_norm,
    training_seed,
):

    command = [
        sys.executable,
        str(EVALUATE_SCRIPT),
        "--model-path",
        model_path,
        "--algorithm",
        algorithm,
        "--training-seed",
        str(training_seed),
        "--episodes",
        str(EVALUATION_EPISODES),
        "--seed",
        str(EVALUATION_SEED),
    ]

    if layer_norm:
        command.append("--layer-norm")

    run_command(command)


# ============================================================
# Main experiment loop
# ============================================================

def main():

    print()
    print("============================================================")
    print("FINAL EXPERIMENT")
    print("============================================================")

    print(
        f"Configurations:        {len(EXPERIMENTS)}"
    )

    print(
        f"Training seeds:        {len(TRAINING_SEEDS)}"
    )

    print(
        f"Total trainings:       "
        f"{len(EXPERIMENTS) * len(TRAINING_SEEDS)}"
    )

    print(
        f"Timesteps per model:   {TOTAL_TIMESTEPS}"
    )

    print(
        f"Evaluation episodes:   {EVALUATION_EPISODES}"
    )

    print(
        f"Evaluation seeds:      "
        f"{EVALUATION_SEED} "
        f"to "
        f"{EVALUATION_SEED + EVALUATION_EPISODES - 1}"
    )

    print("============================================================")
    print()


    total_experiments = (
        len(EXPERIMENTS)
        * len(TRAINING_SEEDS)
    )

    experiment_number = 0


    # ========================================================
    # Loop over all four conditions
    # ========================================================

    for experiment in EXPERIMENTS:

        algorithm = experiment["algorithm"]

        layer_norm = experiment["layer_norm"]


        # ====================================================
        # Loop over training seeds
        # ====================================================

        for training_seed in TRAINING_SEEDS:

            experiment_number += 1

            ln_label = (
                "LayerNorm"
                if layer_norm
                else "No LayerNorm"
            )


            print()
            print()
            print("############################################################")
            print(
                f"Experiment "
                f"{experiment_number}/{total_experiments}"
            )
            print("############################################################")

            print(
                f"Algorithm:      {algorithm}"
            )

            print(
                f"LayerNorm:      {layer_norm}"
            )

            print(
                f"Training seed:  {training_seed}"
            )

            print(
                f"Configuration:  "
                f"{algorithm} + {ln_label}"
            )

            print("############################################################")
            print()


            # =================================================
            # TRAIN
            # =================================================

            model_path = train_model(
                algorithm=algorithm,
                layer_norm=layer_norm,
                training_seed=training_seed,
            )


            print()
            print(
                f"Training complete."
            )

            print(
                f"Model: {model_path}"
            )


            # =================================================
            # EVALUATE
            # =================================================

            evaluate_model(
                model_path=model_path,
                algorithm=algorithm,
                layer_norm=layer_norm,
                training_seed=training_seed,
            )


            print()
            print(
                f"Experiment "
                f"{experiment_number}/{total_experiments} "
                f"completed successfully."
            )


    # ========================================================
    # Finished
    # ========================================================

    print()
    print()
    print("============================================================")
    print("ALL EXPERIMENTS COMPLETED")
    print("============================================================")

    print(
        f"Models trained: {total_experiments}"
    )

    print(
        f"Results directory: "
        f"{PROJECT_ROOT / 'results'}"
    )

    print(
        f"Runs directory: "
        f"{PROJECT_ROOT / 'runs'}"
    )

    print("============================================================")


if __name__ == "__main__":
    main()