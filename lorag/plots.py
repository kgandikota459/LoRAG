"""Plots

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from lorag.utils import load_trainer_logs

COLOR_MAP = "Dark2"
sns.set_style("darkgrid")


def plot_loss(output_dir):
    logs = load_trainer_logs(output_dir)
    if logs is None:
        return

    train_steps = []
    train_loss = []
    eval_steps = []
    eval_loss = []

    for entry in logs:
        if "loss" in entry and "epoch" in entry:
            train_steps.append(entry["epoch"])
            train_loss.append(entry["loss"])

        if "eval_loss" in entry:
            eval_steps.append(entry["epoch"])
            eval_loss.append(entry["eval_loss"])

    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    cmap = cm.get_cmap(COLOR_MAP)
    c_train = cmap(0)
    c_eval = cmap(2)

    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_loss, label="Train Loss", color=c_train, marker="o")
    plt.plot(eval_steps, eval_loss, label="Eval Loss", color=c_eval, marker="x")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Eval Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "plots/loss_curve.png"))
    plt.close()
    print("Saved:", os.path.join(output_dir, "plots/loss_curve.png"))


def plot_eval_metrics(output_dir):
    logs = load_trainer_logs(output_dir)

    if logs is None:
        return

    metrics_to_plot = {
        "BERTScore F1": "eval_bertscore_f1",
        "Semantic Similarity": "eval_semscore_mean",
        "NLI Entailment": "eval_entail_mean",
    }

    epochs = []
    metric_values = {name: [] for name in metrics_to_plot}

    for entry in logs:
        if "eval_loss" in entry:
            epochs.append(entry["epoch"])
            for readable, key in metrics_to_plot.items():
                metric_values[readable].append(entry.get(key, None))

    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    cmap = cm.get_cmap(COLOR_MAP)
    colors = [cmap(i) for i in range(len(metrics_to_plot))]

    plt.figure(figsize=(12, 8))
    for (name, vals), color in zip(metric_values.items(), colors):
        plt.plot(epochs, vals, marker="o", label=name, linewidth=2, color=color)

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Evaluation Metrics Dashboard")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "plots/eval_metrics.png"))
    plt.close()
    print("Saved:", os.path.join(output_dir, "plots/eval_metrics.png"))


def plot_metric_epoch_curve(study, output_dir, metric="eval_loss", param="lr"):
    """Create a multi line plot for each epoch by metric"""

    all_rows = []

    for trial in study.trials:
        eval_logs = trial.user_attrs.get("eval_logs", [])
        if len(eval_logs) == 0:
            continue

        trial_param_value = trial.params.get(param, None)
        if trial_param_value is None:
            continue

        for row in eval_logs:
            new_row = {
                "trial": trial.number,
                "epoch": row["epoch"],
                "metric": row.get(metric, None),
                param: trial_param_value,
            }
            all_rows.append(new_row)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print(f"logs not found for metric={metric}!")
        return

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df, x="epoch", y="metric", hue=param, palette=COLOR_MAP, marker="o"
    )

    plt.title(f"{metric} vs Epoch\ncolored by {param}")
    plt.ylabel(metric)
    plt.savefig(f"{output_dir}/{metric}_vs_epoch_colored_by_{param}.png")
    plt.close()

    print(f"Saved: {output_dir}/{metric}_vs_epoch_colored_by_{param}.png")
