"""Grid Search

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import copy

import optuna
import optuna.visualization as visualization
import transformers

from lorag.plots import *
from lorag.train import *


class MetricLoggerCallback(transformers.TrainerCallback):
    def __init__(self):
        self.train_logs = []
        self.eval_logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        entry = {"epoch": state.epoch}

        # Save all the metrics from the compute metrics fn for that epoch
        if state.is_local_process_zero:
            if any(k.startswith("eval_") for k in logs):
                entry.update({k: v for k, v in logs.items() if k.startswith("eval_")})
                self.eval_logs.append(entry)

            if "loss" in logs:
                entry.update({"loss": logs["loss"]})
                self.train_logs.append(entry)


def objective(
    trial, train_dataset, eval_dataset, 
    base_cfg, cache_path=None, force_regen=False,
    mode="base"
):

    lr = trial.suggest_float("lr", 1e-7, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    max_length = trial.suggest_categorical("max_length", [128, 256, 384, 512])

    if mode in ("lora", "qlora"):
        lora_r = trial.suggest_int("lora_r", 4, 64)
        lora_alpha = trial.suggest_int("lora_alpha", 8, 64)
        lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.3)

    cfg = copy.deepcopy(base_cfg)
    cfg["model"]["out_dir"] = (
        f"grid/{mode}/" + base_cfg["model"]["out_dir"] + f"_{lr}_{batch_size}_{max_length}"
    )
    cfg["model"]["lr"] = lr
    cfg["model"]["batch_size"] = batch_size
    cfg["model"]["max_len"] = max_length

    if mode == "base":
        cfg["model"]["quantization"] = False
        cfg["lora"]["enabled"] = False
    else:
        # lora or qlora
        cfg["lora"].update(
            {"enabled": True, "r": lora_r, "alpha": lora_alpha, "dropout": lora_dropout}
        )
        cfg["model"]["quantization"] = mode == "qlora"

    tokenizer = get_tokenizer(cfg["model"])
    model = get_model(cfg["model"])
    model = configure_lora(model, cfg["lora"])

    metrics_logger = MetricLoggerCallback()
    trainer = get_trainer(
        model, tokenizer, train_dataset, eval_dataset, cfg, cache_path, force_regen
    )
    # Add callback class to record metrics throughout
    trainer.add_callback(metrics_logger)

    trainer.train()
    metrics = trainer.evaluate()

    # save metrics logs to trial
    trial.set_user_attr("train_logs", metrics_logger.train_logs)
    trial.set_user_attr("eval_logs", metrics_logger.eval_logs)

    # reurn max or min value if theres an err caluclating eval loss
    return metrics.get("eval_loss", 999.0), metrics.get("eval_entail_mean", 0.0)


def run_study(
    train_dataset,
    eval_dataset,
    cfg,
    n_trials=5,
    output_dir="./out/grid",
    force_regen=False,
    scope="base"
):
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    # Single Objective Optimization
    # study = optuna.create_study(direction="minimize")

    # Multi Objective Optimization
    study = optuna.create_study(
        # Minimize eval loss, maximize entailment
        directions=["minimize", "maximize"], 
        pruner=optuna.pruners.HyperbandPruner(
            # TODO: Play with these values as they are default
            min_resource=1, max_resource=100, reduction_factor=3
        )
    )
    study.optimize(
        lambda trial: objective(
            trial,
            train_dataset,
            eval_dataset,
            cfg,
            cache_path=output_dir,
            force_regen=force_regen,
            mode=scope
        ),
        n_trials=n_trials,
    )

    # print("Best Trial Params:")
    # print(study.best_trial.params)

    pareto_front_trials = study.best_trials

    print(f"number of trials on the Pareto front: {len(pareto_front_trials)}")
    for i, trial in enumerate(pareto_front_trials):
        print(f"trial {i}: Value={trial.values}, Params={trial.params}")

    def optuna_plots(study, target):
        # target=lambda t: t.values[0], target_name="First Objective"
        for name, func in {
            "opt_history": visualization.plot_optimization_history,
            "param_importance": visualization.plot_param_importances,
            "parallel_coord": visualization.plot_parallel_coordinate,
            "contour": visualization.plot_contour,
            "slice": visualization.plot_slice,
            "edf": visualization.plot_edf,
        }.items():
            func(study).write_image(f"{plots_dir}/{name}.png")

    return study
