"""Experiments

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import os
import shutil

from lorag.grid import *
from lorag.plots import *
from lorag.rag import *
from lorag.train import *
from lorag.utils import *

supported_datasets = {
    "derm_qa": load_derm_qa_dataset,
    "medXpert": load_medXpert_dataset,
    "no_robots": load_no_robots,
}


def single_experiment(cfg, train_dataset, test_dataset, force_regen):
    output_dir = os.path.join("./out", cfg["model"]["out_dir"])
    tokenizer = get_tokenizer(cfg["model"])
    model = get_model(cfg["model"])

    model = configure_lora(model, cfg.get("lora", {}))

    cache_path = os.path.join("./data", f"{cfg['data']['dataset']}")
    trainer = get_trainer(
        model,
        tokenizer,
        train_dataset,
        test_dataset,
        cfg,
        cache_path=cache_path,
        force_regen=force_regen,
    )

    model = train_model(model, trainer, cfg)
    metrics = evaluate_model(trainer, cfg)

    plot_loss(output_dir)
    plot_eval_metrics(output_dir)

    preview_samples(model, tokenizer, test_dataset, num_samples=5, out=output_dir)

    if cfg.get("rag", None):
        if cfg["rag"].get("enabled", False):
            rag_experiment(cfg, model, tokenizer, train_dataset, test_dataset)

    return metrics


def grid_search(cfg, train_dataset, test_dataset, force_regen):
    print("\n\nRunning Grid Search...")
    search_scopes = [
        "base",
        "lora",
        # "qlora" # Requires cuda env
    ]

    for scope in search_scopes:
        print(f"\n\nRunning {scope} study...")
        out_dir = f"./out/grid/{scope}"
        if os.path.exists(out_dir):
            raise f"Warning :: path already exists: {out_dir} | Move dir or use --no-cache"

        os.makedirs(out_dir)
        os.makedirs(f"{out_dir}/plots")

        study = run_study(
            train_dataset,
            test_dataset,
            cfg,
            n_trials=5,
            output_dir=out_dir,
            force_regen=force_regen,
            scope=scope,
        )

        metrics_to_plot = [
            "eval_loss",
            "eval_bertscore_f1",
            "eval_semscore_mean",
            "eval_entail_mean",
        ]
        # TODO: Add check for lora to plot lora prams too
        params_to_group_by = ["lr", "batch_size", "max_length"]

        for metric in metrics_to_plot:
            for param in params_to_group_by:
                plot_metric_epoch_curve(study, out_dir, metric=metric, param=param)


def ds_load(cfg, subset=None):
    if cfg["data"]["dataset"] not in supported_datasets:
        raise ValueError(f"Dataset not supported: {cfg['data']['dataset']}")

    ds = supported_datasets[cfg["data"]["dataset"]](subset=subset)

    train_size = int(0.8 * len(ds))
    train_dataset = ds.select(range(train_size))
    test_dataset = ds.select(range(train_size, len(ds)))
    return train_dataset, test_dataset


def reset_cache(output_dir):
    try:
        print(f"No Cache set, Blowing away old dir: {output_dir}")
        shutil.rmtree(output_dir)
    except OSError as e:
        print(f"Error removing old model: {output_dir} : {e.filename} - {e.strerror}.")


def run(cfg, subset=None, force_regen=False, grid=False):
    d = "grid" if grid else cfg["model"]["out_dir"]
    output_dir = os.path.join("./out", d)

    if force_regen and os.path.exists(output_dir):
        reset_cache(output_dir)

    train_dataset, test_dataset = ds_load(cfg, subset=subset)

    if grid:
        res = grid_search(cfg, train_dataset, test_dataset, force_regen)
    else:
        print("\n\nRunning Single Experiment...")
        res = single_experiment(cfg, train_dataset, test_dataset, force_regen)

    return res
