"""LoRAG

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import argparse
import os

from lorag.experiment import *
from lorag.size_params_calculator import *


def main():
    parser = argparse.ArgumentParser(description="LoRAG Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/bioT5.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force regenerate tokenized dataset and ignore cached version",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Drives a gridsearch rather than single model run",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs("./out", exist_ok=True)
    os.makedirs("./data", exist_ok=True)

    run(
        config,
        subset=config["data"]["subset"],
        force_regen=args.no_cache,
        grid=args.grid_search,
    )


if __name__ == "__main__":
    main()
