#!/usr/bin/env python3
"""User-run raw preprocessing; freeze the supplied five spatial folds, then fit inner-only transforms."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
_PROJECT_DIRECTORY = next(parent for parent in Path(__file__).resolve().parents
                          if (parent / "microcture/layout.py").is_file())
sys.path.insert(0, str(_PROJECT_DIRECTORY))
from microcture.layout import PROJECT_ROOT as ROOT, bootstrap, resolve
bootstrap()


def main():
    from microcture.tasks.task1 import data
    from microcture.tasks.task1.training import write_json
    from microcture.tasks.task1.data import prepare_controlled_fold
    from microcture.tasks.task1.data import _check_frozen_partition
    from microcture.tasks.task1.protocol import fold_record, json_signature, load_protocol, project_path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=resolve(ROOT, "configs/task1_local.json"))
    parser.add_argument("--data-root", type=Path, default=ROOT, help="Project/input root for configs/shared/data.json; matrices are under data/raw/micro_c/")
    parser.add_argument("--cache", type=Path, help="Fresh data/task1/cache/<new_version>; existing or retained evidence is refused")
    parser.add_argument("--data-config", type=Path, help="Alternative data JSON with raw matrix and workbook paths")
    parser.add_argument("--annotations", type=Path, help="Alternative structure CSV, with the fixed IDs and interval conventions")
    parser.add_argument("--reference-folds", type=Path, help="Reference frozen fold directory; default packaged five folds")
    args = parser.parse_args()
    config = load_protocol(project_path(ROOT, args.config))
    settings = dict(config["data"])
    if args.data_config is not None:
        settings["data_config"] = str(args.data_config)
    if args.annotations is not None:
        settings["annotations"] = str(args.annotations)
    destination = data.guard_cache_output(ROOT, args.cache or config["cache_dir"])
    reference = project_path(ROOT, args.reference_folds or config["frozen_run_dir"])
    reference_records = [fold_record(reference, number)[0] for number in config["folds"]]
    dataset = data.prepare_dataset(settings, ROOT, destination, data_root=project_path(ROOT, args.data_root))
    for record in reference_records:
        # The ID lists and spatial partitions are fixed before image fitting.
        _check_frozen_partition(dataset, record)
        training_ids, validation_ids = record["train_ids"], record["validation_ids"]
        _, _, signal = data.prepare_fold(dataset, training_ids, validation_ids, dataset["config"])
        frozen = {**record, "preprocessing": signal}
        _, _, target = prepare_controlled_fold(dataset, frozen)
        output = {"frozen_fold": frozen, "preprocessing": target,
                  "split_provenance": {"reference_directory": str(reference), "reference_fold_sha256": json_signature(record),
                                       "id_lists_reused": True, "signal_fit_scope": "inner training only"}}
        output["signature"] = json_signature(output)
        write_json(destination / "formal/folds" / f"fold_{record['fold']}.json", output)
    print(json.dumps({"status": "prepared", "cache": str(destination), "dataset_signature": dataset["signature"],
                      "frozen_folds": str(destination / "formal"), "model_training_performed": False,
                      "next": "python -B run.py task1 --cache CACHE --folds-dir CACHE/formal check"}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
