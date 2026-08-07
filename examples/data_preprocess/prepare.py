# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess placeholder rows to parquet for verl-agent.

**NOTE**: We do NOT use Geometry3k content. The Hub dataset is only a size/modality
scaffold; coding problems come from env.swebench at train time.
Offline train nodes must not depend on Hugging Face Hub.
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


def _offline_enabled() -> bool:
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(key, "").lower() in ("1", "true", "yes"):
            return True
    return False


def _synthetic_split(n: int, mode: str) -> datasets.Dataset:
    """Rows sized like the trainer batch; content is unused by SWE envs."""
    prompt = "<image>" if mode == "visual" else ""
    rows = []
    for i in range(n):
        row = {
            "data_source": mode,
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "agent",
            "extra_info": {"split": "synthetic", "index": i},
        }
        if mode == "visual":
            row["images"] = None
        rows.append(row)
    return datasets.Dataset.from_list(rows)


def _load_or_synthetic(train_n: int, val_n: int, mode: str):
    data_source = "hiyouga/geometry3k"
    if _offline_enabled():
        print(f"[prepare] offline mode on — skip Hub ({data_source}), use synthetic placeholders")
        return _synthetic_split(train_n, mode), _synthetic_split(val_n, mode)

    try:
        dataset = datasets.load_dataset(data_source)
        train_raw = dataset["train"].select(range(train_n))
        test_raw = dataset["test"].select(range(val_n))
    except Exception as e:
        print(f"[prepare] failed to load {data_source}: {e}")
        print("[prepare] using local synthetic placeholder rows instead")
        return _synthetic_split(train_n, mode), _synthetic_split(val_n, mode)

    instruction_following = {"visual": "<image>", "text": ""}

    def make_map_fn(split):
        def process_fn(example, idx):
            example.pop("problem", None)
            prompt = instruction_following[mode]
            images = example.pop("images", None)
            if mode == "visual":
                return {
                    "data_source": mode,
                    "prompt": [{"role": "user", "content": prompt}],
                    "images": images,
                    "ability": "agent",
                    "extra_info": {"split": split, "index": idx},
                }
            return {
                "data_source": mode,
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "agent",
                "extra_info": {"split": split, "index": idx},
            }

        return process_fn

    train_dataset = train_raw.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_raw.map(function=make_map_fn("test"), with_indices=True, num_proc=8)
    return train_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="visual", choices=["visual", "text"])
    parser.add_argument("--local_dir", default="~/data/verl-agent/")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--train_data_size", default=256, type=int)
    parser.add_argument("--val_data_size", default=256, type=int)

    args = parser.parse_args()
    print(f"processing data for mode: {args.mode}")
    local_dir = os.path.join(os.path.expanduser(args.local_dir), args.mode)

    train_dataset, test_dataset = _load_or_synthetic(
        args.train_data_size, args.val_data_size, args.mode
    )

    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
