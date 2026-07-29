# Copyright 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-turn SFT dataset that supports training on conversation data with multiple turns
"""

from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs


def _normalize_messages(messages: Any) -> List[Dict[str, str]]:
    """Parquet may store messages as ndarray / nested Series; coerce to list[dict]."""
    while isinstance(messages, (pd.Series, np.ndarray)) and getattr(messages, "ndim", 1) == 0:
        messages = messages.item()
    if isinstance(messages, np.ndarray):
        messages = messages.tolist()
    if isinstance(messages, pd.Series):
        messages = messages.tolist()
    out: List[Dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, dict):
            out.append({"role": str(msg["role"]), "content": str(msg["content"])})
        else:
            # e.g. numpy void / Mapping
            out.append({"role": str(msg["role"]), "content": str(msg["content"])})
    return out


def _chat_token_ids(tokenizer: PreTrainedTokenizer, messages: List[Dict[str, str]]) -> torch.Tensor:
    """Return 1D LongTensor of token ids (tokenizer-version safe).

    Newer tokenizers may ignore ``return_tensors='pt'`` and return an Encoding /
    BatchEncoding; older ones return a Tensor. Always normalize to a 1D tensor.
    """
    if not messages:
        return torch.zeros(0, dtype=torch.long)

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
    )

    # Common paths: List[int], BatchEncoding / dict with input_ids, Encoding
    if isinstance(encoded, torch.Tensor):
        ids = encoded.view(-1).tolist()
    elif isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
        raw = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if isinstance(raw, torch.Tensor):
            ids = raw.view(-1).tolist()
        elif isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], (list, tuple)):
            ids = list(raw[0])
        else:
            ids = list(raw)
    elif isinstance(encoded, (list, tuple)):
        if encoded and isinstance(encoded[0], (list, tuple)):
            ids = list(encoded[0])
        else:
            ids = list(encoded)
    else:
        # tokenizers.Encoding
        ids = list(getattr(encoded, "ids", encoded))

    return torch.tensor(ids, dtype=torch.long)


class MultiTurnSFTDataset(Dataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config=None):
        # Set defaults and extract parameters from config if provided
        config = config or {}
        self.truncation = config.get("truncation", "error")
        self.max_length = config.get("max_length", 1024)
        # Get messages_key from the new multiturn config structure
        multiturn_config = config.get("multiturn", {})
        self.messages_key = multiturn_config.get("messages_key", "messages")

        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self._download()
        self._read_files_and_process()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        # Extract messages list from dataframe
        self.messages = self.dataframe[self.messages_key].apply(series_to_item).tolist()

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        messages = _normalize_messages(self.messages[item])

        input_ids = _chat_token_ids(tokenizer, messages)
        attention_mask = torch.ones_like(input_ids)
        loss_mask = torch.zeros_like(input_ids, dtype=torch.long)

        # Process each message to find assistant responses
        for i, msg in enumerate(messages):
            prefix_tokens = _chat_token_ids(tokenizer, messages[: i + 1])
            prev_tokens = _chat_token_ids(tokenizer, messages[:i]) if i > 0 else None

            start_pos = int(prev_tokens.shape[0]) if prev_tokens is not None else 0
            end_pos = int(prefix_tokens.shape[0])
            end_pos = min(end_pos, int(input_ids.shape[0]))
            start_pos = min(start_pos, end_pos)

            if msg["role"] == "assistant" and end_pos > start_pos:
                loss_mask[start_pos:end_pos] = 1

        # Handle sequence length
        sequence_length = int(input_ids.shape[0])
        if sequence_length < self.max_length:
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            pad_len = self.max_length - sequence_length
            padded_input_ids = torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)
            padded_attention_mask = torch.zeros(pad_len, dtype=attention_mask.dtype)
            padded_loss_mask = torch.zeros(pad_len, dtype=loss_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            elif self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise ValueError(f"Unknown truncation method {self.truncation}")

        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        position_ids = position_ids * attention_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
