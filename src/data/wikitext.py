import os
import zipfile
import urllib
import numpy as np
from transformers import AutoTokenizer
import datasets
from tqdm import tqdm


def get_wikitext_data(datasets_base_dir, num_proc=1):
    """Inspired from https://github.com/tysam-code/hlb-gpt"""
    WIKITEXT_DATA_PATH = os.path.join(datasets_base_dir, "wikitext/")
    if not os.path.exists(os.path.join(WIKITEXT_DATA_PATH, "train.bin")):
        os.makedirs(WIKITEXT_DATA_PATH, exist_ok=True)
        
        split_dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        
        hf_tknzr = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        def process(example):
            ids = hf_tknzr.encode(
                text=example["text"],
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )
            out = {"ids": ids, "len": len(ids)}
            return out
    
        # tokenize the dataset
        tokenized = split_dataset.map(
            process,
            remove_columns=["text"],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"])
            filename = os.path.join(WIKITEXT_DATA_PATH, f"{split}.bin")
            print(filename)
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = min(1024, len(dset))

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    return {
        "train": os.path.join(WIKITEXT_DATA_PATH, "train.bin"),
        "val": os.path.join(WIKITEXT_DATA_PATH, "validation.bin"),
    }
