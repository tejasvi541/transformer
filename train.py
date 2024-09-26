import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, NFKC
from pathlib import Path

from dataset import BilingualDataset


# To get for each example the translation in the target language
def get_all_sentences(ds, lang):
    for ex in ds:
        yield ex["translation"][lang]

# Took from huggingface
def get_or_build_tokenizer(config, ds, lang):
    # Check if tokenizer exists
    # config["tokenizer_path"] = "tokenizers/{}.json"
    tokenizer_path = Path(config["tokenizer_path"].format(lang))

    if not Path.exists(tokenizer_path):
        # Build tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        # Pre-tokenizer 
        # Tokenizer will split the text into words
        tokenizer.pre_tokenizer = Whitespace()
        # Normalizer will lowercase the text and normalize unicode characters
        tokenizer.normalizer = Lowercase()
        tokenizer.normalizer = NFKC()

        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer)

        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds(config):
    # Load dataset
    ds_raw = load_dataset("opus_books", f'{config["source_lang"]}-{config["target_lang"]}', split="train")

    # Get tokenizer and tokenized dataset
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["source_lang"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["target_lang"])

    # keep 90% for training and 10% for validation
    train_ds_size = int(0.9*len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = torch.utils.data.random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["source_lang"], config["target_lang"], config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config["source_lang"], config["target_lang"], config["seq_len"])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["source_lang"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["target_lang"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length source: {max_len_src}")
    print(f"Max length target: {max_len_tgt}")

    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt