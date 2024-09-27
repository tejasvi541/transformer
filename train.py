import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, NFKC
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm as tdqm
from config import get_config, get_weights_file_path, latest_weights_file_path
import warnings
from dataset import BilingualDataset
from transformer import build_transformer


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

def get_model(config, vocab_src_size, vocab_tgt_size):

    # Define model
    model = build_transformer(vocab_src_size, vocab_tgt_size, config["seq_len"], config["seq_len"], config["d_model"])
    return model

def train_model(config):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    # Define model
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    # writer = SummaryWriter(config["experiment_name"])
    writer = SummaryWriter(config["experiment_name"])

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if(config["preload"]):
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Loading model from {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer"])
        global_step = state["global_step"]

    # Loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1) 

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tdqm(train_dataloader, desc=f"Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch["encoder_input"].to(device) # [batch_size, seq_len]
            decoder_input = batch["decoder_input"].to(device) # [batch_size, seq_len]
            encoder_mask = batch["encoder_mask"].to(device) # [batch_size, 1, 1, seq_len]
            decoder_mask = batch["decoder_mask"].to(device) # [batch_size, 1, seq_len, seq_len]

            # Forward pass
            encoder_output = model.encode(encoder_input, encoder_mask) # [batch_size, seq_len, d_model]
            decoder_output = model.decode(encoder_output, decoder_input, encoder_mask, decoder_mask) # [batch_size, seq_len, d_model]
            logits = model.project(decoder_output)

            label = batch["label"].to(device) # [batch_size, seq_len]

            # Compute loss
            # logits: [batch_size*seq_len, vocab_size]
            loss = loss_fn(logits.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log
            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.flush()
            # Backward pass
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Save model
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        print(f"Saving model to {model_filename}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step
        }, model_filename)
    
if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)