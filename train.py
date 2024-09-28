from transformer import build_transformer  # Import function to build the transformer model
from dataset import BilingualDataset, causal_mask  # BilingualDataset handles source-target pairs, causal_mask creates the attention mask
from config import get_config, get_weights_file_path, latest_weights_file_path  # Import configuration and model weight utilities

import torchtext.datasets as datasets  # Torchtext datasets for language tasks
import torch
import torch.nn as nn  # Import PyTorch's neural network module
from torch.utils.data import Dataset, DataLoader, random_split  # Dataset management utilities, including DataLoader and random splitting
from torch.optim.lr_scheduler import LambdaLR  # Import learning rate scheduler

import warnings
from tqdm import tqdm  # Progress bar for loops
import os
from pathlib import Path  # Path manipulation for filesystem operations

# Huggingface datasets and tokenizers
from datasets import load_dataset  # Import for loading datasets from Hugging Face
from tokenizers import Tokenizer  # Tokenizer for processing text
from tokenizers.models import WordLevel  # Word-level tokenization model
from tokenizers.trainers import WordLevelTrainer  # Trainer for the word-level tokenizer
from tokenizers.pre_tokenizers import Whitespace  # Pre-tokenizer for splitting text into words using whitespace

import torchmetrics  # Metric library for measuring performance like BLEU, WER
from torch.utils.tensorboard import SummaryWriter  # TensorBoard writer for logging metrics

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Greedy decoding function for generating predictions.
    
    Parameters:
    - model: Transformer model used for prediction.
    - source: Source sentence tensor. Shape: (1, seq_len).
    - source_mask: Mask tensor for source sentence. Shape: (1, 1, 1, seq_len).
    - tokenizer_src: Source language tokenizer.
    - tokenizer_tgt: Target language tokenizer.
    - max_len: Maximum length of target sequence to generate.
    - device: Device to run computation on (CPU/GPU).
    """
    
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')  # [SOS] token index
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')  # [EOS] token index

    # Encode the source sentence once. Shape of encoder_output: (1, seq_len, d_model)
    encoder_output = model.encode(source, source_mask)

    # Initialize decoder input with [SOS] token. Shape: (1, 1)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break  # Stop if maximum target length is reached

        # Create causal mask for decoder input. Shape: (1, 1, seq_len)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Decode the next token step by step
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Project decoder output to vocabulary size and get probabilities. Shape: (1, vocab_size)
        prob = model.project(out[:, -1])

        # Select the token with the highest probability
        _, next_word = torch.max(prob, dim=1)

        # Concatenate the new word to the decoder input
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        # Stop if [EOS] token is generated
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)  # Return the decoded sequence

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    """
    Perform validation on a given validation dataset and log results.
    
    Parameters:
    - model: Transformer model used for prediction.
    - validation_ds: Validation dataset.
    - tokenizer_src: Source language tokenizer.
    - tokenizer_tgt: Target language tokenizer.
    - max_len: Maximum length of target sequence to generate.
    - device: Device to run computation on (CPU/GPU).
    - print_msg: Printing function (used for logging).
    - global_step: Current training step, used for logging in TensorBoard.
    - writer: TensorBoard writer object.
    - num_examples: Number of examples to display in validation log.
    """
    
    model.eval()  # Set model to evaluation mode
    count = 0

    source_texts = []  # Store source sentences
    expected = []  # Store ground-truth target sentences
    predicted = []  # Store model-generated sentences

    try:
        with os.popen('stty size', 'r') as console:  # Get console width for formatting logs
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80  # Default width if unable to get console size

    with torch.no_grad():  # Disable gradient calculation for validation
        for batch in validation_ds:
            count += 1

            encoder_input = batch["encoder_input"].to(device)  # (1, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (1, 1, 1, seq_len)

            # Ensure batch size is 1 during validation (for simplicity in logging)
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # Perform greedy decoding to generate predictions
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            # Extract source, target, and predicted texts for logging
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Log source, target, and predicted sentences
            print_msg('-' * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-' * console_width)
                break

    if writer:
        # Log validation metrics to TensorBoard
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

def get_all_sentences(ds, lang):
    """
    Generator function to yield all sentences from a dataset in a given language.
    Parameters:
    - ds: Dataset with translation pairs.
    - lang: Language to extract from translation pairs.
    """
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    """
    Create or load a tokenizer for a given language based on the dataset.
    
    Parameters:
    - config: Configuration containing tokenizer file paths.
    - ds: Dataset containing translation pairs.
    - lang: Language for which the tokenizer will be built or loaded.
    """
    
    tokenizer_path = Path(config['tokenizer_file'].format(lang))  # Define tokenizer path
    if not Path.exists(tokenizer_path):  # If tokenizer file doesn't exist, create a new tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))  # Save the newly trained tokenizer
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))  # Load existing tokenizer from file
    return tokenizer

def get_ds(config):
    """
    Load and prepare dataset for training and validation.
    
    Parameters:
    - config: Configuration containing dataset parameters.
    """
    
    # Load dataset from Hugging Face. 'train' split is used here.
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers for both source and target languages
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Split dataset into 90% training and 10% validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # Wrap raw datasets into BilingualDataset class to handle tokenization and batching
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Return processed datasets and tokenizers
    return train_ds, val_ds, tokenizer_src, tokenizer_tgt

def get_dataloader(ds, config):
    """
    Create a DataLoader from a dataset.
    
    Parameters:
    - ds: Dataset for which DataLoader is created.
    - config: Configuration containing DataLoader parameters.
    """
    
    # Wrap dataset into a PyTorch DataLoader with specified batch size and other options
    return DataLoader(ds, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])


def get_model(config, vocab_src_len, vocab_tgt_len):
    """
    Build and return a transformer model.

    Parameters:
    - config: Configuration dictionary containing model parameters such as sequence length, d_model (embedding dimension).
    - vocab_src_len: Size of the source vocabulary (number of unique tokens in source language).
    - vocab_tgt_len: Size of the target vocabulary (number of unique tokens in target language).

    Returns:
    - model: A transformer model built with the specified configuration.
    """
    # Build transformer model with source/target vocab size, sequence length, and embedding dimension (d_model)
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def train_model(config):
    """
    Main training function to train the transformer model.
    
    Parameters:
    - config: Configuration dictionary containing hyperparameters and paths for training (e.g., learning rate, num_epochs).
    """
    
    # Determine the computing device (GPU, MPS, or CPU)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    if device == 'cuda':
        # Print GPU information if CUDA is available
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif device == 'mps':
        print(f"Device name: <mps>")
    else:
        # Inform the user about potential GPU support and give tips for enabling it
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    
    device = torch.device(device)

    # Ensure the weights folder exists for saving model checkpoints
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    # Prepare the training and validation datasets and tokenizers
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    # Build the transformer model based on source/target vocab sizes and configuration settings
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # TensorBoard for tracking metrics
    writer = SummaryWriter(config['experiment_name'])

    # Define the optimizer (Adam) for updating model weights during training
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # Preloading model weights, if specified in config
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    # Determine whether to load the latest weights or a specific checkpoint
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        # Load model and optimizer states from a checkpoint
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    # Define the loss function (CrossEntropyLoss) with label smoothing and ignoring the padding token
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # Start training loop
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()  # Clear GPU cache to prevent memory overflow
        model.train()  # Set model to training mode

        # Progress bar for training each epoch
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        
        for batch in batch_iterator:
            # Move input and masks to the correct device (GPU/CPU)
            encoder_input = batch['encoder_input'].to(device)  # (B, seq_len) -> Input for encoder
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len) -> Input for decoder
            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len) -> Mask for encoder
            decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len) -> Mask for decoder

            # Pass data through the encoder and decoder of the transformer model
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model) -> Encoder output
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  # (B, seq_len, d_model) -> Decoder output
            proj_output = model.project(decoder_output)  # (B, seq_len, vocab_size) -> Project to vocab size

            # The target labels for the cross-entropy loss
            label = batch['label'].to(device)  # (B, seq_len)

            # Calculate loss between model output and the actual labels
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))  # Reshape for loss calculation
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})  # Display loss in progress bar

            # Log loss to TensorBoard
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagation to compute gradients
            loss.backward()

            # Update the model weights using the optimizer
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # Clear gradients for the next step

            global_step += 1  # Increment the global step

        # Validate the model after each epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save model checkpoints at the end of each epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,  # Current epoch
            'model_state_dict': model.state_dict(),  # Model weights
            'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
            'global_step': global_step  # Track the number of iterations
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")  # Ignore warnings to keep the console clean
    config = get_config()  # Load configuration settings
    train_model(config)  # Start the training process
