import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        # Initialize the dataset object.
        # Inputs:
        # - ds: The dataset containing source and target translations.
        # - tokenizer_src: Tokenizer for the source language.
        # - tokenizer_tgt: Tokenizer for the target language.
        # - src_lang: Source language code (e.g., "en" for English).
        # - tgt_lang: Target language code (e.g., "fr" for French).
        # - seq_len: The maximum sequence length for the input/output.
        
        super().__init__()
        self.seq_len = seq_len  # Max sequence length for padding/truncation.

        self.ds = ds  # Dataset containing pairs of translations.
        self.tokenizer_src = tokenizer_src  # Tokenizer for the source language.
        self.tokenizer_tgt = tokenizer_tgt  # Tokenizer for the target language.
        self.src_lang = src_lang  # Source language key.
        self.tgt_lang = tgt_lang  # Target language key.

        # Special tokens: start-of-sequence (SOS), end-of-sequence (EOS), padding (PAD).
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        # Returns the size of the dataset.
        return len(self.ds)

    def __getitem__(self, idx):
        # Get a specific item from the dataset at the given index.
        # Inputs:
        # - idx: The index of the dataset element to retrieve.
        # Outputs:
        # - A dictionary containing the encoder/decoder inputs, masks, and the target label.
        
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]  # Source language text.
        tgt_text = src_target_pair['translation'][self.tgt_lang]  # Target language text.

        # Tokenize the source and target texts.
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids  # Token IDs for source text.
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids  # Token IDs for target text.

        # Add padding and special tokens to the sequences.
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # Padding for encoder (+ SOS, EOS).
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # Padding for decoder (+ SOS, EOS added later).

        # Ensure sentences are not too long for the seq_len.
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Construct the encoder input: [SOS] + tokenized source + [EOS] + padding.
        encoder_input = torch.cat(
            [
                self.sos_token,  # Start-of-sequence token.
                torch.tensor(enc_input_tokens, dtype=torch.int64),  # Source tokens.
                self.eos_token,  # End-of-sequence token.
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),  # Padding tokens.
            ],
            dim=0,
        )  # Shape: (seq_len)

        # Construct the decoder input: [SOS] + tokenized target + padding (no EOS here).
        decoder_input = torch.cat(
            [
                self.sos_token,  # Start-of-sequence token.
                torch.tensor(dec_input_tokens, dtype=torch.int64),  # Target tokens.
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),  # Padding tokens.
            ],
            dim=0,
        )  # Shape: (seq_len)

        # Construct the label: tokenized target + [EOS] + padding.
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),  # Target tokens.
                self.eos_token,  # End-of-sequence token.
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),  # Padding tokens.
            ],
            dim=0,
        )  # Shape: (seq_len)

        # Ensure that the encoder_input, decoder_input, and label have the correct length.
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # Return the dictionary containing model inputs and corresponding masks.
        return {
            "encoder_input": encoder_input,  # Shape: (seq_len)
            "decoder_input": decoder_input,  # Shape: (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # Shape: (1, 1, seq_len), mask for the encoder.
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # Shape: (1, seq_len) & (1, seq_len, seq_len), decoder mask with causal masking.
            "label": label,  # Shape: (seq_len), target label.
            "src_text": src_text,  # Original source text.
            "tgt_text": tgt_text,  # Original target text.
        }

# Causal mask for the decoder to prevent attending to future tokens.
def causal_mask(size):
    # Creates a causal mask of shape (1, size, size) where each token can only attend to previous tokens or itself.
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)  # Upper triangular matrix with zeros on and below the diagonal.
    return mask == 0  # Invert to make it a mask (1 where allowed, 0 otherwise).
