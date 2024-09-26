import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.long)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.long)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.long)
        self.unk_token = torch.tensor([tokenizer_src.token_to_id("[UNK]")], dtype=torch.long)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_txt = src_target_pair["translation"][self.src_lang]
        tgt_txt = src_target_pair["translation"][self.tgt_lang]

        src_enc = self.tokenizer_src.encode(src_txt).ids
        tgt_enc = self.tokenizer_tgt.encode(tgt_txt).ids

        # Add special tokens
        # [SOS] token at the beginning of the sequence
        # [EOS] token at the end of the sequence
        # [PAD] token to pad the sequences
        # [UNK] token for unknown tokens

        enc_num_padding = self.seq_len - len(src_enc) - len(tgt_enc) - 2
        dec_num_padding = self.seq_len - len(tgt_enc) - len(src_enc) - 1

        if enc_num_padding < 0 :
            raise ValueError("Sequence too long")
        
        # Making input sequences of the same length & adding special tokens
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(src_enc, dtype=torch.long),
            self.eos_token,
            torch.tensor([self.pad_token]*enc_num_padding, dtype=torch.long)
        ])
        # Making target sequences of the same length & adding special tokens
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(tgt_enc, dtype=torch.long),
            torch.tensor([self.pad_token]*dec_num_padding, dtype=torch.long)
        ])

        # Making label sequences of the same length & adding special tokens
        label = torch.cat([
            torch.tensor(tgt_enc, dtype=torch.long),
            self.eos_token,
            torch.tensor([self.pad_token]*dec_num_padding, dtype=torch.long)
        ])

        # Check that the sequences have the correct length
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # Return the input and target sequences
        return {
            "encoder_input": encoder_input, # input sequence
            "decoder_input": decoder_input, # target sequence
            "encoder_target": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_len) # label sequence
            "decoder_target": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)), # (1,1,seq_len) & (seq_len, seq_len) # label sequence
            "label": label, # label sequence (seq_len)
            "src_txt": src_txt,
            "tgt_txt": tgt_txt
        }

def casual_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask==0