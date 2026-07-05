import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset
import os
from typing import List, Tuple, Dict, Any
from config import Config

class TranslationDataset(Dataset):
    def __init__(self, data, src_tokenizer, tgt_tokenizer, max_len):
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if 'translation' in item:
            src_text = item['translation']['en']
            tgt_text = item['translation']['de']
        else:
            src_text = item['en']
            tgt_text = item['de']
        
        src_encoded = self.src_tokenizer.encode(src_text).ids
        tgt_encoded = self.tgt_tokenizer.encode(tgt_text).ids
        
        # Add SOS and EOS tokens
        sos_id = self.src_tokenizer.token_to_id('[SOS]')
        eos_id = self.src_tokenizer.token_to_id('[EOS]')
        
        src_ids = [sos_id] + src_encoded[:self.max_len - 2] + [eos_id]
        tgt_ids = [sos_id] + tgt_encoded[:self.max_len - 2] + [eos_id]
        
        return {
            'src': src_ids,
            'tgt': tgt_ids
        }

def collate_fn(batch: List[Dict[str, List[int]]], pad_id: int):
    """Dynamically pad batches."""
    src_batch = [torch.tensor(item['src'], dtype=torch.long) for item in batch]
    tgt_batch = [torch.tensor(item['tgt'], dtype=torch.long) for item in batch]
    
    from torch.nn.utils.rnn import pad_sequence
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_id)
    
    return src_padded, tgt_padded

def train_tokenizer(dataset, lang: str, vocab_size: int, special_tokens: List[str]) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    
    def iterator():
        for item in dataset:
            if 'translation' in item:
                yield item['translation'][lang]
            else:
                yield item[lang]
            
    tokenizer.train_from_iterator(iterator(), trainer)
    return tokenizer

def get_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader, Tokenizer, Tokenizer]:
    # Try loading a popular translation dataset from HF
    dataset = load_dataset("bentrevett/multi30k", split="train") 
    val_dataset = load_dataset("bentrevett/multi30k", split="validation")
    
    special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
    
    os.makedirs(config.train.checkpoint_dir, exist_ok=True)
    src_tok_path = os.path.join(config.train.checkpoint_dir, "tokenizer_en.json")
    tgt_tok_path = os.path.join(config.train.checkpoint_dir, "tokenizer_de.json")
    
    if os.path.exists(src_tok_path) and os.path.exists(tgt_tok_path):
        src_tokenizer = Tokenizer.from_file(src_tok_path)
        tgt_tokenizer = Tokenizer.from_file(tgt_tok_path)
    else:
        src_tokenizer = train_tokenizer(dataset, 'en', config.model.vocab_size, special_tokens)
        tgt_tokenizer = train_tokenizer(dataset, 'de', config.model.vocab_size, special_tokens)
        src_tokenizer.save(src_tok_path)
        tgt_tokenizer.save(tgt_tok_path)
        
    train_ds = TranslationDataset(dataset, src_tokenizer, tgt_tokenizer, config.model.max_seq_len)
    val_ds = TranslationDataset(val_dataset, src_tokenizer, tgt_tokenizer, config.model.max_seq_len)
    
    pad_id = src_tokenizer.token_to_id('[PAD]')
    
    train_dl = DataLoader(
        train_ds, 
        batch_size=config.train.batch_size, 
        shuffle=True, 
        collate_fn=lambda x: collate_fn(x, pad_id)
    )
    val_dl = DataLoader(
        val_ds, 
        batch_size=config.train.batch_size, 
        shuffle=False, 
        collate_fn=lambda x: collate_fn(x, pad_id)
    )
    
    return train_dl, val_dl, src_tokenizer, tgt_tokenizer
