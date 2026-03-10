"""
Dataset loader for preprocessed mC4 data from mrt5/lm_datasets.

This module decodes the ByT5 byte-level tokens back to text and 
re-tokenizes them with BERT tokenizer for MLM training.
"""

import json
import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import BertTokenizer
from typing import Optional, Dict, List
import random


def decode_byt5_tokens(input_ids: List[int]) -> str:
    """
    Decode ByT5 byte tokens back to text.
    
    ByT5 uses byte tokens with offset of 3:
    - 0 = pad
    - 1 = eos  
    - 2 = unk
    - 3+ = actual UTF-8 bytes
    
    We also need to handle sentinel tokens (259+) which are span corruption masks.
    """
    bytes_list = []
    for token_id in input_ids:
        if token_id < 3:
            # Skip special tokens (pad, eos, unk)
            continue
        elif token_id >= 259:
            # Sentinel tokens (span corruption masks) - replace with space
            bytes_list.append(32)  # space
        else:
            # Regular byte token
            bytes_list.append(token_id - 3)
    
    try:
        text = bytes(bytes_list).decode('utf-8', errors='replace')
        return text
    except:
        return ""


class MC4Dataset(Dataset):
    """
    Dataset for loading preprocessed mC4 data and converting to BERT format.
    
    Loads the full dataset into memory. Use MC4IterableDataset for large files.
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer: BertTokenizer,
        max_length: int = 512,
        mlm_probability: float = 0.15,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.examples = []
        
        print(f"Loading dataset from {file_path}...")
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                    
                data = json.loads(line)
                input_ids = data['input_ids']
                if isinstance(input_ids[0], list):
                    input_ids = input_ids[0]
                
                # Decode ByT5 tokens to text
                text = decode_byt5_tokens(input_ids)
                if len(text.strip()) > 10:  # Skip very short texts
                    self.examples.append(text)
                
                if (i + 1) % 10000 == 0:
                    print(f"  Loaded {i + 1} examples...")
        
        print(f"Loaded {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        text = self.examples[idx]
        
        # Tokenize with BERT tokenizer
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Create MLM labels
        labels = input_ids.clone()
        
        # Create probability matrix for masking
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # Don't mask special tokens
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            input_ids.tolist(), already_has_special_tokens=True
        )
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        
        # Don't mask padding
        probability_matrix.masked_fill_(attention_mask == 0, value=0.0)
        
        # Create masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens
        
        # 80% MASK, 10% random, 10% original
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
        
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


class MC4IterableDataset(IterableDataset):
    """
    Iterable dataset for streaming large mC4 files without loading into memory.
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer: BertTokenizer,
        max_length: int = 512,
        mlm_probability: float = 0.15,
        shuffle_buffer_size: int = 10000,
    ):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.shuffle_buffer_size = shuffle_buffer_size
    
    def __iter__(self):
        buffer = []
        
        with open(self.file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                input_ids = data['input_ids']
                if isinstance(input_ids[0], list):
                    input_ids = input_ids[0]
                
                # Decode ByT5 tokens to text
                text = decode_byt5_tokens(input_ids)
                if len(text.strip()) <= 10:
                    continue
                
                buffer.append(text)
                
                # Shuffle buffer when full
                if len(buffer) >= self.shuffle_buffer_size:
                    random.shuffle(buffer)
                    while len(buffer) > self.shuffle_buffer_size // 2:
                        yield self._process_text(buffer.pop())
        
        # Process remaining items
        random.shuffle(buffer)
        for text in buffer:
            yield self._process_text(text)
    
    def _process_text(self, text: str) -> Dict[str, torch.Tensor]:
        # Tokenize with BERT tokenizer
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Create MLM labels
        labels = input_ids.clone()
        
        # Create probability matrix for masking
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # Don't mask special tokens
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            input_ids.tolist(), already_has_special_tokens=True
        )
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        
        # Don't mask padding
        probability_matrix.masked_fill_(attention_mask == 0, value=0.0)
        
        # Create masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        
        # 80% MASK, 10% random, 10% original
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
        
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


def load_mc4_dataset(
    split: str = "train",
    tokenizer: Optional[BertTokenizer] = None,
    max_length: int = 512,
    mlm_probability: float = 0.15,
    max_samples: Optional[int] = None,
    streaming: bool = True,
    data_dir: str = "mrt5/lm_datasets",
) -> Dataset:
    """
    Load mC4 dataset from preprocessed files.
    
    Args:
        split: One of "train", "validation", or "test"
        tokenizer: BERT tokenizer (will create one if not provided)
        max_length: Maximum sequence length
        mlm_probability: Probability of masking tokens
        max_samples: Maximum number of samples to load (None = all)
        streaming: Use iterable dataset for memory efficiency
        data_dir: Directory containing the preprocessed files
    
    Returns:
        Dataset object
    """
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    file_path = f"{data_dir}/mc4-en-{split}.json"
    
    if streaming:
        return MC4IterableDataset(
            file_path=file_path,
            tokenizer=tokenizer,
            max_length=max_length,
            mlm_probability=mlm_probability,
        )
    else:
        return MC4Dataset(
            file_path=file_path,
            tokenizer=tokenizer,
            max_length=max_length,
            mlm_probability=mlm_probability,
            max_samples=max_samples,
        )


if __name__ == "__main__":
    # Test the dataset
    from transformers import BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Test with validation set (smaller)
    print("Testing MC4Dataset loader...")
    dataset = load_mc4_dataset(
        split="validation",
        tokenizer=tokenizer,
        max_samples=100,
        streaming=False,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"input_ids shape: {sample['input_ids'].shape}")
    print(f"attention_mask shape: {sample['attention_mask'].shape}")
    print(f"labels shape: {sample['labels'].shape}")
    
    # Decode and print
    decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
    print(f"\nDecoded text (first 200 chars):\n{decoded[:200]}")
    
    # Count masked tokens
    num_masked = (sample['labels'] != -100).sum().item()
    total_tokens = (sample['attention_mask'] == 1).sum().item()
    print(f"\nMasked tokens: {num_masked}/{total_tokens} ({num_masked/total_tokens:.1%})")
