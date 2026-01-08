"""
PyTorch Dataset for Poker Transformer

Loads (state, action) pairs for training action prediction.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import json

from tokenizer import PokerTokenizer, ActionEncoder


class PokerDataset(Dataset):
    """
    Dataset for poker action prediction.
    
    Each example is a (state, action_type, action_amount) tuple where:
    - state: tokenized game state (everything before ACTION→)
    - action_type: categorical (fold, check, call, bet, raise, allin)
    - action_amount: numerical (for c, b, r actions)
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PokerTokenizer,
        max_seq_length: int = 256,
    ):
        """
        Args:
            data_path: Path to train.txt or val.txt
            tokenizer: PokerTokenizer instance
            max_seq_length: Maximum sequence length (will pad/truncate)
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # Load data
        self.examples: List[Tuple[str, str]] = []
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Split at ACTION→
                if 'ACTION→' in line:
                    state, action = line.split('ACTION→')
                    state = state + 'ACTION→'  # Keep the marker
                    self.examples.append((state, action))
        
        print(f"Loaded {len(self.examples)} examples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.
        
        Returns dict with:
            - input_ids: tokenized state sequence [seq_len]
            - attention_mask: 1 for real tokens, 0 for padding [seq_len]
            - action_type: categorical action label [1]
            - action_amount: bet/raise amount or 0 [1]
        """
        state, action = self.examples[idx]
        
        # Tokenize state
        input_ids = self.tokenizer.encode(state, add_special_tokens=True)
        
        # Truncate if needed
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
        
        # Create attention mask (1 for real tokens)
        attention_mask = [1] * len(input_ids)
        
        # Pad if needed
        padding_length = self.max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        # Parse action
        action_type, action_amount = ActionEncoder.decode_action(action)
        action_type_id = ActionEncoder.get_action_type_id(action_type)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'action_type': torch.tensor(action_type_id, dtype=torch.long),
            'action_amount': torch.tensor(action_amount or 0, dtype=torch.float),
        }


class PokerDatasetSimple(Dataset):
    """
    Simplified dataset for Phase 1-6: character-level next-token prediction.
    
    Before we tackle action prediction, we train the transformer basics
    on simple sequence modeling: given previous tokens, predict next token.
    
    This is similar to how GPT is trained, and lets us verify the 
    transformer architecture works before adding the action prediction head.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PokerTokenizer,
        block_size: int = 128,
    ):
        """
        Args:
            data_path: Path to train.txt or val.txt
            tokenizer: PokerTokenizer instance
            block_size: Context window size
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Load and concatenate all examples into one long sequence
        print(f"Loading data from {data_path}...")
        
        all_tokens = []
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = self.tokenizer.encode(line, add_special_tokens=False)
                    all_tokens.extend(tokens)
                    all_tokens.append(self.tokenizer.eos_id)  # Separator between hands
        
        self.data = torch.tensor(all_tokens, dtype=torch.long)
        
        print(f"Total tokens: {len(self.data):,}")
        print(f"Block size: {block_size}")
        print(f"Number of training examples: {len(self):,}")
    
    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size - 1)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training example.
        
        Returns:
            x: input sequence [block_size]
            y: target sequence [block_size] (shifted by 1)
        """
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def get_dataloaders(
    data_dir: str,
    tokenizer: PokerTokenizer,
    batch_size: int = 32,
    block_size: int = 128,
    num_workers: int = 0,
    simple: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Directory containing train.txt, val.txt
        tokenizer: PokerTokenizer instance
        batch_size: Batch size
        block_size: Context window (for simple dataset)
        num_workers: DataLoader workers
        simple: If True, use PokerDatasetSimple for next-token prediction
                If False, use PokerDataset for action prediction
    
    Returns:
        train_loader, val_loader
    """
    data_dir = Path(data_dir)
    
    DatasetClass = PokerDatasetSimple if simple else PokerDataset
    
    if simple:
        train_dataset = DatasetClass(
            data_path=str(data_dir / 'train.txt'),
            tokenizer=tokenizer,
            block_size=block_size,
        )
        val_dataset = DatasetClass(
            data_path=str(data_dir / 'val.txt'),
            tokenizer=tokenizer,
            block_size=block_size,
        )
    else:
        train_dataset = DatasetClass(
            data_path=str(data_dir / 'train.txt'),
            tokenizer=tokenizer,
            max_seq_length=block_size,
        )
        val_dataset = DatasetClass(
            data_path=str(data_dir / 'val.txt'),
            tokenizer=tokenizer,
            max_seq_length=block_size,
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'processed'
    
    # Check if data exists
    if not (data_dir / 'train.txt').exists():
        print("No processed data found. Run data.py first:")
        print("  python src/data.py")
        exit(1)
    
    # Create tokenizer
    tokenizer = PokerTokenizer()
    print(f"Vocabulary size: {len(tokenizer)}")
    print()
    
    # Test simple dataset (for Phase 1-6)
    print("=== Testing Simple Dataset (Next-Token Prediction) ===")
    print()
    
    train_dataset = PokerDatasetSimple(
        data_path=str(data_dir / 'train.txt'),
        tokenizer=tokenizer,
        block_size=128,
    )
    
    # Get a sample
    x, y = train_dataset[0]
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    print()
    
    print("First example:")
    print(f"  Input (first 50 tokens):  {x[:50].tolist()}")
    print(f"  Target (first 50 tokens): {y[:50].tolist()}")
    print()
    
    # Decode
    print("Decoded input (first 100 chars):")
    decoded = tokenizer.decode(x.tolist())
    print(f"  {decoded[:100]}...")
    print()
    
    # Test DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    batch_x, batch_y = next(iter(train_loader))
    print(f"Batch shapes: x={batch_x.shape}, y={batch_y.shape}")
    print()
    
    # Test action prediction dataset
    print("=== Testing Action Prediction Dataset ===")
    print()
    
    action_dataset = PokerDataset(
        data_path=str(data_dir / 'train.txt'),
        tokenizer=tokenizer,
        max_seq_length=256,
    )
    
    sample = action_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  attention_mask shape: {sample['attention_mask'].shape}")
    print(f"  action_type: {sample['action_type'].item()}")
    print(f"  action_amount: {sample['action_amount'].item()}")
    
    # Show action distribution
    print()
    print("Action type distribution (first 1000 examples):")
    from collections import Counter
    action_counts = Counter()
    for i in range(min(1000, len(action_dataset))):
        action_counts[action_dataset.examples[i][1][0]] += 1
    
    action_names = {'f': 'fold', 'x': 'check', 'c': 'call', 'b': 'bet', 'r': 'raise', 'p': 'post'}
    for action, count in action_counts.most_common():
        name = action_names.get(action, action)
        print(f"  {name}: {count} ({100*count/sum(action_counts.values()):.1f}%)")
