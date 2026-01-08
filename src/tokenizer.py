"""
Poker Tokenizer

Defines the vocabulary for encoding poker states and actions.
This is the foundation for everything — the model only sees token IDs.

Vocabulary Design Decisions:
1. Cards as single tokens (Ah, Kd, etc.) — not split into rank+suit
2. Actions as single tokens (f, x, c, b, r) + separate number tokens
3. Positions as single tokens (SB, BB, UTG, MP, CO, BTN)
4. Special markers for structure (HAND, FLOP, TURN, RIVER, etc.)
5. Numbers tokenized as individual digits (easier for model to learn arithmetic patterns)
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# VOCABULARY DEFINITION
# =============================================================================

# Special tokens
SPECIAL_TOKENS = [
    '<PAD>',      # Padding for batch alignment
    '<UNK>',      # Unknown token (should rarely be used)
    '<SOS>',      # Start of sequence
    '<EOS>',      # End of sequence
    '<SEP>',      # Separator
    '<MASK>',     # For potential masked training
]

# Structural markers
STRUCTURE_TOKENS = [
    'HAND',       # Start of hand metadata
    'HISTORY',    # Start of action history
    'ACTION',     # Action prediction point
    'PRE',        # Preflop
    'FLOP',       # Flop
    'TURN',       # Turn
    'RIVER',      # River
    'WIN',        # Showdown/result
    '|',          # Field separator
    ':',          # Key-value separator
    '→',          # Action arrow (ACTION→)
    ',',          # List separator (for stacks)
    '_',          # Empty board slot
]

# Positions (6-max table)
POSITION_TOKENS = [
    'SB',         # Small blind
    'BB',         # Big blind
    'UTG',        # Under the gun
    'MP',         # Middle position
    'CO',         # Cutoff
    'BTN',        # Button
]

# Extended positions (for 9-max or varying table sizes)
EXTENDED_POSITION_TOKENS = [
    'UTG1',       # UTG+1
    'UTG2',       # UTG+2
    'MP1',        # MP+1
    'MP2',        # MP+2
    'HJ',         # Hijack
]

# Actions
ACTION_TOKENS = [
    'f',          # Fold
    'x',          # Check
    'c',          # Call (followed by amount)
    'b',          # Bet (followed by amount)
    'r',          # Raise (followed by amount)
    'p',          # Post blind (followed by amount)
    'a',          # All-in
]

# Cards: 52 cards as individual tokens
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['c', 'd', 'h', 's']  # clubs, diamonds, hearts, spades
CARD_TOKENS = [f"{rank}{suit}" for rank in RANKS for suit in SUITS]

# Numbers (for bet sizes, pot sizes, stacks)
# Using individual digits allows the model to learn numerical patterns
DIGIT_TOKENS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Keywords for state encoding
KEYWORD_TOKENS = [
    'pos',        # Position
    'cards',      # Hole cards
    'board',      # Community cards
    'pot',        # Pot size
    'stacks',     # Stack sizes
    'bet',        # Current bet to call
]


def build_vocabulary() -> Dict[str, int]:
    """
    Build the complete vocabulary mapping token -> index.
    
    Order matters for special tokens (PAD should be 0).
    """
    vocab = {}
    idx = 0
    
    # Special tokens first
    for token in SPECIAL_TOKENS:
        vocab[token] = idx
        idx += 1
    
    # Structure
    for token in STRUCTURE_TOKENS:
        vocab[token] = idx
        idx += 1
    
    # Positions
    for token in POSITION_TOKENS + EXTENDED_POSITION_TOKENS:
        vocab[token] = idx
        idx += 1
    
    # Actions
    for token in ACTION_TOKENS:
        vocab[token] = idx
        idx += 1
    
    # Cards
    for token in CARD_TOKENS:
        vocab[token] = idx
        idx += 1
    
    # Digits
    for token in DIGIT_TOKENS:
        vocab[token] = idx
        idx += 1
    
    # Keywords
    for token in KEYWORD_TOKENS:
        vocab[token] = idx
        idx += 1
    
    return vocab


# =============================================================================
# TOKENIZER CLASS
# =============================================================================

@dataclass
class TokenizerConfig:
    """Configuration for the tokenizer."""
    max_sequence_length: int = 512
    pad_token: str = '<PAD>'
    unk_token: str = '<UNK>'
    sos_token: str = '<SOS>'
    eos_token: str = '<EOS>'


class PokerTokenizer:
    """
    Tokenizer for poker game states.
    
    Handles encoding (text → token IDs) and decoding (token IDs → text).
    """
    
    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self.vocab = build_vocabulary()
        self.vocab_size = len(self.vocab)
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        
        # Cache special token IDs
        self.pad_id = self.vocab[self.config.pad_token]
        self.unk_id = self.vocab[self.config.unk_token]
        self.sos_id = self.vocab[self.config.sos_token]
        self.eos_id = self.vocab[self.config.eos_token]
    
    def tokenize(self, text: str) -> List[str]:
        """
        Convert text to list of tokens.
        
        This is a simple tokenizer that splits on known patterns.
        For poker notation, most tokens are clearly delimited.
        """
        tokens = []
        i = 0
        text = text.strip()
        
        while i < len(text):
            # Skip whitespace
            if text[i].isspace():
                i += 1
                continue
            
            matched = False
            
            # Try to match multi-character tokens first (longest match)
            # Order: keywords, positions, structure, cards, actions, digits
            
            # Check for keywords (pos, cards, board, pot, stacks, bet)
            for keyword in KEYWORD_TOKENS:
                if text[i:i+len(keyword)] == keyword:
                    tokens.append(keyword)
                    i += len(keyword)
                    matched = True
                    break
            
            if matched:
                continue
            
            # Check for structure tokens
            for struct in ['HISTORY', 'ACTION', 'HAND', 'FLOP', 'TURN', 'RIVER', 'WIN', 'PRE']:
                if text[i:i+len(struct)] == struct:
                    tokens.append(struct)
                    i += len(struct)
                    matched = True
                    break
            
            if matched:
                continue
            
            # Check for positions (before checking 2-char cards)
            for pos in POSITION_TOKENS + EXTENDED_POSITION_TOKENS:
                if text[i:i+len(pos)] == pos:
                    # Make sure it's not part of a longer word
                    end = i + len(pos)
                    if end >= len(text) or not text[end].isalnum():
                        tokens.append(pos)
                        i = end
                        matched = True
                        break
            
            if matched:
                continue
            
            # Check for cards (2 characters: rank + suit)
            if i + 1 < len(text):
                potential_card = text[i:i+2]
                if potential_card in CARD_TOKENS:
                    tokens.append(potential_card)
                    i += 2
                    matched = True
            
            if matched:
                continue
            
            # Check for single-character tokens
            char = text[i]
            
            if char in ACTION_TOKENS:
                tokens.append(char)
                i += 1
            elif char in DIGIT_TOKENS:
                tokens.append(char)
                i += 1
            elif char in ['|', ':', '→', ',', '_']:
                tokens.append(char)
                i += 1
            else:
                # Unknown character — skip or add as UNK
                # For now, skip (we shouldn't hit this with clean data)
                i += 1
        
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Convert text to list of token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add SOS/EOS tokens
        
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = [self.config.sos_token] + tokens + [self.config.eos_token]
        
        ids = [self.vocab.get(t, self.unk_id) for t in tokens]
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert token IDs back to text.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip PAD, SOS, EOS, etc.
        
        Returns:
            Decoded text
        """
        special = {self.pad_id, self.sos_id, self.eos_id}
        
        tokens = []
        for idx in ids:
            if skip_special_tokens and idx in special:
                continue
            token = self.idx_to_token.get(idx, self.config.unk_token)
            tokens.append(token)
        
        # Join with spaces, but handle separators intelligently
        result = []
        for i, token in enumerate(tokens):
            if token in ['|', ':', '→', ',']:
                result.append(token)
            elif i > 0 and tokens[i-1] in ['|', ':', '→', ',']:
                result.append(token)
            else:
                result.append(' ' + token if result else token)
        
        return ''.join(result)
    
    def pad_sequence(self, ids: List[int], max_length: Optional[int] = None) -> List[int]:
        """Pad sequence to max_length."""
        max_length = max_length or self.config.max_sequence_length
        
        if len(ids) >= max_length:
            return ids[:max_length]
        
        return ids + [self.pad_id] * (max_length - len(ids))
    
    def save(self, path: str) -> None:
        """Save vocabulary to file."""
        with open(path, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'config': {
                    'max_sequence_length': self.config.max_sequence_length,
                    'pad_token': self.config.pad_token,
                    'unk_token': self.config.unk_token,
                    'sos_token': self.config.sos_token,
                    'eos_token': self.config.eos_token,
                }
            }, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'PokerTokenizer':
        """Load tokenizer from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        config = TokenizerConfig(**data['config'])
        tokenizer = cls(config)
        tokenizer.vocab = data['vocab']
        tokenizer.idx_to_token = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.vocab_size = len(tokenizer.vocab)
        return tokenizer
    
    def __len__(self) -> int:
        return self.vocab_size


# =============================================================================
# ACTION ENCODING
# =============================================================================

class ActionEncoder:
    """
    Encodes and decodes poker actions.
    
    Actions are encoded as:
    - Fold: 'f' (single token)
    - Check: 'x' (single token)  
    - Call: 'c' + amount tokens (e.g., 'c', '1', '2' for call 12)
    - Bet: 'b' + amount tokens
    - Raise: 'r' + amount tokens
    - All-in: 'a' (single token)
    """
    
    # Action type IDs (for classification head)
    ACTION_TYPES = {
        'f': 0,   # Fold
        'x': 1,   # Check
        'c': 2,   # Call
        'b': 3,   # Bet
        'r': 4,   # Raise
        'a': 5,   # All-in
    }
    
    NUM_ACTION_TYPES = len(ACTION_TYPES)
    
    @classmethod
    def encode_action(cls, action_type: str, amount: Optional[int] = None) -> str:
        """
        Encode an action as a string.
        
        Args:
            action_type: One of 'f', 'x', 'c', 'b', 'r', 'a'
            amount: Bet/raise/call amount (required for c, b, r)
        
        Returns:
            Encoded action string
        """
        if action_type in ['f', 'x', 'a']:
            return action_type
        elif action_type in ['c', 'b', 'r']:
            if amount is None:
                raise ValueError(f"Amount required for action type '{action_type}'")
            return f"{action_type}{amount}"
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    
    @classmethod
    def decode_action(cls, action_str: str) -> Tuple[str, Optional[int]]:
        """
        Decode an action string.
        
        Returns:
            Tuple of (action_type, amount or None)
        """
        if not action_str:
            raise ValueError("Empty action string")
        
        action_type = action_str[0]
        
        if action_type in ['f', 'x', 'a']:
            return action_type, None
        elif action_type in ['c', 'b', 'r']:
            amount_str = action_str[1:]
            amount = int(amount_str) if amount_str else None
            return action_type, amount
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    
    @classmethod
    def get_action_type_id(cls, action_type: str) -> int:
        """Get numeric ID for action type."""
        return cls.ACTION_TYPES.get(action_type, -1)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    # Build and inspect vocabulary
    tokenizer = PokerTokenizer()
    
    print("=== Poker Tokenizer ===")
    print(f"Vocabulary size: {len(tokenizer)}")
    print()
    
    # Show vocabulary by category
    print("Vocabulary breakdown:")
    print(f"  Special tokens: {len(SPECIAL_TOKENS)}")
    print(f"  Structure tokens: {len(STRUCTURE_TOKENS)}")
    print(f"  Position tokens: {len(POSITION_TOKENS) + len(EXTENDED_POSITION_TOKENS)}")
    print(f"  Action tokens: {len(ACTION_TOKENS)}")
    print(f"  Card tokens: {len(CARD_TOKENS)}")
    print(f"  Digit tokens: {len(DIGIT_TOKENS)}")
    print(f"  Keyword tokens: {len(KEYWORD_TOKENS)}")
    print()
    
    # Test encoding/decoding
    test_input = "HAND|pos:CO|cards:AhKd|board:7s9c2h__|pot:45|stacks:180,95,220|HISTORY|SB:p1|BB:p2|UTG:f|MP:c2|CO:r6|FLOP|BB:x|MP:b4|ACTION→c4"
    
    print("Test input:")
    print(f"  {test_input}")
    print()
    
    tokens = tokenizer.tokenize(test_input)
    print(f"Tokens ({len(tokens)}):")
    print(f"  {tokens}")
    print()
    
    ids = tokenizer.encode(test_input)
    print(f"Token IDs ({len(ids)}):")
    print(f"  {ids}")
    print()
    
    decoded = tokenizer.decode(ids)
    print("Decoded:")
    print(f"  {decoded}")
    print()
    
    # Save vocabulary
    vocab_path = Path(__file__).parent.parent / 'data' / 'processed' / 'vocab.json'
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(vocab_path))
    print(f"Vocabulary saved to: {vocab_path}")
