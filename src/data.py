"""
IRC Poker Database Parser

Parses the University of Alberta IRC Poker Database into our training format.

IRC Data Format (from documentation):
- Each hand is stored across multiple files by table
- Main format uses timestamp-based hand IDs
- Player actions recorded with amounts

Reference: https://poker.cs.ualberta.ca/irc_poker_database.html

Our Target Format:
    HAND|pos:CO|cards:AhKd|board:7s9c2h__|pot:45|stacks:180,95,220|
    HISTORY|SB:p1|BB:p2|UTG:f|MP:c2|CO:r6|BTN:f|SB:f|BB:c4|FLOP|BB:x|MP:b4|
    ACTION→c4
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Iterator
from collections import defaultdict
import json
import random


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PlayerAction:
    """A single action by a player."""
    player: str           # Player name or seat
    position: str         # SB, BB, UTG, MP, CO, BTN
    action: str           # f, x, c, b, r, p (fold, check, call, bet, raise, post)
    amount: Optional[int] # Amount for c, b, r, p
    
    def to_string(self) -> str:
        """Convert to our notation format."""
        if self.amount is not None:
            return f"{self.position}:{self.action}{self.amount}"
        return f"{self.position}:{self.action}"


@dataclass 
class Street:
    """Actions on a single street."""
    name: str                        # PRE, FLOP, TURN, RIVER
    cards: Optional[str] = None      # Board cards revealed this street
    actions: List[PlayerAction] = field(default_factory=list)


@dataclass
class PokerHand:
    """Complete representation of a poker hand."""
    hand_id: str
    timestamp: Optional[str] = None
    
    # Game setup
    num_players: int = 0
    button_seat: int = 0
    small_blind: int = 0
    big_blind: int = 0
    
    # Players
    players: Dict[int, str] = field(default_factory=dict)  # seat -> name
    stacks: Dict[int, int] = field(default_factory=dict)   # seat -> stack
    positions: Dict[int, str] = field(default_factory=dict) # seat -> position
    
    # Cards
    hole_cards: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # player -> (card1, card2)
    board: List[str] = field(default_factory=list)  # community cards
    
    # Action
    streets: List[Street] = field(default_factory=list)
    
    # Result
    winners: List[str] = field(default_factory=list)
    pot: int = 0
    
    def get_hero_cards(self, hero: str) -> Optional[Tuple[str, str]]:
        """Get hole cards for a specific player (if known)."""
        return self.hole_cards.get(hero)
    
    def get_position(self, seat: int) -> str:
        """Get position name for a seat."""
        return self.positions.get(seat, 'UNK')


# =============================================================================
# IRC FORMAT PARSER
# =============================================================================

class IRCParser:
    """
    Parser for IRC poker database format.
    
    The IRC database has a specific format with files like:
    - hdb: hand database (metadata)
    - hroster: player rosters per hand
    - pdb: player database
    
    Format example (simplified):
    
    Hand #12345 timestamp
    Table 'name' max_players
    Seat 1: Player1 (1000 in chips)
    Seat 2: Player2 (2000 in chips)
    ...
    Player1: posts small blind 5
    Player2: posts big blind 10
    *** HOLE CARDS ***
    Player3: folds
    Player4: calls 10
    *** FLOP *** [Ah 7c 2d]
    ...
    """
    
    # Position mapping based on number of players and seat relative to button
    POSITION_NAMES_6MAX = ['BTN', 'SB', 'BB', 'UTG', 'MP', 'CO']
    POSITION_NAMES_9MAX = ['BTN', 'SB', 'BB', 'UTG', 'UTG1', 'MP', 'MP1', 'HJ', 'CO']
    
    def __init__(self):
        self.hands: List[PokerHand] = []
    
    def assign_positions(self, num_players: int, button_seat: int, active_seats: List[int]) -> Dict[int, str]:
        """
        Assign position names based on button location and active seats.
        
        In poker, positions are relative to the button:
        - Button is the dealer
        - SB is left of button
        - BB is left of SB
        - Then UTG, MP, CO going around
        """
        positions = {}
        
        if num_players <= 6:
            pos_names = self.POSITION_NAMES_6MAX[:num_players]
        else:
            pos_names = self.POSITION_NAMES_9MAX[:num_players]
        
        # Sort seats and find button index
        sorted_seats = sorted(active_seats)
        
        # Rotate so button is first
        try:
            btn_idx = sorted_seats.index(button_seat)
        except ValueError:
            btn_idx = 0
        
        rotated = sorted_seats[btn_idx:] + sorted_seats[:btn_idx]
        
        # Assign positions
        for i, seat in enumerate(rotated):
            if i < len(pos_names):
                positions[seat] = pos_names[i]
            else:
                positions[seat] = f'S{seat}'
        
        return positions
    
    def parse_action_line(self, line: str, positions: Dict[int, str], player_seats: Dict[str, int]) -> Optional[PlayerAction]:
        """Parse a single action line."""
        
        # Common patterns:
        # "Player1: folds"
        # "Player1: calls 10"
        # "Player1: raises 20 to 30"
        # "Player1: bets 15"
        # "Player1: checks"
        # "Player1: posts small blind 5"
        
        patterns = [
            (r'(.+?): posts (?:small blind|big blind) (\d+)', 'p'),
            (r'(.+?): folds', 'f'),
            (r'(.+?): checks', 'x'),
            (r'(.+?): calls (\d+)', 'c'),
            (r'(.+?): bets (\d+)', 'b'),
            (r'(.+?): raises .* to (\d+)', 'r'),
            (r'(.+?): raises (\d+)', 'r'),
        ]
        
        for pattern, action_type in patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                player = match.group(1).strip()
                amount = int(match.group(2)) if len(match.groups()) > 1 else None
                
                seat = player_seats.get(player, 0)
                position = positions.get(seat, 'UNK')
                
                return PlayerAction(
                    player=player,
                    position=position,
                    action=action_type,
                    amount=amount
                )
        
        return None
    
    def parse_cards(self, card_str: str) -> List[str]:
        """
        Parse card string to list of cards.
        
        Input formats: "[Ah 7c 2d]" or "Ah7c2d" or "Ah 7c 2d"
        Output: ['Ah', '7c', '2d']
        """
        # Remove brackets
        card_str = card_str.replace('[', '').replace(']', '').strip()
        
        # If space-separated
        if ' ' in card_str:
            return card_str.split()
        
        # If continuous, split every 2 chars
        cards = []
        for i in range(0, len(card_str), 2):
            if i + 1 < len(card_str):
                cards.append(card_str[i:i+2])
        
        return cards


# =============================================================================
# SYNTHETIC DATA GENERATOR (for testing without IRC data)
# =============================================================================

class SyntheticPokerGenerator:
    """
    Generates synthetic poker hands for testing.
    
    This creates plausible-looking poker hands without needing the real dataset.
    Useful for:
    1. Testing the pipeline before downloading IRC data
    2. Augmenting training data
    3. Understanding the data format
    """
    
    POSITIONS_6MAX = ['SB', 'BB', 'UTG', 'MP', 'CO', 'BTN']
    
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    SUITS = ['c', 'd', 'h', 's']
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.deck = [f"{r}{s}" for r in self.RANKS for s in self.SUITS]
    
    def shuffle_deck(self) -> List[str]:
        """Return a shuffled deck."""
        deck = self.deck.copy()
        self.rng.shuffle(deck)
        return deck
    
    def generate_hand(self, num_players: int = 6, sb: int = 1, bb: int = 2) -> PokerHand:
        """Generate a single synthetic poker hand."""
        
        deck = self.shuffle_deck()
        deck_idx = 0
        
        hand = PokerHand(
            hand_id=f"synth_{self.rng.randint(100000, 999999)}",
            num_players=num_players,
            button_seat=num_players,  # Button is last seat
            small_blind=sb,
            big_blind=bb,
        )
        
        # Setup players with positions and stacks
        positions = self.POSITIONS_6MAX[:num_players]
        for i, pos in enumerate(positions):
            seat = i + 1
            hand.players[seat] = f"Player{seat}"
            hand.stacks[seat] = self.rng.randint(50, 200) * bb
            hand.positions[seat] = pos
        
        # Deal hole cards to each player
        for seat in hand.players:
            card1, card2 = deck[deck_idx], deck[deck_idx + 1]
            deck_idx += 2
            hand.hole_cards[hand.players[seat]] = (card1, card2)
        
        # Generate preflop action
        preflop = Street(name='PRE')
        active_players = list(hand.players.keys())
        
        # Blinds
        sb_seat = [s for s, p in hand.positions.items() if p == 'SB'][0]
        bb_seat = [s for s, p in hand.positions.items() if p == 'BB'][0]
        
        preflop.actions.append(PlayerAction(
            player=hand.players[sb_seat],
            position='SB',
            action='p',
            amount=sb
        ))
        preflop.actions.append(PlayerAction(
            player=hand.players[bb_seat],
            position='BB',
            action='p',
            amount=bb
        ))
        
        # Generate actions for other positions
        current_bet = bb
        for seat in active_players:
            pos = hand.positions[seat]
            if pos in ['SB', 'BB']:
                continue
            
            # Random action based on simple probabilities
            r = self.rng.random()
            if r < 0.4:  # Fold
                preflop.actions.append(PlayerAction(
                    player=hand.players[seat],
                    position=pos,
                    action='f',
                    amount=None
                ))
                active_players.remove(seat)
            elif r < 0.7:  # Call
                preflop.actions.append(PlayerAction(
                    player=hand.players[seat],
                    position=pos,
                    action='c',
                    amount=current_bet
                ))
            else:  # Raise
                raise_to = current_bet * self.rng.randint(2, 4)
                preflop.actions.append(PlayerAction(
                    player=hand.players[seat],
                    position=pos,
                    action='r',
                    amount=raise_to
                ))
                current_bet = raise_to
        
        hand.streets.append(preflop)
        
        # Generate flop if multiple players remain
        if len(active_players) >= 2:
            flop_cards = [deck[deck_idx + i] for i in range(3)]
            deck_idx += 3
            hand.board.extend(flop_cards)
            
            flop = Street(name='FLOP', cards=''.join(flop_cards))
            
            # Generate flop actions
            for seat in active_players:
                pos = hand.positions[seat]
                r = self.rng.random()
                
                if current_bet == 0:  # No bet yet
                    if r < 0.5:
                        flop.actions.append(PlayerAction(
                            player=hand.players[seat],
                            position=pos,
                            action='x',
                            amount=None
                        ))
                    else:
                        bet_amount = self.rng.randint(1, 3) * bb
                        flop.actions.append(PlayerAction(
                            player=hand.players[seat],
                            position=pos,
                            action='b',
                            amount=bet_amount
                        ))
                        current_bet = bet_amount
                else:
                    if r < 0.3:
                        flop.actions.append(PlayerAction(
                            player=hand.players[seat],
                            position=pos,
                            action='f',
                            amount=None
                        ))
                    else:
                        flop.actions.append(PlayerAction(
                            player=hand.players[seat],
                            position=pos,
                            action='c',
                            amount=current_bet
                        ))
            
            hand.streets.append(flop)
        
        # Could continue with turn/river but this is enough for testing
        
        # Set pot (simplified)
        hand.pot = sum(a.amount or 0 for s in hand.streets for a in s.actions)
        
        return hand


# =============================================================================
# TRAINING FORMAT CONVERTER
# =============================================================================

class TrainingFormatConverter:
    """
    Converts PokerHand objects to training format strings.
    
    Output format:
        HAND|pos:CO|cards:AhKd|board:7s9c2h__|pot:45|stacks:180,95,220|
        HISTORY|SB:p1|BB:p2|UTG:f|MP:c2|CO:r6|FLOP|BB:x|MP:b4|
        ACTION→c4
    """
    
    @staticmethod
    def format_board(board: List[str], street: str) -> str:
        """
        Format board cards with placeholders for unrevealed cards.
        
        PRE: _____
        FLOP: AhKd7c__
        TURN: AhKd7c2s_
        RIVER: AhKd7c2sJs
        """
        board_str = ''.join(board)
        
        # Pad to 5 cards (10 chars)
        while len(board_str) < 10:
            board_str += '_'
        
        return board_str
    
    @staticmethod
    def format_stacks(stacks: Dict[int, int]) -> str:
        """Format stack sizes as comma-separated values."""
        return ','.join(str(stacks[s]) for s in sorted(stacks.keys()))
    
    def convert_to_training_examples(
        self, 
        hand: PokerHand, 
        hero_position: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """
        Convert a hand to training examples.
        
        Each decision point becomes a (state, action) pair.
        If hero_position is specified, only generate examples for that position.
        Otherwise, generate examples for all positions where we know the hole cards.
        
        Returns:
            List of (state_string, action_string) tuples
        """
        examples = []
        
        # For each player whose cards we know
        for player, cards in hand.hole_cards.items():
            # Find this player's seat and position
            seat = None
            for s, p in hand.players.items():
                if p == player:
                    seat = s
                    break
            
            if seat is None:
                continue
            
            position = hand.positions.get(seat)
            if hero_position and position != hero_position:
                continue
            
            # Track game state as we replay the hand
            board_so_far = []
            history_parts = []
            
            for street in hand.streets:
                # Add street marker (except PRE which is implicit)
                if street.name != 'PRE' and street.cards:
                    # Add new board cards
                    new_cards = self.parse_board_cards(street.cards)
                    board_so_far.extend(new_cards)
                    history_parts.append(street.name)
                
                for action in street.actions:
                    if action.player == player:
                        # This is a decision point for our hero
                        state = self.build_state_string(
                            position=position,
                            cards=cards,
                            board=board_so_far,
                            pot=self.calculate_pot(history_parts),
                            stacks=hand.stacks,
                            history=history_parts
                        )
                        
                        action_str = action.to_string().split(':')[1]  # Just the action part
                        
                        examples.append((state, action_str))
                    
                    # Add action to history
                    history_parts.append(action.to_string())
        
        return examples
    
    def build_state_string(
        self,
        position: str,
        cards: Tuple[str, str],
        board: List[str],
        pot: int,
        stacks: Dict[int, int],
        history: List[str]
    ) -> str:
        """Build the state string for a decision point."""
        
        cards_str = ''.join(cards)
        board_str = self.format_board(board, self.get_current_street(history))
        stacks_str = self.format_stacks(stacks)
        history_str = '|'.join(history)
        
        state = f"HAND|pos:{position}|cards:{cards_str}|board:{board_str}|pot:{pot}|stacks:{stacks_str}|HISTORY|{history_str}|ACTION→"
        
        return state
    
    @staticmethod
    def get_current_street(history: List[str]) -> str:
        """Determine current street from history."""
        for item in reversed(history):
            if item in ['PRE', 'FLOP', 'TURN', 'RIVER']:
                return item
        return 'PRE'
    
    @staticmethod
    def calculate_pot(history: List[str]) -> int:
        """Calculate pot size from history (simplified)."""
        pot = 0
        for item in history:
            if ':' in item:
                action_part = item.split(':')[1]
                # Extract amount if present
                amount_match = re.search(r'\d+', action_part)
                if amount_match:
                    pot += int(amount_match.group())
        return pot
    
    @staticmethod
    def parse_board_cards(cards_str: str) -> List[str]:
        """Parse board cards string to list."""
        cards_str = cards_str.replace('_', '')
        cards = []
        for i in range(0, len(cards_str), 2):
            if i + 1 < len(cards_str):
                cards.append(cards_str[i:i+2])
        return cards


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_irc_data(
    raw_dir: str,
    output_dir: str,
    max_hands: Optional[int] = None,
    train_ratio: float = 0.9
) -> Dict:
    """
    Process IRC poker data into training format.
    
    Args:
        raw_dir: Directory containing raw IRC data
        output_dir: Directory for processed output
        max_hands: Maximum hands to process (None for all)
        train_ratio: Fraction for training (rest is validation)
    
    Returns:
        Statistics dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we have real IRC data
    raw_path = Path(raw_dir)
    if not raw_path.exists() or not any(raw_path.iterdir()):
        print("No IRC data found. Generating synthetic data for testing...")
        return generate_synthetic_data(output_dir, num_hands=max_hands or 10000, train_ratio=train_ratio)
    
    # TODO: Implement actual IRC parsing
    # For now, fall back to synthetic
    print("IRC parsing not yet implemented. Using synthetic data...")
    return generate_synthetic_data(output_dir, num_hands=max_hands or 10000, train_ratio=train_ratio)


def generate_synthetic_data(
    output_dir: Path,
    num_hands: int = 10000,
    train_ratio: float = 0.9
) -> Dict:
    """Generate synthetic poker data for testing."""
    
    generator = SyntheticPokerGenerator(seed=42)
    converter = TrainingFormatConverter()
    
    all_examples = []
    
    print(f"Generating {num_hands} synthetic hands...")
    for i in range(num_hands):
        hand = generator.generate_hand(num_players=6)
        examples = converter.convert_to_training_examples(hand)
        all_examples.extend(examples)
        
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1} hands, {len(all_examples)} examples...")
    
    print(f"Total training examples: {len(all_examples)}")
    
    # Shuffle and split
    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * train_ratio)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]
    
    # Save training data
    train_path = output_dir / 'train.txt'
    with open(train_path, 'w') as f:
        for state, action in train_examples:
            f.write(f"{state}{action}\n")
    
    # Save validation data  
    val_path = output_dir / 'val.txt'
    with open(val_path, 'w') as f:
        for state, action in val_examples:
            f.write(f"{state}{action}\n")
    
    # Compute statistics
    stats = {
        'num_hands': num_hands,
        'num_train_examples': len(train_examples),
        'num_val_examples': len(val_examples),
        'synthetic': True,
    }
    
    # Save statistics
    stats_path = output_dir / 'stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nData saved to {output_dir}")
    print(f"  Training examples: {len(train_examples)}")
    print(f"  Validation examples: {len(val_examples)}")
    
    return stats


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / 'data' / 'raw'
    processed_dir = project_root / 'data' / 'processed'
    
    stats = process_irc_data(
        raw_dir=str(raw_dir),
        output_dir=str(processed_dir),
        max_hands=10000
    )
    
    print("\n=== Data Processing Complete ===")
    print(f"Statistics: {stats}")
