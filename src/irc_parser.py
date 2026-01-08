"""
IRC Poker Database Parser

Parses the University of Alberta IRC Poker Database format.
Joins data from three file types:
- hdb: Hand database (hand metadata, pot sizes, board cards)
- hroster: Hand roster (which players in each hand, seat order)
- pdb.{name}: Player database (individual player actions per hand)

Reference: https://poker.cs.ualberta.ca/irc_poker_database.html
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Iterator, Set
from collections import defaultdict
import json


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class HandMetadata:
    """Hand-level metadata from hdb file."""
    timestamp: int
    table_id: int
    hand_num: int
    num_players: int
    preflop_players: int
    preflop_pot: int
    flop_players: int
    flop_pot: int
    turn_players: int
    turn_pot: int
    river_players: int  # Actually winners count
    total_pot: int
    board: List[str] = field(default_factory=list)


@dataclass
class PlayerHand:
    """Single player's participation in a hand, from pdb file."""
    player_name: str
    timestamp: int
    num_players: int
    seat: int
    preflop_actions: str
    flop_actions: str
    turn_actions: str
    river_actions: str
    stack_after: int
    chips_invested: int
    chips_won: int
    hole_cards: Optional[Tuple[str, str]] = None
    
    @property
    def net_profit(self) -> int:
        return self.chips_won - self.chips_invested
    
    @property
    def won_hand(self) -> bool:
        return self.chips_won > 0


@dataclass
class HandRoster:
    """Player roster for a hand, from hroster file."""
    timestamp: int
    num_players: int
    players: List[str]  # In seat order


@dataclass 
class CompleteHand:
    """Fully reconstructed hand with all information joined."""
    timestamp: int
    num_players: int
    board: List[str]
    total_pot: int
    players: Dict[str, PlayerHand]  # name -> PlayerHand
    seat_order: List[str]  # Player names in seat order
    
    # Derived fields
    button_seat: int = 0  # Will be inferred
    small_blind: int = 0
    big_blind: int = 0
    
    def get_player_position(self, player_name: str) -> Optional[str]:
        """Get position name (SB, BB, UTG, etc.) for a player."""
        if player_name not in self.seat_order:
            return None
        
        seat_idx = self.seat_order.index(player_name)
        n = len(self.seat_order)
        
        # In IRC data, seat 1 posted SB, seat 2 posted BB
        # Positions are: SB, BB, UTG, MP, CO, BTN (for 6-max)
        positions_6max = ['SB', 'BB', 'UTG', 'MP', 'CO', 'BTN']
        positions_4max = ['SB', 'BB', 'CO', 'BTN']
        positions_3max = ['SB', 'BB', 'BTN']
        positions_2max = ['SB', 'BB']
        
        if n == 2:
            positions = positions_2max
        elif n == 3:
            positions = positions_3max
        elif n == 4:
            positions = positions_4max
        elif n <= 6:
            positions = positions_6max[:n]
        else:
            # For larger tables, use generic position names
            positions = ['SB', 'BB'] + [f'P{i}' for i in range(3, n+1)]
        
        return positions[seat_idx] if seat_idx < len(positions) else f'P{seat_idx+1}'


# =============================================================================
# PARSERS
# =============================================================================

class HDBParser:
    """Parser for hdb (hand database) files."""
    
    @staticmethod
    def parse_pot_info(pot_str: str) -> Tuple[int, int]:
        """Parse 'players/pot' format like '2/540'."""
        if '/' not in pot_str:
            return 0, 0
        parts = pot_str.split('/')
        return int(parts[0]), int(parts[1])
    
    @staticmethod
    def parse_line(line: str) -> Optional[HandMetadata]:
        """Parse a single hdb line."""
        parts = line.strip().split()
        if len(parts) < 8:
            return None
        
        try:
            timestamp = int(parts[0])
            table_id = int(parts[1])
            hand_num = int(parts[2])
            num_players = int(parts[3])
            
            pre_players, pre_pot = HDBParser.parse_pot_info(parts[4])
            flop_players, flop_pot = HDBParser.parse_pot_info(parts[5])
            turn_players, turn_pot = HDBParser.parse_pot_info(parts[6])
            river_players, total_pot = HDBParser.parse_pot_info(parts[7])
            
            # Board cards are at the end (if present)
            board = []
            if len(parts) > 8:
                board = parts[8:]
            
            return HandMetadata(
                timestamp=timestamp,
                table_id=table_id,
                hand_num=hand_num,
                num_players=num_players,
                preflop_players=pre_players,
                preflop_pot=pre_pot,
                flop_players=flop_players,
                flop_pot=flop_pot,
                turn_players=turn_players,
                turn_pot=turn_pot,
                river_players=river_players,
                total_pot=total_pot,
                board=board
            )
        except (ValueError, IndexError) as e:
            return None
    
    @staticmethod
    def parse_file(filepath: str) -> Dict[int, HandMetadata]:
        """Parse entire hdb file, return dict keyed by timestamp."""
        hands = {}
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                hand = HDBParser.parse_line(line)
                if hand:
                    hands[hand.timestamp] = hand
        return hands


class HRosterParser:
    """Parser for hroster (hand roster) files."""
    
    @staticmethod
    def parse_line(line: str) -> Optional[HandRoster]:
        """Parse a single hroster line."""
        parts = line.strip().split()
        if len(parts) < 3:
            return None
        
        try:
            timestamp = int(parts[0])
            num_players = int(parts[1])
            players = parts[2:]
            
            return HandRoster(
                timestamp=timestamp,
                num_players=num_players,
                players=players
            )
        except (ValueError, IndexError):
            return None
    
    @staticmethod
    def parse_file(filepath: str) -> Dict[int, HandRoster]:
        """Parse entire hroster file, return dict keyed by timestamp."""
        rosters = {}
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                roster = HRosterParser.parse_line(line)
                if roster:
                    rosters[roster.timestamp] = roster
        return rosters


class PDBParser:
    """Parser for pdb.{player} (player database) files."""
    
    # Card pattern: rank + suit (e.g., Ah, Kd, Tc)
    CARD_PATTERN = re.compile(r'^[2-9TJQKA][cdhs]$')
    
    @staticmethod
    def parse_line(line: str) -> Optional[PlayerHand]:
        """Parse a single pdb line."""
        parts = line.strip().split()
        if len(parts) < 11:
            return None
        
        try:
            player_name = parts[0]
            timestamp = int(parts[1])
            num_players = int(parts[2])
            seat = int(parts[3])
            
            preflop = parts[4]
            flop = parts[5]
            turn = parts[6]
            river = parts[7]
            
            stack_after = int(parts[8])
            chips_invested = int(parts[9])
            chips_won = int(parts[10])
            
            # Hole cards at the end (if shown)
            hole_cards = None
            if len(parts) >= 13:
                card1, card2 = parts[11], parts[12]
                if PDBParser.CARD_PATTERN.match(card1) and PDBParser.CARD_PATTERN.match(card2):
                    hole_cards = (card1, card2)
            
            return PlayerHand(
                player_name=player_name,
                timestamp=timestamp,
                num_players=num_players,
                seat=seat,
                preflop_actions=preflop,
                flop_actions=flop,
                turn_actions=turn,
                river_actions=river,
                stack_after=stack_after,
                chips_invested=chips_invested,
                chips_won=chips_won,
                hole_cards=hole_cards
            )
        except (ValueError, IndexError):
            return None
    
    @staticmethod
    def parse_file(filepath: str) -> List[PlayerHand]:
        """Parse entire pdb file, return list of PlayerHands."""
        hands = []
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                hand = PDBParser.parse_line(line)
                if hand:
                    hands.append(hand)
        return hands


# =============================================================================
# DATA JOINER
# =============================================================================

class IRCDataJoiner:
    """Joins data from hdb, hroster, and pdb files into complete hands."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
        # Parsed data
        self.hand_metadata: Dict[int, HandMetadata] = {}
        self.rosters: Dict[int, HandRoster] = {}
        self.player_hands: Dict[int, Dict[str, PlayerHand]] = defaultdict(dict)  # timestamp -> {player -> hand}
        
        # Player statistics
        self.player_stats: Dict[str, Dict] = defaultdict(lambda: {
            'hands_played': 0,
            'hands_won': 0,
            'total_invested': 0,
            'total_won': 0,
            'showdowns': 0,
            'vpip_hands': 0,  # Voluntarily put money in pot
            'pfr_hands': 0,   # Pre-flop raise
        })
    
    def load_all(self) -> None:
        """Load all data files from the directory."""
        print(f"Loading data from {self.data_dir}...")
        
        # Load hdb
        hdb_path = self.data_dir / 'hdb'
        if hdb_path.exists():
            print(f"  Loading hdb...")
            self.hand_metadata = HDBParser.parse_file(str(hdb_path))
            print(f"    Found {len(self.hand_metadata):,} hands")
        
        # Load hroster
        hroster_path = self.data_dir / 'hroster'
        if hroster_path.exists():
            print(f"  Loading hroster...")
            self.rosters = HRosterParser.parse_file(str(hroster_path))
            print(f"    Found {len(self.rosters):,} rosters")
        
        # Load all pdb files
        pdb_dir = self.data_dir / 'pdb'
        if pdb_dir.exists() and pdb_dir.is_dir():
            pdb_files = list(pdb_dir.glob('pdb.*'))
        else:
            # pdb files might be directly in data_dir
            pdb_files = list(self.data_dir.glob('pdb.*'))
        
        print(f"  Loading {len(pdb_files)} pdb files...")
        total_player_hands = 0
        for pdb_file in pdb_files:
            player_hands = PDBParser.parse_file(str(pdb_file))
            for ph in player_hands:
                self.player_hands[ph.timestamp][ph.player_name] = ph
                self._update_player_stats(ph)
            total_player_hands += len(player_hands)
        
        print(f"    Found {total_player_hands:,} player-hand records")
        print(f"    Unique players: {len(self.player_stats):,}")
    
    def _update_player_stats(self, ph: PlayerHand) -> None:
        """Update aggregate statistics for a player."""
        stats = self.player_stats[ph.player_name]
        stats['hands_played'] += 1
        stats['total_invested'] += ph.chips_invested
        stats['total_won'] += ph.chips_won
        
        if ph.won_hand:
            stats['hands_won'] += 1
        
        if ph.hole_cards:
            stats['showdowns'] += 1
        
        # VPIP: did they voluntarily put money in? (not just posting blinds)
        preflop = ph.preflop_actions.replace('B', '').replace('b', '')  # Remove blind posts
        if 'c' in preflop or 'r' in preflop:
            stats['vpip_hands'] += 1
        
        # PFR: did they raise preflop?
        if 'r' in ph.preflop_actions:
            stats['pfr_hands'] += 1
    
    def get_complete_hand(self, timestamp: int) -> Optional[CompleteHand]:
        """Reconstruct a complete hand from all data sources."""
        if timestamp not in self.hand_metadata:
            return None
        if timestamp not in self.rosters:
            return None
        if timestamp not in self.player_hands:
            return None
        
        metadata = self.hand_metadata[timestamp]
        roster = self.rosters[timestamp]
        player_hands = self.player_hands[timestamp]
        
        # Verify we have all players
        if len(player_hands) != roster.num_players:
            return None  # Incomplete data
        
        return CompleteHand(
            timestamp=timestamp,
            num_players=metadata.num_players,
            board=metadata.board,
            total_pot=metadata.total_pot,
            players=player_hands,
            seat_order=roster.players
        )
    
    def iter_complete_hands(self) -> Iterator[CompleteHand]:
        """Iterate over all complete hands."""
        for timestamp in self.hand_metadata:
            hand = self.get_complete_hand(timestamp)
            if hand:
                yield hand
    
    def get_player_rankings(self, min_hands: int = 100) -> List[Tuple[str, Dict]]:
        """
        Get players ranked by profitability.
        
        Returns list of (player_name, stats) sorted by net profit.
        Only includes players with at least min_hands.
        """
        rankings = []
        for player, stats in self.player_stats.items():
            if stats['hands_played'] >= min_hands:
                stats_copy = dict(stats)
                stats_copy['net_profit'] = stats['total_won'] - stats['total_invested']
                stats_copy['win_rate'] = stats['hands_won'] / stats['hands_played'] if stats['hands_played'] > 0 else 0
                stats_copy['bb_per_hand'] = stats_copy['net_profit'] / stats['hands_played'] if stats['hands_played'] > 0 else 0
                stats_copy['vpip'] = stats['vpip_hands'] / stats['hands_played'] if stats['hands_played'] > 0 else 0
                stats_copy['pfr'] = stats['pfr_hands'] / stats['hands_played'] if stats['hands_played'] > 0 else 0
                rankings.append((player, stats_copy))
        
        # Sort by net profit descending
        rankings.sort(key=lambda x: x[1]['net_profit'], reverse=True)
        return rankings


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_dataset(data_dir: str) -> Dict:
    """
    Comprehensive analysis of the IRC dataset.
    
    Returns statistics dictionary.
    """
    joiner = IRCDataJoiner(data_dir)
    joiner.load_all()
    
    stats = {
        'total_hands': len(joiner.hand_metadata),
        'total_rosters': len(joiner.rosters),
        'unique_players': len(joiner.player_stats),
        'hands_with_board': 0,
        'hands_by_player_count': defaultdict(int),
        'hands_reaching_showdown': 0,
        'total_player_hand_records': sum(len(ph) for ph in joiner.player_hands.values()),
    }
    
    # Analyze hands
    complete_hands = 0
    for timestamp, metadata in joiner.hand_metadata.items():
        if metadata.board:
            stats['hands_with_board'] += 1
        stats['hands_by_player_count'][metadata.num_players] += 1
        
        # Check if hand is complete
        hand = joiner.get_complete_hand(timestamp)
        if hand:
            complete_hands += 1
    
    stats['complete_hands'] = complete_hands
    stats['hands_by_player_count'] = dict(stats['hands_by_player_count'])
    
    # Player analysis
    rankings = joiner.get_player_rankings(min_hands=100)
    stats['players_with_100plus_hands'] = len(rankings)
    
    if rankings:
        profits = [r[1]['net_profit'] for r in rankings]
        stats['top_10_players'] = [(name, s['net_profit'], s['hands_played']) for name, s in rankings[:10]]
        stats['bottom_10_players'] = [(name, s['net_profit'], s['hands_played']) for name, s in rankings[-10:]]
        
        # How many are profitable?
        profitable = sum(1 for _, s in rankings if s['net_profit'] > 0)
        stats['profitable_players'] = profitable
        stats['profitable_player_pct'] = profitable / len(rankings) * 100 if rankings else 0
    
    return stats, joiner


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        # Default: look for data in standard location
        data_dir = Path(__file__).parent.parent / 'data' / 'raw' / 'holdem'
    
    if not Path(data_dir).exists():
        print(f"Data directory not found: {data_dir}")
        print("Usage: python irc_parser.py <path_to_holdem_data>")
        print("\nExpected directory structure:")
        print("  <data_dir>/")
        print("    hdb")
        print("    hroster")
        print("    pdb/")
        print("      pdb.Player1")
        print("      pdb.Player2")
        print("      ...")
        sys.exit(1)
    
    print("=" * 60)
    print("IRC Poker Database Analysis")
    print("=" * 60)
    print()
    
    stats, joiner = analyze_dataset(str(data_dir))
    
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total hands in hdb:         {stats['total_hands']:,}")
    print(f"Total rosters:              {stats['total_rosters']:,}")
    print(f"Complete hands (all data):  {stats['complete_hands']:,}")
    print(f"Hands with board cards:     {stats['hands_with_board']:,}")
    print(f"Unique players:             {stats['unique_players']:,}")
    print(f"Total player-hand records:  {stats['total_player_hand_records']:,}")
    
    print("\n" + "-" * 40)
    print("HANDS BY PLAYER COUNT")
    print("-" * 40)
    for n_players, count in sorted(stats['hands_by_player_count'].items()):
        print(f"  {n_players} players: {count:,}")
    
    print("\n" + "-" * 40)
    print("PLAYER SKILL DISTRIBUTION")
    print("-" * 40)
    print(f"Players with 100+ hands: {stats['players_with_100plus_hands']}")
    print(f"Profitable players: {stats['profitable_players']} ({stats['profitable_player_pct']:.1f}%)")
    
    if 'top_10_players' in stats:
        print("\nTop 10 Players (by net profit):")
        for name, profit, hands in stats['top_10_players']:
            print(f"  {name:20s}: {profit:+10,} chips ({hands:,} hands)")
        
        print("\nBottom 10 Players (by net profit):")
        for name, profit, hands in stats['bottom_10_players']:
            print(f"  {name:20s}: {profit:+10,} chips ({hands:,} hands)")
