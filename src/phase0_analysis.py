"""
Phase 0: IRC Poker Data Analysis

This script performs comprehensive analysis of the IRC poker database to:
1. Understand data volume and structure
2. Analyze player skill distribution
3. Determine filtering thresholds
4. Examine action distributions
5. Guide preprocessing decisions

Run with: python src/phase0_analysis.py <path_to_holdem_data>
"""

import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from irc_parser import IRCDataJoiner, HDBParser, PDBParser


def analyze_player_skill_distribution(joiner: IRCDataJoiner, min_hands: int = 50) -> Dict:
    """
    Detailed analysis of player skill levels.
    
    This helps determine:
    - What threshold to use for filtering
    - How much data we'll have after filtering
    - Whether there's clear separation between good and bad players
    """
    print("\n" + "=" * 60)
    print("PLAYER SKILL ANALYSIS")
    print("=" * 60)
    
    rankings = joiner.get_player_rankings(min_hands=min_hands)
    
    if not rankings:
        print(f"No players found with {min_hands}+ hands")
        return {}
    
    # Calculate percentiles
    profits = [r[1]['net_profit'] for r in rankings]
    hands = [r[1]['hands_played'] for r in rankings]
    
    import statistics
    
    results = {
        'total_players': len(rankings),
        'min_hands_threshold': min_hands,
    }
    
    # Profit distribution
    print(f"\nPlayers with {min_hands}+ hands: {len(rankings)}")
    print(f"\nProfit Distribution:")
    print(f"  Min:    {min(profits):+,}")
    print(f"  25th %: {sorted(profits)[len(profits)//4]:+,}")
    print(f"  Median: {statistics.median(profits):+,.0f}")
    print(f"  75th %: {sorted(profits)[3*len(profits)//4]:+,}")
    print(f"  Max:    {max(profits):+,}")
    print(f"  Mean:   {statistics.mean(profits):+,.0f}")
    print(f"  StdDev: {statistics.stdev(profits):,.0f}")
    
    # Win/loss breakdown
    profitable = [(name, stats) for name, stats in rankings if stats['net_profit'] > 0]
    breakeven = [(name, stats) for name, stats in rankings if stats['net_profit'] == 0]
    losing = [(name, stats) for name, stats in rankings if stats['net_profit'] < 0]
    
    print(f"\nPlayer Categories:")
    print(f"  Profitable (net > 0):  {len(profitable):4} ({100*len(profitable)/len(rankings):5.1f}%)")
    print(f"  Break-even (net = 0):  {len(breakeven):4} ({100*len(breakeven)/len(rankings):5.1f}%)")
    print(f"  Losing (net < 0):      {len(losing):4} ({100*len(losing)/len(rankings):5.1f}%)")
    
    results['profitable_count'] = len(profitable)
    results['losing_count'] = len(losing)
    results['profitable_pct'] = 100 * len(profitable) / len(rankings)
    
    # Hands from each category
    profitable_hands = sum(s['hands_played'] for _, s in profitable)
    losing_hands = sum(s['hands_played'] for _, s in losing)
    
    print(f"\nHands by Player Category:")
    print(f"  From profitable players: {profitable_hands:,}")
    print(f"  From losing players:     {losing_hands:,}")
    
    results['profitable_hands'] = profitable_hands
    results['losing_hands'] = losing_hands
    
    # Top percentile analysis
    print(f"\n" + "-" * 40)
    print("FILTERING ANALYSIS: What do we get at each threshold?")
    print("-" * 40)
    
    for percentile in [10, 20, 30, 50]:
        cutoff_idx = len(rankings) * percentile // 100
        top_players = rankings[:cutoff_idx]
        
        top_hands = sum(s['hands_played'] for _, s in top_players)
        top_profit = sum(s['net_profit'] for _, s in top_players)
        
        if top_players:
            worst_in_top = top_players[-1][1]['net_profit']
            avg_profit = top_profit / len(top_players)
            avg_hands = top_hands / len(top_players)
        else:
            worst_in_top = 0
            avg_profit = 0
            avg_hands = 0
        
        print(f"\n  Top {percentile}% ({len(top_players)} players):")
        print(f"    Total hands:     {top_hands:,}")
        print(f"    Avg hands/player: {avg_hands:,.0f}")
        print(f"    Min profit to qualify: {worst_in_top:+,}")
        print(f"    Avg profit:      {avg_profit:+,.0f}")
        
        results[f'top_{percentile}_players'] = len(top_players)
        results[f'top_{percentile}_hands'] = top_hands
        results[f'top_{percentile}_min_profit'] = worst_in_top
    
    # Playing style analysis (for top 30%)
    print(f"\n" + "-" * 40)
    print("PLAYING STYLE ANALYSIS (Top 30% players)")
    print("-" * 40)
    
    top_30 = rankings[:len(rankings) * 30 // 100]
    bottom_30 = rankings[len(rankings) * 70 // 100:]
    
    if top_30 and bottom_30:
        top_vpip = statistics.mean(s['vpip'] for _, s in top_30)
        top_pfr = statistics.mean(s['pfr'] for _, s in top_30)
        bottom_vpip = statistics.mean(s['vpip'] for _, s in bottom_30)
        bottom_pfr = statistics.mean(s['pfr'] for _, s in bottom_30)
        
        print(f"\n  VPIP (Voluntarily Put money In Pot):")
        print(f"    Top 30%:    {100*top_vpip:.1f}%")
        print(f"    Bottom 30%: {100*bottom_vpip:.1f}%")
        
        print(f"\n  PFR (Pre-Flop Raise rate):")
        print(f"    Top 30%:    {100*top_pfr:.1f}%")
        print(f"    Bottom 30%: {100*bottom_pfr:.1f}%")
        
        results['top30_vpip'] = top_vpip
        results['top30_pfr'] = top_pfr
        results['bottom30_vpip'] = bottom_vpip
        results['bottom30_pfr'] = bottom_pfr
    
    return results


def analyze_action_distribution(joiner: IRCDataJoiner) -> Dict:
    """
    Analyze the distribution of actions in the dataset.
    """
    print("\n" + "=" * 60)
    print("ACTION DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    action_counts = {
        'preflop': Counter(),
        'flop': Counter(),
        'turn': Counter(),
        'river': Counter(),
    }
    
    total_hands = 0
    hands_with_cards = 0
    
    for timestamp, player_hands in joiner.player_hands.items():
        for player_name, ph in player_hands.items():
            total_hands += 1
            
            if ph.hole_cards:
                hands_with_cards += 1
            
            # Count individual actions
            for action in ph.preflop_actions:
                if action not in ['-', ' ']:
                    action_counts['preflop'][action] += 1
            
            for action in ph.flop_actions:
                if action not in ['-', ' ']:
                    action_counts['flop'][action] += 1
            
            for action in ph.turn_actions:
                if action not in ['-', ' ']:
                    action_counts['turn'][action] += 1
            
            for action in ph.river_actions:
                if action not in ['-', ' ']:
                    action_counts['river'][action] += 1
    
    # Action code meanings
    action_names = {
        'B': 'Big blind post',
        'b': 'Bet',
        'c': 'Call',
        'r': 'Raise',
        'k': 'Check',
        'f': 'Fold',
        'Q': 'Quit/DC',
        'A': 'All-in',
    }
    
    print(f"\nTotal player-hand records: {total_hands:,}")
    print(f"Records with known hole cards: {hands_with_cards:,} ({100*hands_with_cards/total_hands:.1f}%)")
    
    results = {
        'total_records': total_hands,
        'records_with_cards': hands_with_cards,
        'cards_known_pct': 100 * hands_with_cards / total_hands if total_hands > 0 else 0,
    }
    
    for street in ['preflop', 'flop', 'turn', 'river']:
        print(f"\n{street.upper()} Actions:")
        total = sum(action_counts[street].values())
        for action, count in action_counts[street].most_common():
            name = action_names.get(action, f'Unknown({action})')
            pct = 100 * count / total if total > 0 else 0
            print(f"  {action} ({name:15s}): {count:8,} ({pct:5.1f}%)")
        
        results[f'{street}_actions'] = dict(action_counts[street])
    
    return results


def analyze_hand_structure(joiner: IRCDataJoiner) -> Dict:
    """
    Analyze hand structure: how many reach each street, pot sizes, etc.
    """
    print("\n" + "=" * 60)
    print("HAND STRUCTURE ANALYSIS")
    print("=" * 60)
    
    street_counts = {
        'preflop_only': 0,
        'to_flop': 0,
        'to_turn': 0,
        'to_river': 0,
    }
    
    pot_sizes = []
    player_counts = Counter()
    
    for timestamp, metadata in joiner.hand_metadata.items():
        player_counts[metadata.num_players] += 1
        
        if metadata.flop_players == 0:
            street_counts['preflop_only'] += 1
        elif metadata.turn_players == 0:
            street_counts['to_flop'] += 1
        elif metadata.river_players == 0:
            street_counts['to_turn'] += 1
        else:
            street_counts['to_river'] += 1
        
        if metadata.total_pot > 0:
            pot_sizes.append(metadata.total_pot)
    
    total = len(joiner.hand_metadata)
    
    print(f"\nHands ending on each street:")
    print(f"  Preflop (no flop):   {street_counts['preflop_only']:,} ({100*street_counts['preflop_only']/total:.1f}%)")
    print(f"  Flop (no turn):      {street_counts['to_flop']:,} ({100*street_counts['to_flop']/total:.1f}%)")
    print(f"  Turn (no river):     {street_counts['to_turn']:,} ({100*street_counts['to_turn']/total:.1f}%)")
    print(f"  River/Showdown:      {street_counts['to_river']:,} ({100*street_counts['to_river']/total:.1f}%)")
    
    print(f"\nHands by player count:")
    for n, count in sorted(player_counts.items()):
        print(f"  {n} players: {count:,} ({100*count/total:.1f}%)")
    
    if pot_sizes:
        import statistics
        print(f"\nPot sizes:")
        print(f"  Min:    {min(pot_sizes):,}")
        print(f"  Median: {statistics.median(pot_sizes):,.0f}")
        print(f"  Max:    {max(pot_sizes):,}")
        print(f"  Mean:   {statistics.mean(pot_sizes):,.0f}")
    
    return {
        'street_counts': street_counts,
        'player_counts': dict(player_counts),
    }


def generate_recommendations(skill_results: Dict, action_results: Dict) -> None:
    """
    Generate recommendations for data preprocessing based on analysis.
    """
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    print("\n1. PLAYER FILTERING")
    print("-" * 40)
    
    if 'top_30_hands' in skill_results and 'top_20_hands' in skill_results:
        print(f"   Option A: Top 20% players")
        print(f"             → {skill_results['top_20_hands']:,} hands")
        print(f"             → Higher quality, less data")
        print()
        print(f"   Option B: Top 30% players")
        print(f"             → {skill_results['top_30_hands']:,} hands")
        print(f"             → Good balance of quality and quantity")
        print()
        print(f"   Option C: All profitable players")
        print(f"             → {skill_results.get('profitable_hands', 'N/A'):,} hands")
        print(f"             → Maximum data from non-losing players")
        print()
        print("   RECOMMENDATION: Start with Top 30%, adjust if needed.")
    
    print("\n2. HOLE CARDS")
    print("-" * 40)
    cards_pct = action_results.get('cards_known_pct', 0)
    print(f"   Only {cards_pct:.1f}% of hands have known hole cards.")
    print()
    print("   Option A: Use only hands with known cards")
    print("             → Much less data, but cleaner signal")
    print()
    print("   Option B: Use all hands with <UNK> token for unknown cards")
    print("             → More data, model learns position/action patterns too")
    print()
    print("   RECOMMENDATION: Option B (use all hands with <UNK> token)")
    print("   Rationale: Position and action patterns are still valuable")
    
    print("\n3. TABLE SIZE")
    print("-" * 40)
    print("   The data includes 2-10 player tables.")
    print()
    print("   Option A: Train on all table sizes")
    print("             → More data, but model must generalize")
    print()
    print("   Option B: Focus on 6-max (2-6 players)")
    print("             → Most common format, cleaner patterns")
    print()
    print("   RECOMMENDATION: Option B (6-max only)")
    print("   Rationale: Reduces complexity, still plenty of data")


def main():
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = Path(__file__).parent.parent / 'data' / 'raw' / 'holdem'
    
    if not Path(data_dir).exists():
        print(f"Data directory not found: {data_dir}")
        print("Usage: python phase0_analysis.py <path_to_holdem_data>")
        sys.exit(1)
    
    print("=" * 60)
    print("PHASE 0: IRC POKER DATA ANALYSIS")
    print("=" * 60)
    
    # Load data
    joiner = IRCDataJoiner(str(data_dir))
    joiner.load_all()
    
    # Run analyses
    skill_results = analyze_player_skill_distribution(joiner, min_hands=50)
    action_results = analyze_action_distribution(joiner)
    hand_results = analyze_hand_structure(joiner)
    
    # Generate recommendations
    generate_recommendations(skill_results, action_results)
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        'skill': skill_results,
        'actions': action_results,
        'structure': hand_results,
    }
    
    with open(output_dir / 'phase0_analysis.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n\nResults saved to: {output_dir / 'phase0_analysis.json'}")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Review the analysis above
2. Decide on filtering thresholds (player skill, table size)
3. Run the preprocessor to generate training data
4. Proceed to Phase 1 (Embeddings)
""")


if __name__ == '__main__':
    main()
