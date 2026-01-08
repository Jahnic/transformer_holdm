# Poker Transformer: From Imitation to Optimization

A complete educational journey through modern AI: build a transformer from scratch, train it on human poker data, then improve it beyond human play with reinforcement learning.

## The Learning Arc

```
Part 1: Transformer Fundamentals ←── You are here
    │
    │   Build every component from scratch
    │   Deep understanding of attention, embeddings, training
    │
    ▼
Part 2: Supervised Learning (Stage 1)
    │
    │   Train on human poker data (IRC database)
    │   Learn to predict human-like actions
    │   Filter for winning players
    │
    ▼
Part 3: Reinforcement Learning (Stage 2)
    │
    │   Self-play fine-tuning
    │   Exceed human performance
    │   Learn optimal strategy
    │
    ▼
Result: Complete understanding of modern AI pipelines
```

## Quick Start

```bash
# 1. Setup environment
chmod +x setup.sh && ./setup.sh
source venv/bin/activate

# 2. Get IRC Poker Data
#    - Download from https://poker.cs.ualberta.ca/irc_poker_database.html
#    - Get the holdempot.*.tgz files (Texas Hold'em)
#    - Extract to data/raw/holdem/

# 3. Run Phase 0 Analysis (understand your data first!)
python src/phase0_analysis.py data/raw/holdem

# 4. Review output and proceed to preprocessing
```

## Part 1 Phases

| Phase | Focus | Key Insight |
|-------|-------|-------------|
| **0** | Data | Understand IRC format, analyze player skill, decide filtering |
| **1** | Embeddings | Tokens become vectors in learnable space |
| **2** | Single-head attention | The "where to look" mechanism |
| **3** | Multi-head attention | Different heads learn different patterns |
| **4** | Feed-forward | Where computation/transformation happens |
| **5** | Residuals + LayerNorm | Why deep networks train at all |
| **6** | Full transformer | Assemble everything + action head |
| **7** | Ablations | Break things to understand them |

## IRC Data Structure

The IRC Poker Database uses three file types that must be joined:

### `hdb` - Hand Database (one line per hand)
```
1001911159   1  2524  2  2/540   2/540   2/1620   1/3240   Th Jh 7c 5h As
│            │  │     │  │       │       │        │        └─ Board cards
│            │  │     │  │       │       │        └─ winners/total_pot
│            │  │     │  │       │       └─ turn: players/pot
│            │  │     │  │       └─ flop: players/pot
│            │  │     │  └─ preflop: players/pot
│            │  │     └─ num_players
│            │  └─ hand_number
│            └─ table_id
└─ timestamp (unique hand ID)
```

### `hroster` - Hand Roster (who played each hand)
```
1001911628  4 DopeyTwat KVIETYS ein is314onu
│           │ └─ players in seat order
│           └─ num_players
└─ timestamp
```

### `pdb.{Player}` - Player Database (each player's actions)
```
KVIETYS   1001911159  2  2 Brc k     c     b        51140 2430 3240 Kd Jh
│         │           │  │ │   │     │     │        │     │    │    └─ hole cards (if shown)
│         │           │  │ │   │     │     │        │     │    └─ chips won
│         │           │  │ │   │     │     │        │     └─ chips invested
│         │           │  │ │   │     │     │        └─ stack after
│         │           │  │ │   │     │     └─ river actions
│         │           │  │ │   │     └─ turn actions
│         │           │  │ │   └─ flop actions
│         │           │  │ └─ preflop actions
│         │           │  └─ seat position
│         │           └─ num_players
│         └─ timestamp
└─ player_name
```

### Action Codes
| Code | Meaning |
|------|---------|
| B | Posted big blind |
| b | Bet |
| c | Call |
| r | Raise |
| k | Check |
| f | Fold |
| - | Didn't act (folded earlier) |

## Project Structure

```
poker_transformer/
├── data/
│   ├── raw/holdem/          # IRC data goes here
│   │   ├── hdb
│   │   ├── hroster
│   │   └── pdb/
│   │       ├── pdb.Player1
│   │       └── ...
│   └── processed/           # Generated training data
├── src/
│   ├── irc_parser.py        # Parse IRC format
│   ├── phase0_analysis.py   # Analyze data, player skill
│   ├── tokenizer.py         # Vocabulary and encoding
│   ├── dataset.py           # PyTorch Dataset
│   ├── model.py             # Transformer (built incrementally)
│   └── train.py             # Training loop
├── checkpoints/             # Saved models
├── outputs/                 # Generated analysis
├── requirements.txt
├── setup.sh
└── README.md
```

## Key Design Decisions

### 1. Action Types (not sizing)
Since IRC data is fixed-limit poker, we predict action types only:
- Fold, Check, Call, Bet, Raise

Sizing can be added in Stage 2 (RL) if desired.

### 2. Player Filtering
Only train on hands from winning players to learn good patterns.
Run `phase0_analysis.py` to see skill distribution and decide threshold.

### 3. Unknown Hole Cards
~88% of hands lack hole card info (player folded or won without showdown).
We use `<UNK>` token for unknown cards — model still learns from action patterns.

### 4. Table Size
Focus on 6-max (2-6 players) for cleaner patterns.

## Hardware

Developed for Apple Silicon (M4 Pro Mac Mini).
- MPS acceleration for PyTorch
- Stage 1: ~2-4 hours training
- Stage 2: ~8-24 hours (can stop early)
# transformer_holdm
