#!/bin/bash

# Chess data download script
# Downloads a subset of Lichess games for training

set -e

DATA_DIR="$(dirname "$0")"
RAW_DIR="$DATA_DIR/raw"
mkdir -p "$RAW_DIR"

echo "=== Chess Transformer Data Download ==="
echo ""

# Option 1: Download from Lichess API (smaller, filtered)
# This gets recent games from high-rated players

echo "Downloading sample games from Lichess..."
echo "This fetches ~10,000 games from 2200+ rated players."
echo ""

# We'll download games from several strong players
# Each API call gets up to 300 games, we'll get games from multiple sources

PLAYERS=(
    "DrNykterstein"      # Magnus Carlsen
    "Hikaru"             # Hikaru Nakamura  
    "GMWSO"              # Wesley So
    "FairChess_on_YouTube"
    "penguingm1"         # Andrew Tang
    "nihalsarin2004"     # Nihal Sarin
)

OUTPUT_FILE="$RAW_DIR/lichess_games.pgn"
> "$OUTPUT_FILE"  # Clear/create file

for player in "${PLAYERS[@]}"; do
    echo "Fetching games from $player..."
    curl -s "https://lichess.org/api/games/user/$player?max=500&rated=true&perfType=bullet,blitz,rapid,classical" \
        -H "Accept: application/x-chess-pgn" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    sleep 1  # Respect API rate limits
done

echo ""
echo "Download complete!"
echo "Games saved to: $OUTPUT_FILE"
echo ""

# Count games
GAME_COUNT=$(grep -c "^\[Event" "$OUTPUT_FILE" || echo "0")
echo "Total games downloaded: $GAME_COUNT"
echo ""

# If we need more data, provide instructions for bulk download
if [ "$GAME_COUNT" -lt 5000 ]; then
    echo "NOTE: For more data, you can download monthly databases from:"
    echo "https://database.lichess.org/"
    echo ""
    echo "Example:"
    echo "  curl -O https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst"
    echo "  zstd -d lichess_db_standard_rated_2024-01.pgn.zst"
fi
