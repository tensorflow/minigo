#!/bin/sh

# Added to the player image.
# Wraps our call to main.py

set -e

echo creds: $GOOGLE_APPLICATION_CREDENTIALS
echo bucket: $BUCKET_NAME
echo board_size: $BOARD_SIZE
echo parallel games: $GAMES

python3 rl_loop.py selfplay \
  --resign-threshold=0.91

echo Finished a set of games!
