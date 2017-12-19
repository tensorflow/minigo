#!/bin/sh

# Added to the player image.
# Wraps our call to main.py

set -e

BUCKET=$1
DATA_DIR=/gcs

mkdir $DATA_DIR

echo creds: $GOOGLE_APPLICATION_CREDENTIALS
echo bucket: $BUCKET
echo data dir: $DATA_DIR

gcsfuse --implicit-dirs --key-file $GOOGLE_APPLICATION_CREDENTIALS $BUCKET $DATA_DIR

echo Using bucket $BUCKET
echo Finding newest model...
MODEL=$(ls $DATA_DIR/models | tail -n1)
MODEL=${MODEL##*/}
MODEL=${MODEL%.meta}

echo $MODEL

READOUTS=900
# Do adaptive readout based on MODEL number here.

echo Copying model files...
cp -v $DATA_DIR/models/$MODEL.* .
echo Model copy complete.

OUT_DIR=$DATA_DIR/games/$MODEL/$(hostname)
SGF_DIR=$DATA_DIR/sgf/$MODEL/$(hostname)
mkdir -p $OUT_DIR
mkdir -p $SGF_DIR

echo Playing a game using model $MODEL...
echo ...writing out game results to $OUT_DIR
echo ...and sgfs to $SGF_DIR

python3 main.py selfplay $MODEL --output-dir=$OUT_DIR --output-sgf=$SGF_DIR --readouts $READOUTS -v 2 -g 8 --resign-threshold 0.90
echo Finished a set of games!
