#!/bin/bash

set -e

: ${RINGMASTER_CONTROL_PATH?"Need to set RINGMASTER_CONTROL_PATH"}
: ${OUT_PATH?"Need to set OUT_PATH"}
: ${MODEL_ONE?"Need to set MODEL_ONE"}
: ${MODEL_TWO?"Need to set MODEL_TWO"}

gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
gsutil cp $RINGMASTER_CONTROL_PATH .
gsutil cp $MODEL_ONE .
gsutil cp $MODEL_TWO .


RING_BASENAME=`basename $RINGMASTER_CONTROL_PATH`

# if your control file doesn't end in .ctl, things are bad.
RING_FILES=`basename $RINGMASTER_CONTROL_PATH .ctl`

date
echo "Running Ringmaster: $RING_BASENAME"

#/mg_venv/bin/ringmaster $RING_BASENAME check
/mg_venv/bin/ringmaster $RING_BASENAME run

echo "Ringmaster all done"
POD_NAME=`hostname | rev | cut -d'-' -f 1 | rev`

gsutil -m cp -r $RING_FILES.* $OUT_PATH/$POD_NAME/

