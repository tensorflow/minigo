# Usage:
# Set environment variables as follows:
#   - MODEL:  the basename of the .pb of the model to play (e.g. 000303-olympus.pb)
#   - MODEL_NUM: the number of the model, e.g. '303'.  This is used to set the name of the image, the username on CGOS, and the name of the pod.
#   - VERSION_TAG: as in the other cluster/ dockerfiles.
cgos-image:
	  docker build --build-arg MODEL=${MODEL} --build-arg MODEL_NUM=${MODEL_NUM} -t "gcr.io/tensor-go/cgos-player-${MODEL_NUM}:${VERSION_TAG}" .

cgos-push: cgos-image
		gcloud docker -- push "gcr.io/tensor-go/cgos-player-${MODEL_NUM}:${VERSION_TAG}"

.PHONY: cgos-image cgos-push
