# See existing images with
#     gcloud container images list-tags gcr.io/$PROJECT/cgos-player
#
# Usage:
# Set environment variables as follows:
#   - MODEL: the basename of the .pb of the model to play (e.g. 000303-olympus.pb)
#   - MODEL_NUM: the number of the model, e.g. '303'. This is used in the image name, CGOS username, and pod name.
#   - VERSION_TAG: as in the other cluster/ dockerfiles.
#
#     VERSION_TAG=0.XY MODEL=000123-foo.pb MODEL_NUM=123 make target

cgos-image:
	  docker build --build-arg MODEL=${MODEL} --build-arg MODEL_NUM=${MODEL_NUM} -t "gcr.io/${PROJECT}/cgos-player-${MODEL_NUM}:${VERSION_TAG}" .

cgos-push: cgos-image
		gcloud docker -- push "gcr.io/${PROJECT}/cgos-player-${MODEL_NUM}:${VERSION_TAG}"

.PHONY: cgos-image cgos-push
