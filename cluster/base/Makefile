# See existing images with
#     gcloud container images list-tags gcr.io/$PROJECT/cc-base
#
# Usage:
#	  VERSION_TAG=0.XY make target

base-image:
		mkdir -p staging/cc/tensorflow
		cp ../../WORKSPACE staging/
		cp ../../*.bazelrc staging/
		cp ../../requirements.txt staging/
		cp -r -p ../../cc/ staging/
		rm -rfd staging/cc/tensorflow
		docker build -t "gcr.io/${PROJECT}/cc-base:${VERSION_TAG}" .
		rm -rfd staging/

base-push: base-image
		gcloud docker -- push "gcr.io/${PROJECT}/cc-base:${VERSION_TAG}"

.PHONY: base-image base-push
