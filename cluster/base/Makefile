# Version and project can be overridden with the following:
# VERSION_TAG=0.XY make target

base-image:
		mkdir -p staging/cc
		cp ../../WORKSPACE staging/
		cp -r -p ../../cc/ staging/
		rm -rfd staging/cc/tensorflow
	  docker build -t "gcr.io/tensor-go/cc-base:latest" .
		rm -rfd staging/

base-push: base-image
		gcloud docker -- push "gcr.io/tensor-go/cc-base:${VERSION_TAG}"

.PHONY: base-image base-push
