# See existing images with
#     gcloud container images list-tags gcr.io/$PROJECT/minigo-cc-evaluator
#
# Usage:
#     VERSION_TAG=0.XY make target

gpu-image:
		mkdir -p staging
		cp ../../*.py staging/
		cp ../../requirements.txt staging/
		cp evaluator_wrapper.sh staging/
	    docker build -f Dockerfile.gpu -t "gcr.io/$(PROJECT)/$(GPU_EVALUATOR_CONTAINER):$(VERSION_TAG)" .
		rm staging/*.py
		rm staging/requirements.txt
		rm staging/evaluator_wrapper.sh

gpu-push: gpu-image
	    gcloud docker -- push "gcr.io/$(PROJECT)/$(GPU_EVALUATOR_CONTAINER):$(VERSION_TAG)"

cc-image:
		mkdir -p staging/cc
		cp ../../WORKSPACE staging/
		cp evaluator_cc_wrapper.sh staging/
		cp -r ../../cc/* staging/cc/
	    docker build -f Dockerfile-cc -t "gcr.io/$(PROJECT)/minigo-cc-evaluator:$(VERSION_TAG)" .
		rm -rfd staging/

cc-push: cc-image
	    gcloud docker -- push "gcr.io/$(PROJECT)/minigo-cc-evaluator:$(VERSION_TAG)"

.PHONY: gpu-image gpu-push
