# Version and project can be overridden with the following:
#  make target VERSION_TAG=0.XY PROJECT=foo-project

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

.PHONY: gpu-image gpu-push
