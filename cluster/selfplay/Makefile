# Version and project can be overridden with the following:
#  make target VERSION_TAG=0.XY PROJECT=foo-project

gpu-image:
		mkdir -p staging
		cp ../../*.py staging/
		cp ../../requirements.txt staging/
		cp player_wrapper.sh staging/
	    docker build -f Dockerfile.gpu -t "gcr.io/$(PROJECT)/$(GPU_PLAYER_CONTAINER):$(VERSION_TAG)" .
		rm staging/*.py
		rm staging/requirements.txt
		rm staging/player_wrapper.sh

cpu-image:
		mkdir -p staging
		cp ../../*.py staging/
		cp ../../requirements.txt staging/
		cp player_wrapper.sh staging/
	    docker build -t "gcr.io/$(PROJECT)/$(CPU_PLAYER_CONTAINER):$(VERSION_TAG)" .
		rm staging/*.py
		rm staging/requirements.txt
		rm staging/player_wrapper.sh

cpu-image-clean:
		mkdir -p staging/
		cp ../../*.py staging/
		cp ../../requirements.txt staging/
		cp player_wrapper.sh staging/
	    docker build --no-cache -t "gcr.io/$(PROJECT)/$(CPU_PLAYER_CONTAINER):$(VERSION_TAG)" .
		rm staging/*.py
		rm staging/requirements.txt
		rm staging/player_wrapper.sh

cc-image:
		mkdir -p staging/cc
		cp ../../requirements.txt staging/
		cp ../../WORKSPACE staging/
		cp cc_wrapper.sh staging/
		cp -r ../../cc/* staging/cc/
	    docker build -f cc-Dockerfile -t "gcr.io/$(PROJECT)/$(CC_PLAYER_CONTAINER):$(VERSION_TAG)" .
		rm -rfd staging/


cpu-push: cpu-image
	    gcloud docker -- push "gcr.io/$(PROJECT)/$(CPU_PLAYER_CONTAINER):$(VERSION_TAG)"

gpu-push: gpu-image
	    gcloud docker -- push "gcr.io/$(PROJECT)/$(GPU_PLAYER_CONTAINER):$(VERSION_TAG)"

.PHONY: cpu-image image-push
