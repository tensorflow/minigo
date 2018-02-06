# Version and project can be overridden with the following:
#  make target VERSION=0.XY PROJECT=foo-project

VERSION_TAG?=0.13
PROJECT?=minigo-pub
GPU_PLAYER_CONTAINER?=minigo-gpu-player
CPU_PLAYER_CONTAINER?=minigo-player

make echo:
		echo $(VERSION)
		echo $(PROJECT)

make gpu-image:
		mkdir -p staging
		cp ../*.py staging/
		cp ../requirements.txt staging/
		cp player_wrapper.sh staging/
	    docker build -f Dockerfile.gpu -t "gcr.io/$(PROJECT)/$(GPU_PLAYER_CONTAINER):$(VERSION_TAG)" .
		rm staging/*.py
		rm staging/requirements.txt
		rm staging/player_wrapper.sh

image:
		mkdir -p staging
		cp ../*.py staging/
		cp ../requirements-cpu.txt staging/
		cp player_wrapper.sh staging/
	    docker build -t "gcr.io/$(PROJECT)/$(CPU_PLAYER_CONTAINER):$(VERSION_TAG)" .
		rm staging/*.py
		rm staging/requirements-cpu.txt
		rm staging/player_wrapper.sh

image-clean:
		mkdir -p staging/
		cp ../*.py staging/
		cp ../requirements-cpu.txt staging/
		cp player_wrapper.sh staging/
	    docker build --no-cache -t "gcr.io/$(PROJECT)/$(CPU_PLAYER_CONTAINER):$(VERSION_TAG)" .
		rm staging/*.py
		rm staging/requirements-cpu.txt
		rm staging/player_wrapper.sh

push: image
	    gcloud docker -- push "gcr.io/$(PROJECT)/$(CPU_PLAYER_CONTAINER):$(VERSION_TAG)"

gpu-push: gpu-image
	    gcloud docker -- push "gcr.io/$(PROJECT)/$(GPU_PLAYER_CONTAINER):$(VERSION_TAG)"

.PHONY: image push
