# Version and project can be overridden with the following:
#  make target VERSION=0.XY PROJECT=foo-project

VERSION?=0.12
PROJECT?=minigo-pub

make echo:
		echo $(VERSION)
		echo $(PROJECT)

make gpu-image:
		cp ../*.py staging/
		cp ../requirements-gpu.txt staging/
		cp player_wrapper.sh staging/
	    docker build -f Dockerfile.gpu -t "gcr.io/$(PROJECT)/minigo-gpu-player:$(VERSION)" .
		rm staging/*.py
		rm staging/requirements-gpu.txt
		rm staging/player_wrapper.sh

image:
		cp ../*.py staging/
		cp ../requirements-cpu.txt staging/
		cp player_wrapper.sh staging/
	    docker build -t "gcr.io/$(PROJECT)/minigo-player:$(VERSION)" .
		rm staging/*.py
		rm staging/requirements-cpu.txt
		rm staging/player_wrapper.sh

image-clean:
		mkdir -p staging/
		cp ../*.py staging/
		cp ../requirements-cpu.txt staging/
		cp player_wrapper.sh staging/
	    docker build --no-cache -t "gcr.io/$(PROJECT)/minigo-player:$(VERSION)" .
		rm staging/*.py
		rm staging/requirements-cpu.txt
		rm staging/player_wrapper.sh

push: image
	    gcloud docker -- push "gcr.io/$(PROJECT)/minigo-player:$(VERSION)"

gpu-push: gpu-image
	    gcloud docker -- push "gcr.io/$(PROJECT)/minigo-gpu-player:$(VERSION)"

.PHONY: image push
