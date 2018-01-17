VERSION=0.9
PROJECT=tensor-go


make gpu-image:
		mkdir -p staging/data
		cp ../*.py staging/
		cp ../data/names.txt staging/data/
		cp ../requirements.txt staging/
		cp player_wrapper.sh staging/
	    docker build -f Dockerfile.gpu -t "gcr.io/$(PROJECT)/mugo-gpu-player:$(VERSION)" .
		rm staging/*.py
		rm staging/requirements.txt
		rm staging/player_wrapper.sh

image:
		mkdir -p staging/data
		cp ../*.py staging/
		cp -r ../data staging/
		cp ../requirements-cpu.txt staging/
		cp player_wrapper.sh staging/
	    docker build -t "gcr.io/$(PROJECT)/mugo-player:$(VERSION)" .
		rm staging/*.py
		rm -r staging/data
		rm staging/requirements-cpu.txt
		rm staging/player_wrapper.sh

image-clean:
		mkdir -p staging/
		cp ../*.py staging/
		cp ../requirements-cpu.txt staging/
		cp player_wrapper.sh staging/
	    docker build --no-cache -t "gcr.io/$(PROJECT)/mugo-player:$(VERSION)" .
		rm staging/*.py
		rm staging/requirements-cpu.txt
		rm staging/player_wrapper.sh

push: image
	    gcloud docker -- push "gcr.io/$(PROJECT)/mugo-player:$(VERSION)"

gpu-push: gpu-image
	    gcloud docker -- push "gcr.io/$(PROJECT)/mugo-gpu-player:$(VERSION)"

.PHONY: image push
