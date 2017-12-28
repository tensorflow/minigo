VERSION=0.3
PROJECT=tensor-go


make gpu-image:
		mkdir -p staging/
		cp ../*.py staging/
		cp ../requirements.txt staging/
		cp player_wrapper.sh staging/
	    docker build -f Dockerfile.gpu -t "gcr.io/$(PROJECT)/mugo-gpu-player:$(VERSION)" .
		rm staging/*.py
		rm staging/requirements.txt
		rm staging/player_wrapper.sh

image:
		mkdir -p staging/
		cp ../*.py staging/
		cp ../requirements.txt staging/
		cp player_wrapper.sh staging/
	    docker build -t "gcr.io/$(PROJECT)/mugo-player:$(VERSION)" .
		rm staging/*.py
		rm staging/requirements.txt
		rm staging/player_wrapper.sh

image-clean:
		mkdir -p staging/
		cp ../*.py staging/
		cp ../requirements.txt staging/
		cp player_wrapper.sh staging/
	    docker build --no-cache -t "gcr.io/$(PROJECT)/mugo-player:$(VERSION)" .
		rm staging/*.py
		rm staging/requirements.txt
		rm staging/player_wrapper.sh

push: image
	    gcloud docker -- push "gcr.io/$(PROJECT)/mugo-player:$(VERSION)"

gpu-push: gpu-image
	    gcloud docker -- push "gcr.io/$(PROJECT)/mugo-gpu-player:$(VERSION)"

.PHONY: image push
