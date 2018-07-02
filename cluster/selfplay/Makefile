# See existing images with
#     gcloud container images list
#     gcloud container images list-tags gcr.io/$PROJECT/<PLAYER_CONTAINER>
#
# Usage:
# Set environment variables as follows:
#   - CPU_PLAYER_CONTAINER/GPU_PLAYER_CONTAINER: name of the cpu/gpu player container (e.g. minigo-gpu-player)
#   - VERSION_TAG: as in the other cluster/ dockerfiles.
#
#  	  VERSION_TAG=0.XY make target

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
	  docker build -f Dockerfile-cc -t "gcr.io/$(PROJECT)/minigo-cc-player:$(VERSION_TAG)" .

cc-push: cc-image
		gcloud docker -- push "gcr.io/$(PROJECT)/minigo-cc-player:$(VERSION_TAG)"

cpu-push: cpu-image
		gcloud docker -- push "gcr.io/$(PROJECT)/$(CPU_PLAYER_CONTAINER):$(VERSION_TAG)"

gpu-push: gpu-image
		gcloud docker -- push "gcr.io/$(PROJECT)/$(GPU_PLAYER_CONTAINER):$(VERSION_TAG)"

.PHONY: cpu-image image-push
