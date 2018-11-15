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

define staging =
	mkdir -p staging
	cp ../../WORKSPACE staging/
	cp -r -p ../../rl_loop/ staging/
	cp -r -p ../../cc/ staging/
	cp -r -p ../../proto/ staging/
	rm -rfd staging/cc/tensorflow
	cp ../../*.py staging/
	cp ../../requirements.txt staging/
endef

tpu-image:
	$(staging)
	docker build -f Dockerfile.tpu -t "gcr.io/$(PROJECT)/minigo-tpu-player:$(VERSION_TAG)" .
	rm -rfd staging

py-image:
	$(staging)
	docker build -t "gcr.io/$(PROJECT)/$(CPU_PLAYER_CONTAINER):$(VERSION_TAG)" .
	rm -rfd staging

cc-image:
	$(staging)
	docker build -f Dockerfile-cc -t "gcr.io/$(PROJECT)/$(CC_PLAYER_CONTAINER):$(VERSION_TAG)" .

cc-push: cc-image
	gcloud docker -- push "gcr.io/$(PROJECT)/$(CC_PLAYER_CONTAINER):$(VERSION_TAG)"

py-push: py-image
	gcloud docker -- push "gcr.io/$(PROJECT)/$(CPU_PLAYER_CONTAINER):$(VERSION_TAG)"

tpu-push: tpu-image
	gcloud docker -- push "gcr.io/$(PROJECT)/minigo-tpu-player:$(VERSION_TAG)"

.PHONY: cc-image cpu-image tpu-image gpu-image image-push cc-push cpu-push gpu-push tpu-push
