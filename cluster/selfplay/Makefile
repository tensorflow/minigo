# See existing images with
#     gcloud container images list
#
# Usage:
#  	  PROJECT=(gcp project id) VERSION_TAG=0.XY make target

define staging =
	mkdir -p staging
	cp ../../WORKSPACE staging/
	cp ../../.bazelrc staging/
	cp -r -p ../../rl_loop/ staging/
	cp -r -p ../../cc/ staging/
	rm -rfd staging/cc/tensorflow
	cp ../../*.py staging/
	cp ../../requirements.txt staging/
endef


tpu-image:
	$(staging)
	docker build --quiet --build-arg RUNMODE=$(RUNMODE) --build-arg PROJECT=$(PROJECT) -f Dockerfile-tpu -t "gcr.io/$(PROJECT)/minigo-tpu-player:$(VERSION_TAG)" .

py-image:
	$(staging)
	docker build --quiet -f Dockerfile-py -t "gcr.io/$(PROJECT)/minigo-py-player:$(VERSION_TAG)" .
	rm -rfd staging

cc-image:
	$(staging)
	docker build --quiet --build-arg PROJECT=$(PROJECT) -f Dockerfile-cc -t "gcr.io/$(PROJECT)/minigo-cc-player:$(VERSION_TAG)" .

cc-push: cc-image
	gcloud docker --verbosity=error -- push "gcr.io/$(PROJECT)/minigo-cc-player:$(VERSION_TAG)"

py-push: py-image
	gcloud docker --verbosity=error -- push "gcr.io/$(PROJECT)/minigo-py-player:$(VERSION_TAG)"

tpu-push: tpu-image
	gcloud docker --verbosity=error -- push "gcr.io/$(PROJECT)/minigo-tpu-player:$(VERSION_TAG)"

clean:
	rm -rfd staging

.PHONY: cc-image cpu-image tpu-image gpu-image image-push cc-push cpu-push gpu-push tpu-push
