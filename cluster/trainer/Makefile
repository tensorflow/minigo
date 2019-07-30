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


image:
	$(staging)
	docker build --quiet --build-arg PROJECT=$(PROJECT) -f Dockerfile -t "gcr.io/$(PROJECT)/minigo-tpu-trainer:$(VERSION_TAG)" .

push: image
	gcloud docker --verbosity=error -- push "gcr.io/$(PROJECT)/minigo-tpu-trainer:$(VERSION_TAG)"

clean:
	rm -rfd staging

.PHONY: image push clean
