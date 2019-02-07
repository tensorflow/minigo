
gtp:
		mkdir -p staging/cc
		cp ../../WORKSPACE staging/
		cp -r -p ../../cc/ staging/
		cp p100-lz-tuning staging/
		rm -rfd staging/cc/tensorflow
		docker build --build-arg PROJECT=${PROJECT} -f mggtp-Dockerfile -t "gcr.io/$(PROJECT)/mg-gtp:$(VERSION_TAG)" .

ring:
		mkdir -p staging/cc
		cp ../../WORKSPACE staging/
		cp -r -p ../../cc/ staging/
		cp p100-lz-tuning staging/
		cp ringmaster_wrapper.sh staging/
		rm -rfd staging/cc/tensorflow
		docker build --build-arg PROJECT=${PROJECT} -f lz-Dockerfile -t "gcr.io/$(PROJECT)/mg-ring:$(VERSION_TAG)" .


gtp-push: gtp
	gcloud docker --verbosity=error -- push "gcr.io/$(PROJECT)/mg-gtp:$(VERSION_TAG)"

ring-push: ring
	gcloud docker --verbosity=error -- push "gcr.io/$(PROJECT)/mg-ring:$(VERSION_TAG)"

clean:
		rm -rfd staging/
.PHONY: gtp gtp-push clean
