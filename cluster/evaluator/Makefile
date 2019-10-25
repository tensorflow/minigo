# See existing images with
#     gcloud container images list-tags gcr.io/$PROJECT/minigo-cc-evaluator
#
# Usage:
#     VERSION_TAG=0.XY make target

py-image:
		mkdir -p staging
		cp ../../*.py staging/
		cp -r -p ../../rl_loop/ staging/
		cp ../../requirements.txt staging/
		cp evaluator_py_wrapper.sh staging/
	  docker build -f Dockerfile-py -t "gcr.io/$(PROJECT)/minigo-py-evaluator:$(VERSION_TAG)" .
		rm -rfd staging/

py-push: py-image
	    gcloud docker --verbosity=error -- push "gcr.io/$(PROJECT)/minigo-py-evaluator:$(VERSION_TAG)"

cc-image:
		mkdir -p staging/cc
		cp ../../WORKSPACE staging/
		cp ../../.bazelrc staging/
		cp ../../mask_flags.py staging/
		cp evaluator_cc_wrapper.sh staging/
		cp -r -p ../../cc/ staging/
		rm -rfd staging/cc/tensorflow
	  docker build -f Dockerfile-cc --build-arg project=$(PROJECT) -t "gcr.io/$(PROJECT)/minigo-cc-evaluator:$(VERSION_TAG)" .
		rm -rfd staging/

cc-push: cc-image
	    gcloud docker --verbosity=error -- push "gcr.io/$(PROJECT)/minigo-cc-evaluator:$(VERSION_TAG)"

ringmaster-image:
		cp evaluator_ringmaster_wrapper.py staging/
	  docker build -f Dockerfile-ringmaster --build-arg project=$(PROJECT) -t "gcr.io/$(PROJECT)/minigo-ringmaster-evaluator:$(VERSION_TAG)" .
		rm -rfd staging/

ringmaster-push: cc-image
		mkdir -p staging
		cp ../../bigtable_input.py staging/
	    gcloud docker --verbosity=error -- push "gcr.io/$(PROJECT)/minigo-ringmaster-evaluator:$(VERSION_TAG)"

clean:
		rm -rfd staging/
.PHONY: py-image py-push cc-image cc-push ringmaster-image ringmaster-push clean
