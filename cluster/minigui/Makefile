# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Reqires the following te be set:
#  BUCKET_NAME
#  GCS_DIR
#  MODEL
#  MODEL_DIR
#  BOARD_SIZE

build: check-env
		mkdir -p staging
		mkdir -p staging/minigui
		cp ../../*.py staging/
		cp ../../requirements.txt staging/
		cp -r ../../minigui staging
		cd staging/minigui
		tsc
		cd ../..
		docker build -t "gcr.io/$(PROJECT)/$(MINIGUI_PY_CPU_CONTAINER):$(VERSION_TAG)" .

clean: check-env
		rm staging/*.py
		rm staging/requirements.txt
		rm -r staging/minigui
		rmdir staging

build-clean: check-env build clean

check-env:
ifndef PROJECT
	$(error PROJECT is undefined)
endif
ifndef MINIGUI_PY_CPU_CONTAINER
	$(error MINIGUI_PY_CPU_CONTAINER is undefined)
endif
ifndef VERSION_TAG
	$(error VERSION_TAG is undefined)
endif

push: build-clean
		gcloud docker -- push "gcr.io/$(PROJECT)/$(MINIGUI_PY_CPU_CONTAINER):$(VERSION_TAG)"
