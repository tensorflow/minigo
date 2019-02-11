#!/bin/bash

set -e

if [[ -z "$PROJECT" ]]; then
  echo 'PROJECT must be set to a nonempty string'
  exit 1;
fi

red=`tput setaf 1`
green=`tput setaf 2`
yellow=`tput setaf 3`
reset=`tput sgr0`

echo "${yellow}Building... ${reset}"

echo -e "\n\n${yellow}Base image ${reset}"
pushd base
VERSION_TAG="latest" make base-image
popd

echo -e "\n\n${yellow}Selfplay ${reset}"
pushd selfplay
VERSION_TAG="latest" make cc-image py-image
VERSION_TAG="latest" RUNMODE="tpu" make tpu-image
VERSION_TAG="latest_nr" RUNMODE="tpu_nr" make tpu-image
popd

echo -e "\n\n${yellow}Evaluator ${reset}"
pushd evaluator
VERSION_TAG="latest" make py-image cc-image
popd

echo -e "\n\n${yellow}Trainer ${reset}"
pushd trainer
VERSION_TAG="latest" make image
popd

echo -e "\n\n${yellow}Calibrator ${reset}"
pushd calibrator
VERSION_TAG="latest" make image
popd

echo "${green}Everything built?  Ok, pushing... ${reset}"

echo -e "\n\n${yellow}Base image ${reset}"
pushd base
VERSION_TAG="latest" make base-push
popd

echo -e "\n\n${yellow}Selfplay ${reset}"
pushd selfplay
VERSION_TAG="latest" make cc-push py-push
VERSION_TAG="latest" RUNMODE="tpu" make tpu-push
VERSION_TAG="latest_nr" RUNMODE="tpu_nr" make tpu-push
popd

echo -e "\n\n${yellow}Evaluator ${reset}"
pushd evaluator
VERSION_TAG="latest" make py-push cc-push
popd

echo -e "\n\n${yellow}Trainer ${reset}"
pushd trainer
VERSION_TAG="latest" make push
popd

echo -e "\n\n${yellow}Calibrator ${reset}"
pushd calibrator
VERSION_TAG="latest" make push
popd
