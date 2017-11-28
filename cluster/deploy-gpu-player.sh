#!/bin/bash

source ./common

envsubst < gpu-player.yaml | kubectl apply -f -
