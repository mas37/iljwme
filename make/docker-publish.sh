#!/bin/bash

export DOCKER_ID_USER="mlip"
docker tag $1 $DOCKER_ID_USER/mlip
docker login
docker push $DOCKER_ID_USER/mlip