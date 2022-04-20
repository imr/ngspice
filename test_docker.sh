#!/bin/bash

IMAGENAME="registry.gitlab.com/dospm/ngspice"
TAG="latest"

docker build -t $IMAGENAME:$TAG . 
docker run -it --rm --user "$(id -u)":"$(id -g)" -v "${PWD}":/tmp $IMAGENAME:$TAG /bin/bash