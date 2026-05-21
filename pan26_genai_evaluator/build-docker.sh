#!/usr/bin/env bash

set -e
docker build -t ghcr.io/pan-webis-de/pan26-generative-authorship-evaluator "$@" .
docker push ghcr.io/pan-webis-de/pan26-generative-authorship-evaluator
