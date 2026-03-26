#!/bin/zsh

# models: EleutherAI/pythia-14m-deduped EleutherAI/pythia-70m-deduped EleutherAI/pythia-160m-deduped EleutherAI/pythia-410m-deduped EleutherAI/pythia-1b-deduped EleutherAI/pythia-1.4b-deduped EleutherAI/pythia-2.8b-deduped EleutherAI/pythia-6.9b-deduped EleutherAI/pythia-12b-deduped
script/run \
  --models EleutherAI/pythia-70m-deduped \
  --tasks rules hierarchical matrix matrix_easy \
  --n-examples 0 1 3 5 7 10 20 \
  --n-stimuli 100 \
  --systematic