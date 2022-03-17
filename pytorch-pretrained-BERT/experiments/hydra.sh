#!/usr/bin/env bash

TASK=$1
OPTIONS="${@:2}"

here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $here/prepare_task.sh $TASK

PERCENTS=${PERCENTS:-"`seq 5 5 100`"}

echo $base_acc
prune_options="--do_prune --eval_pruned --prune_percent $PERCENTS --retrain_pruned_heads --no-progress-bars --exact_pruning  $OPTIONS"
run_eval "$prune_options"

