#!/usr/bin/env bash

TASK=$1
OPTIONS="${@:2}"

here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
#source $here/prepare_task.sh $TASK


echo "${base_acc},baseacc"
echo $part
echo "Layer \\textbackslash~Head & 1 & 2 & 3 & 4 \\\\"
for layer in `seq 1 12`
do
    echo -n "$layer"
    for head in `seq 1 12`
    do
        mask_str="${layer}:${head}"
        acc=$(run_eval "--attention_mask_heads $mask_str $OPTIONS" | grep $metric | rev | cut -d" " -f1 | rev)
        printf " & %.5f" $(echo "$acc - $base_acc" | bc )
    done
    echo " \\\\"
done

