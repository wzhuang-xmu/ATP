#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

beta=0.002
best_ppl=""
best_beta=""

while true; do
    echo "Run with beta=${beta}"
    
    output=""
    illegal_beta_found=false
    
    while IFS= read -r line; do
        echo "$line"
        output+="$line"$'\n'
        
        if [[ "$line" == *"Illegal beta"* ]]; then
            illegal_beta_found=true
        fi
    done < <(python main.py \
        --model meta-llama/Llama-2-7b \
        --prune_method wanda \
        --sparsity_ratio 0.7 \
        --beta ${beta} 2>&1)
    
    if [ "$illegal_beta_found" = true ]; then
        echo "Stop loop with beta=${beta} due to illegal beta"
        break
    fi
    
    ppl=$(echo "$output" | grep -oP 'wikitext perplexity \K[0-9]+\.?[0-9]*')
    
    echo "Get beta=${beta}, LLM ppl = ${ppl}"
    
    if [ -z "$best_ppl" ] || [ $(echo "$ppl < $best_ppl" | bc -l) -eq 1 ]; then
        best_ppl=$ppl
        best_beta=$beta
    fi
    
    beta=$(echo "$beta + 0.002" | bc -l)
done

echo "Best LLM ppl = ${best_ppl} when beta=${best_beta}"
