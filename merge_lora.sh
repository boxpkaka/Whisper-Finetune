#!/bin/bash
python merge_lora.py \
        --base_model $1 \
        --lora_model $2 \
        --output_dir models \
        --local_files_only True \