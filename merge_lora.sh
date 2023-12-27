#!/bin/bash
python merge_lora.py \
	--base_model /data1/yumingdong/model/huggingface/whisper-large-v3 \
	--lora_model $1 \
	--output_dir models \
	--local_files_only True \
