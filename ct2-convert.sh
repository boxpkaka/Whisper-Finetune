ct2-transformers-converter \
--model $1 \
--output_dir $2 \
--copy_files tokenizer.json preprocessor_config.json \
--quantization float16 \
--force