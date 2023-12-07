ct2-transformers-converter \
--model $1 \
--output_dir $2 \
--copy_files tokenizer_config.json preprocessor_config.json tokenizer.json \
--quantization float16 \
--force
