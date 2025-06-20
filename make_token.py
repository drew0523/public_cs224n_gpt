from transformers import GPT2Tokenizer
# --cache_dir 인자로 폴더 경로를 지정하면, 그 위치에 필요한 파일들을 저장해 줍니다.
GPT2Tokenizer.from_pretrained("gpt2", cache_dir="gpt2_tokenizer_cache")