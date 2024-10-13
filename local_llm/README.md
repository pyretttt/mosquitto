## Prerequisites
Cmake
build-essential
python3
git-lfs

## Initial setup
- Init and update submodules
```
git submodule init local_llm/submodules/llama.cpp local_llm/submodules/llama-cpp-python
git submodule update --recursive
```

- Load model/weights
For example from hugging face via `git clone git@hf.co:google/gemma-2-9b-it && git lfs pull`

## Build llama.cpp
- Init project `cmake -S submodules/llama.cpp -B submodules/llama.cpp/build`
- Build `cmake --build submodules/llama.cpp/build --config Release -j 8`

## Create and activate python env
- Create env `python3.12 -m venv no_index/env`
- Activate `source no_index/env/bin/activate`
- Update `pip install -r submodules/llama.cpp/requirements.txt`

## Now convert loaded model to gguf format
For example
```
python submodules/llama.cpp/convert_hf_to_gguf.py no_index/Meta-Llama-3-8B-Instruct/ --outfile no_index/Meta-Llama-3-8B-Instruct.gguf --outtype auto
```

## Quantize converted model
For example
```
submodules/llama.cpp/build/bin/llama-quantize no_index/Meta-Llama-3-8B-Instruct.gguf no_index/Meta-Llama-3-8B-Instruct-q5_k_m.gguf Q5_K_M
```
where `Q5_K_M` described quantization methods, where 5bits per float and M enables medium mixed precision mode.

## Or make conversion and quantization in single step
For example
```
python submodules/llama.cpp/convert_hf_to_gguf.py no_index/Meta-Llama-3-8B-Instruct/ \
  --outfile no_index/Meta-Llama-3-8B-Instruct-q8.gguf \
  --outtype q8_0
```

## Check inference
For example
```
submodules/llama.cpp/build/bin/llama-simple -m no_index/Meta-Llama-3-8B-Instruct-q5_k_m.gguf -n 20 -p "What is the capital of Australia?"
```