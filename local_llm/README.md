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

