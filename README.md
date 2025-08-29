```bash
conda create -n abliteration-exp python=3.11
conda activate abliteration-exp
pip install "huggingface_hub[cli]" transformers transformers_stream_generator tiktoken transformer_lens einops jaxtyping torch
git clone https://huggingface.co/mlabonne/Daredevil-8B meta-llama/Meta-Llama-3-8B-Instruct
# make sure git lfs is installed
huggingface-cli download mlabonne/Daredevil-8B --local-dir Daredevil-8B --local-dir-use-symlinks False
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir Meta-Llama-3-8B-Instruct --local-dir-use-symlinks False
```