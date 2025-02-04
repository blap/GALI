# A Training-Free Length Extrapolation Approach for LLMs: Greedy Attention Logit Interpolation


Implementation of the training-free length extrapolation method GALI in [A Training-Free Length Extrapolation Approach for LLMs: Greedy Attention Logit Interpolation](). 


## Updates
- [05/02/2024]:🎉 Open source!


## Requirements:
- Please use the versions of the libraries written in the requirements.txt.

## Baseline Methods Implementations

**SelfExtend** [https://github.com/datamllab/LongLM]

**ChunkLlama** [https://github.com/HKUNLP/ChunkLlama]

**YaRN, NTK, Dyn-NTK Llama** [https://huggingface.co/]


## 1. Overview 
Transformer-based Large Language Models (LLMs) struggle to process inputs exceeding their training context window, with performance degrading due to positional out-of-distribution (O.O.D.) that disrupt attention computations. Existing solutions, fine-tuning and training-free methods, are limited by computational inefficiency, attention logit outliers or loss of local positional information. To address this, we propose Greedy Attention Logit Interpolation (GALI), a training-free length extrapolation method that maximizes the utilization of pretrained positional intervals while avoiding attention logit outliers through attention logit interpolation. The result demonstrates that GALI consistently outperforms state-of-the-art training-free methods. Our findings reveal that LLMs interpret positional intervals unevenly within their training context window, suggesting that extrapolating within a smaller positional interval range yields superior results—even for short-context tasks. GALI represents a significant step toward resolving the positional O.O.D. challenge, enabling more reliable long-text understanding in LLMs.

<p align="center">
<img width="600" src="./images/main.png">


## 2. How to Use GALI

### 2.1 Setup

- Install the libaries listed in the requirements.txt.

- Download the source code.

### 2.2 Get the model

We provide the GALI implementation based on Llama. The model modification are written in the models/DPI/patches/Llama.py, other help function are written in files in the models/DPI dir. Users can get a GALI-Llama by using the following code:

```python
from models.DPI.dpi import get_model_and_tokenizer

method = "dpi"
params = dict(use_chunk_softmax = false, chunk_coe = 3000, appro_attn = true, local_window = 128, noise_type = "gaussian", addon_relcoef = 1,std_base = 1.0, scale_mean = 0, scale_std = 1)
ori_max_position_embeddings = 8192 # The initial context window of the LLM defined in our paper.
model, tokenizer = get_model_and_tokenizer(model_name, config, method, params, max_position_embeddings=ori_max_position_embeddings)

```
User can also use function "get_model_and_tokenizer" in models/DPI/dpi.py to get other baseline models with flash-attention used in our paper, including SelfExtend, ChunkLlama, NTK, Dyn-NTK, ChunkLlama. We also provide an alternative function "get_model_and_tokenizer" for non flash-attention versions for each method in models/DPI/dpi_withoutflash.py. We provide explanation of the params required by each method in the following:

- GALI:
```
method = "dpi"
params = {
    # Whether use chunked softmax function when calculating the attention scores. This param can slightly save peek memory.
    use_chunk_softmax = false, 
    # The chunk size in our paper. It can be a float number as well, which means the chunk size is dynamic. See the details in get_chunk_size_list function.
    chunk_coe = 3000, 
    # Whether use attention logit interpolation
    appro_attn = true, 
    # The local window in our paper. See the details in construct_new_pi function
    local_window = 128, 
    # The noise params used in attention logit interpolation. See the details in attention_score_approximate.
    noise_type = "gaussian", addon_relcoef = 1,std_base = 1.0, scale_mean = 0, scale_std = 1}
```

- SelfExtend:
```
method = "repro_se"
params = {
    # The group size in their paper
    group_size = 3,
    # The window size in thir paper
    window_size = 4096 
    
    # In practice, we must ensure that (initial context window - window)*group_size+window >= target context window.}
```

- ChunkLlama:
```
method = "repro_chunkllama"
params = {} # leave it blank
```

- NTK:
```
method = "repro_ntk"
params = {
    # The scaling factor
    factor = 4 
    # In practice, we must ensure that initial context window * factor >= target context window.
} 
```

- Dyn-NTK:
```
method = "repro_dynamic_ntk"
params = {
    # The scaling factor
    factor = 4 
    # In practice, we must ensure that initial context window * factor >= target context window.
} 
```

- YaRN:
```
method = "repro_yarn"
params = {
     # The scaling factor
    factor = 4 
    # In practice, we must ensure that initial context window * factor >= target context window.
}
```

- Original Llama:
```
method = "repro_original"
params = {} # leave it blank.
```

### 2.3 Run the experiments

We provide the code for runing the experiments including LongBench, L-Eval, PG19 PPL test, Needle-in-a-stack. Users can use the following code to run the experiments and collect the results.

```
# Run the experiments
python pred.py --cfg expcfg/longbench_llama2_16k_dpi.toml

# Evaluate the predictions 
python eval.py --task longbenh --exp dpi-llama2-7b-4k-to-16k

# Collect the results to the excel file
python collect_results.py --task longbenh --exp all

```
For different experiments, we only need modify the "task" param in the toml config file. We provide the params requird for each experiments in the following:
- LongBench: 
```
task = "longbench"
```
- L-Eval: leval
```
task = "leval"
```
- PG19
```
task = "pg19"
stride = 2000 # split the inputs to avoid high peek memory
```
- Needle-in-a-stack
```
task = "needle"
times = 20 # repeat 20 times for each length setting
min_k = 1 # the minimal length of the input, use thousands as the unit.
max_k = 32 # the maximum length of the input use thousands as the unit.
gap = 4 # the percentage interval of the lengths
```

Other important params in the config file:
```
exp_name = 'dpi-llama2-7b-4k-to-16k' # experiment directory to save the results generated by LLM. 
max_pe = 16384 # 8192, 16384 32768, the target context window defined in our paper
ori_max_position_embeddings = 4096 # 2048, 4096, 8192, the initial context window defined in our paper
```
## Get the analysis results and images

The code for analysis and images in our paper are in the "analysis_paper.ipynb" file. Note that for attention analysis, we average the attention score or logit metrix along the head and layer dimension, i.e., [batch size, layer, head, q_len, k_len] -> [batch size, q_len, k_len]. 

We also upload our generated images in the directory ./images/


------


<!-- If you find our method useful, please kindly cite our paper.
```bibtex
@misc{jin2024llm,
      title={LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning}, 
      author={Hongye Jin and Xiaotian Han and Jingfeng Yang and Zhimeng Jiang and Zirui Liu and Chia-Yuan Chang and Huiyuan Chen and Xia Hu},
      year={2024},
      eprint={2401.01325},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
``` -->


## 4. Contributing
We welcome contributions from the research community to improve the effeicency of SelfExtend. If you have any idea or would like to report a bug, please open an issue or submit a pull request.

## 5. License
The code is released under the MIT License.

