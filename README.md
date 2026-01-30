# ReSCO: Reheated Gradient-based Discrete Sampling for Combinatorial Optimization

This repository contains the official implementation for the paper **"Reheated Gradient-based Discrete Sampling for Combinatorial Optimization"** (TMLR 2025).

ReSCO introduces a novel **reheating mechanism** inspired by critical temperature and specific heat physics to address the "wandering in contours" phenomenon in gradient-based discrete sampling.

## 🚀 Installation

This project is built using Python. To get started, clone the repository and install the dependencies:

```bash
# 1. Clone the repository
git clone https://github.com/PotatoJnny/ReSCO.git
cd ReSCO

# 2. Install dependencies (TensorFlow, Flax, Optax, etc.)
pip install -e .

```

## 📊 Running Experiments

We provide a shell script `run_sampling_local.sh` to run experiments locally. The script utilizes XLA with 4 local devices by default for parallel chain sampling.

You can control the experiment setup using environment variables: `model`, `sampler`, and `graph_type`.

### Usage

```bash
model=<MODEL_NAME> sampler=<SAMPLER_NAME> graph_type=<DATASET> bash run_sampling_local.sh

```

### Arguments

* **`model`**: The target combinatorial optimization problem.
* Options: `mis`  `max_clique`, `max_cut`,`normcut`.


* **`sampler`**: The gradient-based discrete sampler to use.
* Options: `pas`, `dmala` , `dlmc` , etc.


* **`graph_type`**: The dataset configuration.
* Examples: `ER_700_800`, `SATLIB`, `RB`, `Twitter`, etc.
* *Note:* If not specified, it defaults to the configuration in `ReSCO/common/configs.py`.



### Example Commands

To run a **Maximum Independent Set (MIS)** experiment using the **PAS** sampler with the reheated setting
 on an **Erdős-Rényi** graph:

```bash
model=mis sampler=pas graph_type=ER_700_800 bash run_sampling_local.sh
```



## 📝 Hyperparameters

Key hyperparameters for the ReSCO reheating mechanism can be found in the configuration files. The default recommended settings from the paper are:

* **Value Threshold ($\epsilon$)**: `0.01`
* **Wandering Length ($N$)**: `100`
* **Sample Size ($m$)**: `100`

## Citation

If you find this code useful, please cite our paper:

```bibtex
@article{li2025reheated,
  title={Reheated Gradient-based Discrete Sampling for Combinatorial Optimization},
  author={Li, Muheng and Zhang, Ruqi},
  journal={arXiv preprint arXiv:2503.04047},
  year={2025}
}
```
