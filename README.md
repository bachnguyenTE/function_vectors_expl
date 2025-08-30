# Function Vectors in Large Language Models
### [Project Website](https://functions.baulab.info) | [Arxiv Preprint](https://arxiv.org/abs/2310.15213) | [OpenReview](https://openreview.net/forum?id=AwyxtyMwaG)

This repository contains data and code for the paper: [Function Vectors in Large Language Models](https://arxiv.org/pdf/2310.15213).

<p align="left">
<img src="https://functions.baulab.info/images/Paper/fv-demonstrations.png" style="width:100%;"/>
</p> 

## Setup

We recommend using conda as a package manager. 
The environment used for this project can be found in the `fv_environment.yml` file.
To install, you can run: 
```
conda env create -f fv_environment.yml
conda activate fv
```

### Platform Compatibility
This repository now supports multiple platforms:
- **CUDA**: Full GPU acceleration on NVIDIA systems
- **MPS**: Apple Silicon GPU acceleration on macOS (M1/M2/M3 chips)
- **CPU**: Fallback for systems without GPU acceleration

The codebase automatically detects and uses the optimal device available. See `MPS_COMPATIBILITY_CHANGES.md` for detailed information about macOS compatibility changes.

## Demo Notebook
Checkout `notebooks/fv_demo.ipynb` for a jupyter notebook with a demo of how to create a function vector and use it in different contexts.

## Analysis and Visualization
- `analyze_intervention_topk.ipynb`: Comprehensive analysis notebook for visualizing and comparing intervention results across different models and tasks. Includes plotting capabilities for top-k accuracy metrics and cross-platform result analysis.

## Data
The datasets used in our project can be found in the `dataset_files` folder.

## Code
Our main evaluation scripts are contained in the `src` directory with sample script wrappers in `src/eval_scripts`.

Other main code is split into various util files:
- `eval_utils.py` contains code for evaluating function vectors in a variety of contexts
- `extract_utils.py`  contains functions for extracting function vectors and other relevant model activations.
- `intervention_utils.py` contains main functionality for intervening with function vectors during inference
- `model_utils.py` contains helpful functions for loading models & tokenizers from huggingface
- `prompt_utils.py` contains data loading and prompt creation functionality
- `device_utils.py` contains cross-platform device detection and tensor operations (CUDA/MPS/CPU)

### Recent Updates
- **Enhanced Chain-of-Thought Support**: Improved integration of function vectors with CoT prompting strategies
- **Cross-Platform Compatibility**: Full support for macOS MPS acceleration alongside existing CUDA support
- **Visualization Tools**: New analysis notebook with comprehensive plotting capabilities for intervention results

## Citing our work
This work appeared at ICLR 2024. The paper can be cited as follows:

```bibtex
@inproceedings{todd2024function,
    title={Function Vectors in Large Language Models}, 
    author={Eric Todd and Millicent L. Li and Arnab Sen Sharma and Aaron Mueller and Byron C. Wallace and David Bau},
    booktitle={Proceedings of the 2024 International Conference on Learning Representations},
    url={https://openreview.net/forum?id=AwyxtyMwaG},
    note={arXiv:2310.15213},
    year={2024},
}
