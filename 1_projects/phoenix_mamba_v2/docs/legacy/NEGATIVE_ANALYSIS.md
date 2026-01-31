# Negative Analysis Report

This report identifies abandoned concepts, failed experiments, and limitations of the current approaches in the brain-tumor-detection project.

## 1. Abandoned Architectures and Experiments

The project's history is littered with experimental architectures and research directions that were ultimately abandoned. This is not necessarily a sign of failure, but rather a testament to the project's ambitious and exploratory nature. However, it is important to document these abandoned paths to understand the project's trajectory and avoid repeating past mistakes.

### Early Attention Mechanisms (v1)

Version 1 of the project included a variety of attention mechanisms, such as:

- `models/coordinate_attention.py`
- `models/fallback_attention.py`
- `models/sevector_attention.py`

These models were likely part of an early exploratory phase to determine the most effective attention mechanisms for brain tumor detection. However, they do not appear in later versions of the project, suggesting that they were abandoned in favor of the more focused NeuroSnake and NeuroKAN architectures. The reasons for this are not explicitly documented, but it is likely that these generic attention mechanisms did not provide a significant enough performance boost to justify their inclusion in the final models.

### Cloud-Based Training (v2, v3)

The presence of a `Deepnote_Camber_Training.ipynb` notebook in both v2 and v3 suggests that a cloud-based training approach was considered. However, the lack of integration with the main training pipelines and the presence of local training scripts (`one_click_train_test.py`, `start_cloud_training.sh`) suggest that this approach was not fully adopted. The reasons for this are unclear, but it could be due to cost, complexity, or a lack of perceived benefits over local training.

### The "Graveyard" of `v3/research_artifacts`

The `v3/research_artifacts` directory is a treasure trove of abandoned experiments. It contains multiple iterations of research, each with its own set of models, papers, and peer reviews. This suggests a period of intense experimentation, with the project exploring a variety of different approaches before settling on the final Hyper-Liquid Snake architecture. Some of the notable abandoned experiments in this directory include:

- **Kolmogorov-Arnold Networks (KANs)**: `iteration_2` and `iteration_3` focus heavily on KANs, with models like `neuro_snake_kan.py` and `ttt_kan.py`. The presence of peer reviews and papers on this topic suggests that this was a serious research direction at one point. However, the project ultimately moved on to the Hyper-Liquid Snake architecture, suggesting that KANs did not meet the project's expectations.
- **Test-Time Training (TTT)**: `iteration_3` explores the concept of Test-Time Training, which is a technique for adapting a model to new data at inference time. This was likely an attempt to improve the model's robustness to domain shift. However, this approach was also abandoned, suggesting that it was either not effective enough or too complex to implement.
- **Liquid State Machines**: `iteration_4` focuses on Liquid State Machines, which are a type of recurrent neural network that is well-suited for processing time-series data. This was likely an attempt to better model the temporal aspects of brain tumor growth. However, this approach was also abandoned in favor of the final Hyper-Liquid Snake architecture.

## 2. Redundant or Deprecated Code

Over the course of its evolution, the project has accumulated a significant amount of redundant or deprecated code. This is a natural consequence of a long-running research project, but it is important to identify and document this code to avoid confusion and streamline future development.

### `one_click_train_test.py`

This script, which is present in both v1 and v2, appears to be an early attempt at a simplified user interface for training and testing models. However, the move towards more complex and configurable training pipelines in v3 (`start_cloud_training.sh`) suggests that this script is now deprecated. It is likely that the one-click approach was too simplistic for the project's evolving needs, and that a more flexible and powerful training infrastructure was required.

### v1 Analysis and Example Scripts

The `v1` directory contains a number of scripts that appear to be for initial exploration and analysis, such as:

- `analyze_project.py`
- `examples.py`

These scripts were likely useful in the early stages of the project, but they have since been superseded by the more robust and comprehensive analysis tools in v3 (`tools/`). As such, they should be considered deprecated.

## 3. Implicit Failures and Limitations

By analyzing the project's evolution, we can infer a number of implicit failures and limitations that are not explicitly documented. These are not necessarily criticisms of the project, but rather observations about the challenges and trade-offs that were made along the way.

### The Failure of the "One Size Fits All" Approach

The evolution from a flexible, model-agnostic framework in v1 to a highly specialized, single-architecture system in v2 and v3 strongly implies that a "one size fits all" approach was not effective for this problem domain. The project likely found that a custom-built architecture was necessary to achieve the desired performance. This suggests that the initial hypothesis of v1 – that a single framework could be used to compare a variety of different models – was ultimately proven to be incorrect.

### The Performance Ceiling of NeuroKAN (v2)

The shift from the production-oriented NeuroKAN model in v2 to the experimental Hyper-Liquid Snake in v3 suggests that v2, while mature, likely hit a performance ceiling. The focus on adaptability and robustness in v3 is a direct response to the perceived limitations of v2. This suggests that while NeuroKAN was a successful model, it was not able to meet the project's long-term goals.

### The "Kitchen Sink" Approach of v3

The sheer number of experimental models in the `v3/src/models` directory (`neuro_snake_hyper.py`, `neuro_snake_kan.py`, `neuro_snake_liquid.py`, `ttt_kan.py`) suggests a period of intense and perhaps somewhat frantic experimentation. This "kitchen sink" approach, where a variety of different ideas are thrown at a problem in the hope that one of them will stick, is a common pattern in research projects. However, it can also be a sign of a project that is struggling to find a clear path forward. The lack of a clear, unifying vision in the early stages of v3 may have led to a significant amount of wasted effort.
