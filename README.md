# A Cross-View Hierarchical Graph Learning Hypernetwork for Skill Demand-Supply Joint Prediction
![Testing Status](https://img.shields.io/badge/docs-in_progress-green)
![Testing Status](https://img.shields.io/badge/pypi_package-in_progress-green)
![Testing Status](https://img.shields.io/badge/license-MIT-blue)
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [A Cross-View Hierarchical Graph Learning Hypernetwork for Skill Demand-Supply Joint Prediction](#cross-view-hierarchical-graph-learning-with-hypernetwork-for-skill-supply-demand-joint-prediction)
  * [Overview](#overview)
	* [Installation](#installation)
	* [How to Run Model](#how-to-run-model)
	* [Dataset](#dataset)
  * [Citation](#citation)
	* [License](#license)
	* [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->
## Overview

Official code for article "[A Cross-View Hierarchical Graph Learning Hypernetwork for Skill Demand-Supply Joint Prediction](https://arxiv.org/abs/2401.17838)".


The rapidly changing landscape of technology and industries leads to dynamic skill requirements, making it crucial for employees and employers to anticipate such shifts to maintain a competitive edge in the labor market. In this paper, we propose a Cross-view Hierarchical Graph learning Hypernetwork (CHGH) framework for joint skill demand-supply prediction.

![Framework](img/framework.pdf)



## Installation
Create a python 3.7 environment and install dependencies:

  ```
  conda create -n python3.7 CHG
  source activate CHG
  ```

Install library

  ```
  pip install -r ./requirements.txt
  ```

## 4 Dataset

We conduct experiments on 3 datasets, i.e. IT, FIN, CONS. They have the same format and in this repository we provide an example of job postings (demand) and work experiences (supply) data. Due to the privacy issue, you could collect your own datasets and run our code.

### Job Postings (Demand)

|Company|Released Date|Skill List|
|:-:|:-:|:-:|:-:|
|Google|202101|Python, Statistics, Machine Learning|
|...|...|...|...|

### Work Experiences (Supply)

|People|Company|StartDate|EndDate|Skill List|
|:-:|:-:|:-:|:-:|:-:|
|XXXXXX|Microsoft|201603|202210|Database, SQL, Linux|
|...|...|...|...|

## How to Run Model

To run the CHGH, you should set args in `config.json` or in command and `wandb` account in advance to run `train.py` for both training and testing phase: 

  ```
  python train.py --config config.json --name "it_hier_add_32" --sk_emb_dim 32 -um Skill_Evolve_Hetero --emb_dim 32  -m hier --dataset it -lr 0.001 --delta 0.1 -gcn 2 -dev 2 --wandb true -hyp true -hypcomb add --dropout 0.3
  python train.py --config config.json --name it_mtgnn_32 --sk_emb_dim 32 -um Graph_Baseline --emb_dim 32 -m mtgnn --dataset it -lr 0.001 --subgraph 7446 --skill_num 7446 -dev 1 --wandb true --dropout 0.3
  ```


## Citation

If you find our work is useful for your research, please consider citing:

```
@article{chao2024cross,
  title={A Cross-View Hierarchical Graph Learning Hypernetwork for Skill Demand-Supply Joint Prediction},
  author={Chao, Wenshuo and Qiu, Zhaopeng and Wu, Likang and Guo, Zhuoning and Zheng, Zhi and Zhu, Hengshu and Liu, Hao},
  journal={arXiv preprint arXiv:2401.17838},
  year={2024}
}
```

```
@inproceedings{chao2024cross,
  title={A Cross-View Hierarchical Graph Learning Hypernetwork for Skill Demand-Supply Joint Prediction},
  author={Chao, Wenshuo and Qiu, Zhaopeng and Wu, Likang and Guo, Zhuoning and Zheng, Zhi and Zhu, Hengshu and Liu, Hao},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  year={2024}
}
```

## License
This project is licensed under the MIT License.

## Acknowledgements
This project is follow the [Pytorch-Project-Template](https://github.com/victoresque/pytorch-template) built by [Victor Huang](https://github.com/victoresque)
