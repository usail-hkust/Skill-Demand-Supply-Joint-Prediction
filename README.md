# Cross-View Hierarchical Graph Learning with Hypernetwork for Skill Supply-Demand Joint Prediction

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [Cross-View Hierarchical Graph Learning with Hypernetwork for Skill Supply-Demand Joint Prediction](#cross-view-hierarchical-graph-learning-with-hypernetwork-for-skill-supply-demand-joint-prediction)
  * [Overview](#overview)
	* [Installation](#installation)
	* [How to Run Model](#how-to-run-model)
	* [Contribution](#contribution)
	* [License](#license)
	* [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->
## Overview
The rapidly evolving landscape of technology and industries leads to changes in skill requirements, posing challenges for individuals to adapt to the dynamic work environment.
Anticipating these skill shifts is vital for employees to choose the right skills to learn and stay competitive in the job market.
We propose a Cross-view Hierarchical Graph learning model (CHG) for jointly predicting skill supply-demand to bridge the gap between employees and employers by learning the relation between skills. 
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


## How to Run Model
Modify the configurations in `.json` config files, or use the args stated in `trainer.py`:

  ```
  python train.py --config config.json --name "it_hier_add_32" --sk_emb_dim 32 -um Skill_Evolve_Hetero --emb_dim 32  -m hier --dataset it -lr 0.001 --delta 0.1 -gcn 2 -dev 2 --wandb true -hyp true -hypcomb add --dropout 0.3
  python train.py --config config.json --name it_mtgnn_32 --sk_emb_dim 32 -um Graph_Baseline --emb_dim 32 -m mtgnn --dataset it -lr 0.001 --subgraph 7446 --skill_num 7446 -dev 1 --wandb true --dropout 0.3
  ```
## License
This project is licensed under the MIT License.

## Acknowledgements
This project is follow the [Pytorch-Project-Template](https://github.com/victoresque/pytorch-template) built by [Victor Huang](https://github.com/victoresque)
