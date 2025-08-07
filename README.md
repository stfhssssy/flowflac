
# ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning

<!-- schematic: -->
<p align="center">
  <img src="sample_figs/schematic.png" alt="Architecture Diagram" width="90%">
</p>


<!-- links: -->


<!-- <div align="center">
  <img src="./sample_figs/ShortCutFlow_can_step1_1080_1920.gif" title="Fine-tuned Shortcut Flow in Robomimic-can at 1 denoising step" width="65%">
</div> -->

<p align="center">
  <img src="./sample_figs/ShortCutFlow_can_step1_1080_1920.gif" width="42%" alt="Shortcut Flow Can">
  <img src="./sample_figs/shortcut_transport_4steps.gif" width="42%" alt="Shortcut Transport">
</p>

<hr>
<div align="center">
  <a href="https://reinflow.github.io/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/Visit-Website-blue?style=for-the-badge" />
  </a>
  <a href="https://github.com/ReinFlow/ReinFlow/tree/release/docs" target="_blank">
    <img alt="Docs" src="https://img.shields.io/badge/Read-Docs-purple?style=for-the-badge" />
  </a>
  <a href="https://huggingface.co/datasets/ReinFlow/ReinFlow-data-checkpoints-logs/" target="_blank">
    <img alt="Checkpoints" src="https://img.shields.io/badge/Download-Checkpoints-orange?style=for-the-badge" />
  </a>
  <a href="https://x.com/TongheZhang01/status/1951121394886226194" target="_blank">
    <img alt="Checkpoints" src="https://img.shields.io/badge/View-Post-On-X-black?style=for-the-badge" />
  </a>
</div>
<!-- mini table of contents: -->
<p align="center">
  <a href="#rocket-installation">Installation</a> |
  <a href="#rocket-quick-start-reproduce-our-results">Quick Start</a> |
  <a href="#rocket-implementation-details">Implementation Details</a> |
  <a href="#rocket-adding-your-own-dataset-or-environment">Add Dataset/Environment</a> <br>
  <a href="#rocket-debug-aid-and-known-issues">Debug & Known Issues</a> |
  <a href="#license">License</a> |
  <a href="#acknowledgement">Acknowledgement</a>
</p>




This is the official implementation of _"ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning"_.  

If you like our work, we'll be happy if you give us a star **:star:**!

## 🚀 About ReinFlow


**ReinFlow** is a flexible **policy gradient framework** for fine-tuning **flow matching policies** at **any denoising step**.

How does it work?  
👉 First, train flow policies using **imitation learning** (behavior cloning).  
👉 Then, fine-tune them with **online reinforcement learning** using ReinFlow!

🧩 **Supports**:
- ✅ 1-Rectified Flow  
- ✅ Shortcut Models  
- ✅ Any other policy defined by ODEs (in principle)

📈 **Empirical Results**: ReinFlow achieves strong performance across a variety of robotic tasks:
- 🦵 Legged Locomotion (OpenAI Gym)  
- ✋ State-based manipulation (Franka Kitchen)  
- 👀 Visual manipulation (Robomimic)

🧠 **Key Innovation**: ReinFlow trains a **noise injection network** end-to-end:
- ✅ Makes policy probabilities tractable, even with **very few denoising steps** (e.g., 4, 2, or 1)  
- ✅ Robust to discretization and Monte Carlo approximation errors

Learn more on our 🔗 [project website](https://reinflow.github.io/) or check out the arXiv paper. 



## :loudspeaker: News
* [2025/07/30] Fixed the rendering bug in Robomimic. Now supports rendering at 1080p resolution. 
* [2025/07/29] Add tutorial on how to record videos during evaluation in the [docs](docs/ReproduceExps.md)
* [2025/06/14] Updated webpage for a detailed explanation to the algorithm design.
* [2025/05/28] Paper is posted on arXiv! 

## :rocket:  Installation
Please follow the steps in [installation/reinflow-setup.md](./installation/reinflow-setup.md).

## :rocket: Quick Start: Reproduce Our Results
To fully reproduce our experiments, please refer to [ReproduceExps.md](docs/ReproduceExps.md). 

To download our training data and reproduce the plots in the paper, please refer to [ReproduceFigs.md](docs/ReproduceFigs.md).

## :rocket: Implementation Details
Please refer to [Implement.md](docs/Implement.md) for descriptions of key hyperparameters of FQL, DPPO, and ReinFlow.

## :rocket: Adding Your Own Dataset or Environment
Please refer to [Custom.md](docs/Custom.md).

## :rocket: Debug Aid and Known Issues
Please refer to [KnownIssues.md](docs/KnownIssues.md) to see how to resolve errors you encounter.

## :star: Comming Soon
- Support fine-tuning [Mean Flow](https://arxiv.org/abs/2505.13447) with online RL
- Possible open-source the WandB projects via a corporate account.   (currently is in .csv format)
- Replace figs with videos in the drop-down menu of specific tasks in the webpage.

## License
This repository is released under the MIT license. See [LICENSE](LICENSE). 
If you use our code, we appreciate it if you paste the license at the beginning of the script. 

## Acknowledgement
This repository was developed from multiple open-source projects. Major references include:  
- [TorchCFM, Tong et al.](https://github.com/atong01/conditional-flow-matching): Conditional flow-matching repository.  
- [Shortcut Models, Francs et al.](https://github.com/kvfrans/shortcut-models): One-step Diffusion via Shortcut Models. 
- [DPPO, Ren et al.](https://github.com/irom-princeton/dppo): DPPO official implementation.  

For more references, please refer to [Acknowledgement.md](docs/Acknowledgement.md).

## Star History
<div style="display: flex; justify-content: center; align-items: center; height: 100vh; width: 100%; margin: 0; padding: 0;">
    <img src="https://api.star-history.com/svg?repos=ReinFlow/ReinFlow&type=Date" alt="Star History Chart" >
</div>