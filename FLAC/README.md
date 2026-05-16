<div align="center">
 👋 Hi, everyone! 
    <br>
    We are <b>ByteDance Seed team.</b>
</div>

![seed logo](https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216)

# FLAC: Maximum Entropy RL via Kinetic Energy Regularized Bridge Matching

We are delighted to introduce **FLAC** (**F**ield **L**east-Energy **A**ctor-**C**ritic), a likelihood-free framework for maximum entropy reinforcement learning that regulates policy stochasticity by penalizing the kinetic energy of the velocity field. FLAC integrates flow-based generative policies with principled entropy regularization — without ever computing action log-densities.

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2602.12829-B31B1B.svg)](https://arxiv.org/abs/2602.12829)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://pinkmoon-io.github.io/flac.github.io/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

## News
- [2026/03] 🔥 We release the code for FLAC.
- [2026/02] 🎉 We release our paper on arXiv.

## Introduction

Iterative generative policies, such as diffusion models and flow matching, offer superior expressivity for continuous control but complicate Maximum Entropy Reinforcement Learning because their action log-densities are not directly accessible. FLAC addresses this challenge by formulating policy optimization as a **Generalized Schrödinger Bridge (GSB)** problem relative to a high-entropy reference process (e.g., uniform)[[FLAC: Maximum Entropy RL via Kinetic Energy Regularized Bridge Matching]](https://lvlei-221.github.io/flac.github.io/).

Under this view, the maximum-entropy principle emerges naturally as staying close to a high-entropy reference while optimizing return, without requiring explicit action densities. Kinetic energy serves as a physically grounded proxy for divergence from the reference: minimizing path-space energy bounds the deviation of the induced terminal action distribution[[FLAC: Maximum Entropy RL via Kinetic Energy Regularized Bridge Matching]](https://lvlei-221.github.io/flac.github.io/).

### Key Features

- **Likelihood-Free**: No need to compute intractable log π(a|s) for generative policies.
- **Principled**: GSB theory guarantees the terminal distribution matches the Boltzmann form.


### The FLAC Objective

FLAC combines GSB formulation, RL potential, and kinetic energy regularization into a single tractable objective:

$$\min_{\theta} J_{\text{FLAC}}(\theta) = \mathbb{E}_{\mathbb{P}^\theta} \left[ \alpha \int_0^1 \frac{1}{2} \left\| u_\theta(s, \tau, X_\tau) \right\|^2 d\tau - Q(s, X_1) \right]$$

The objective minimizes kinetic energy (as an entropy proxy) while maximizing return — fully tractable with no density evaluation needed[[FLAC: Maximum Entropy RL via Kinetic Energy Regularized Bridge Matching]](https://lvlei-221.github.io/flac.github.io/).

## Getting Started

1. **Setup Conda Environment:**
    Create an environment with
    ```bash
    conda create -n flac python=3.11
    ```

2. **Clone this Repository:**
    ```bash
    git clone https://github.com/bytedance/FLAC.git
    cd FLAC
    ```

3. **Install FLAC Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Training Examples:**
    - Run parallel training:
        ```bash
        bash scripts/train_parallel.sh
        ```


## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Citation

If you find FLAC useful for your research and applications, please consider giving us a star ⭐ or cite us using:

```bibtex
@article{lv2026flac,
  title={FLAC: Maximum Entropy RL via Kinetic Energy Regularized Bridge Matching},
  author={Lv, Lei and Li, Yunfei and Luo, Yu and Sun, Fuchun and Ma, Xiao},
  journal={arXiv preprint arXiv:2602.12829},
  year={2026}
}
```

## About [ByteDance Seed Team](https://seed.bytedance.com/)

Founded in 2023, ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models. The team aspires to become a world-class research team and make significant contributions to the advancement of science and society.
