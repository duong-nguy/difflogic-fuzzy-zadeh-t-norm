# difflogic - A Library for Differentiable Logic Gate Networks

This repository is a fork of the original implementation of "Deep Differentiable Logic Gate Networks", presented at NeurIPS 2022 (ArXiv). We extend this work in our paper "Deep Differentiable Logic Gate Networks Based on Fuzzy Zadeh's T-norm", building upon the differentiable logic gate framework to further enhance its capabilities.

Differentiable logic gate networks aim to solve machine learning tasks by learning compositions of logic gates, forming logic gate networks. Traditionally, logic gate networks are non-differentiable and cannot be trained using gradient-based methods. The differentiable relaxation introduced in this approach enables efficient training via gradient descent. Specifically, difflogic integrates real-valued logics with a continuously parameterized relaxation, allowing each neuron to determine the optimal logic gate (out of 16 possibilities).

This formulation results in discretized logic gate networks that achieve high inference speeds—processing over a million MNIST images per second on a single CPU core.

We acknowledge the original authors for their foundational work, which this repository builds upon.
`difflogic` is a Python 3.6+ and PyTorch 1.9.0+ based library for training and inference with logic gate networks.

Visit the original work at https://github.com/Felix-Petersen/difflogic/tree/main

## 🧪 Experiments

In the following, we present a few example experiments which are contained in the `experiments` directory.
`main.py` executes the experiments for difflogic and `main_baseline.py` contains regular neural network baselines.


```shell
./run.sh
```



## 📜 License

`difflogic` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it. 

