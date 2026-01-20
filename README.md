# The-Delta-Operator: A Universal Distance-Based Primitive for Neural Networks

This repository contains the PyTorch implementation of the paper **"The Delta Operator: A Universal Distance-Based Primitive for Neural Networks"**.

We propose the **Delta Operator**, a distance-based primitive in neural networks. This repository provides implementations across various domains, including Large Language Models (LLMs), Diffusion Transformers (DiT), and Vision tasks.

## Repository Structure

* **`Delta-DiT/`**: Implementation of Delta-Operator in Diffusion Transformers (trained on CelebA and Tiny-ImageNet).
* **`Delta-LLM/`**: Small-scale (100M) language modeling experiments comparing DeltaNet, BitNet, Performer, and Standard Transformer.
* **`Delta-LLM-0.94B/`**: Large-scale (0.94B parameter) language modeling implementation.
* **`Delta-Vis/`**: Implementation for computer vision classification tasks (CIFAR-100, ImageNet).

## Requirements

The code is implemented in PyTorch. Please install the dependencies using the following command:

```bash
pip install -r requirements.txt

```

*Key Dependencies:*

* `torch>=2.0`
* `transformers`
* `datasets`

---


## Training Logs

Due to the size limit of the anonymous repository, we do not provide the full model checkpoints for the 1B models. However, we provide **training logs** to verify the convergence and performance claims made in the paper.

You can find the training logs in the respective directories:

* `Delta-LLM-0.94B/checkpoints_delta_llm-1B/training-delta.log`
* `Delta-Vis/Delta_Vis_CIFAR100-Trainlog.txt`
* `Delta-DiT/celeba/results/.../logs/training_log.txt`

## License

This project is released under the MIT License.
