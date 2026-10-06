# NLP with JAX AI Stack

A curated collection of practical JAX implementations, functional training loops, and modern deep learning patterns for Natural Language Processing (NLP) built on the Google JAX ecosystem (**JAX**, **Flax**, **Optax**, and **Grain**).

This repository serves as a reference and cookbook for training, fine-tuning, and scaling NLP models—from dense text representations to small language models (SLMs)—with an emphasis on clean functional idioms, accelerator efficiency (GPU/TPU), and explicit distributed state management.

---

## 🛠️ The Tech Stack

- **Core & Autodiff:** [JAX](https://github.com/google/jax) (`jax.jit`, `jax.vmap`, `jax.grad`, `jax.shard_map`)
- **Neural Network Modules:** [Flax Linen / NNX](https://github.com/google/flax)
- **Optimization:** [Optax](https://github.com/google-deepmind/optax) (gradient transformations, schedules, and custom optimizers)
- **Data Loading & Preprocessing:** [Grain](https://github.com/google/grain) (deterministic, high-throughput pipelines)
- **Evaluation & Metrics:** [CLU](https://github.com/google/common-loop-utils) (Common Loop Utilities)

---
