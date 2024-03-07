# g-gymnust
Minimalistic GPU-enabled reinforcement learning environment API built in Rust, inspired by Gymnasium.

## Features
### Why in Rust?
As of 2024, Python-based frameworks have provided strong tools for RL researchers and developers to make significant advancement in the field.
The focus of the field, however, has been shifting to real-world safety-critical applications such as automatives and avionics, which Python is not necessarily optimized for in terms of the memory safety and execution speed.
Meanwhile, the ML community has also been exploring faster and safer solutions for deep learning, and we now observe an increasing number of ML frameworks written in Rust which features the safer memory management, high abstractions, and decent execution speed.
Nonetheless, there is neither a standard RL framework nor an environment API written in Rust as of 2024.
We hope that this project serves as a reference to further development of RL ecosystems in the Rust community.

### How similar to the original Gynmasium api?
We aim to faithfully implement the core features of Gymnasium (e.g. `core.py`, `seeding.py`, `spaces`).
For the time-being, we will not focus on Python-specific parts of the original project such as JAX to NumPy conversions.

### Why GPU?
Graphical Processing Unit (GPU) is the essential computing component for tensor operations heavily used in deep learning and consequently in reinforcement learning.
While the core APIs do not use GPUs as it is more about providing interfaces, the example environments will be GPU-enabled.
For those environments, we will use an existing ML framework written in Rust, namely `candle`, though, we will try our best to minimize the exposure to a specific API considering the scalability.

## Installation

## License
We follow Rust's dual license convention: Apache License 2.0 and MIT license.
