# Execution Results of Different Execution

## Hardware - Software

- Environment execution: Colab Pro
- CUDA driver (library) version: 12.2
- GPU name: T4
- GPU comptute capability: 7.5
- GPU architecture: .

## Table of executed version

| ID  | Program specification | Time execution - Forward only[^1] (ms) | Test Accuracy | Time of testing |
| --- | --------------------- | ---------------------------------- | ----------------------- | - |
| A0 | LeNet-5 - Original - MNIST || 0.9737 ||
| A1 | LeNet-5 - Original - Fashion-MNSIT | $ \approx 1040.63$ | 0.8207 | 78854 |
| A2 | LeNet-5 - Original - Transpose Matrix Multiplication | $ \approx 985.677$ | 0.8194 ||
| B0 | LeNet-5 - Sequential (Im2Col) - ~~Transpose Matrix Multiplication~~ | $ \approx 1115.75 \rightarrow 916.248$ | 0.8297 ||
| C0 | LeNet-5 - Parallel Version 1 (Im2Col) - Get unrolled image matrix avoid naive copy | $ \approx 696.307 $ | 0.8297 ||
| C1 | LeNet-5 - Parallel Version 1 (Im2Col) - Same as (C0), but get unrolled image results directly from dynamic memory to Matrix object | $ \approx 539.801 $ | 0.8297 | 41050.3 |

[^1]: Avarage of all forward passes in first epoch.
