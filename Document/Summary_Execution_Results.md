# Execution Results of Different Execution

## Hardware - Software

- Environment execution: Colab Pro
- CUDA driver (library) version: 12.2
- GPU name: T4
- GPU comptute capability: 7.5
- GPU architecture: .

## Table of executed version

| ID  | Program specification | Time execution - Forward only[^1] (ms) | Test Accuracy | Time of running Test set[^2] (ms) |
| --- | --------------------- | ---------------------------------- | ----------------------- | - |
| A0 | LeNet-5 - Original - MNIST || 0.9737 ||
| **A1** | **LeNet-5 - Original - Fashion-MNSIT** | **$ \approx 1040.63$** | **0.8297** | **78152.4** |
| A2 | LeNet-5 - Original - Transpose Matrix Multiplication | $ \approx 985.677$ | 0.8194 ||
| **B0** | **LeNet-5 - Sequential (Im2Col) - ~~Transpose Matrix Multiplication~~** | **$ \approx 1115.75 \rightarrow 916.248$** | **0.8297** | **73255.4** |
| C0 | LeNet-5 - Parallel Version 1 (Im2Col) - Get unrolled image matrix avoid naive copy | $ \approx 696.307 $ | 0.8297 ||
| **C1** | **LeNet-5 - Parallel Version 1 (Im2Col) - Same as (C0), but get unrolled image results directly from dynamic memory to Matrix object** | **$ \approx 539.801 $** | **0.8297** | **41349.5** |
| **D0** | **LeNet-5 - Parallel Version 2 (MatMul) - Save matrix while multiplying as column-major order** | | **0.8297** | **31260.9** |

[^1]: Avarage of all forward passes in first epoch.
[^2]: The Test set contains 10000 samples. The results is the best one after re-run the executable files 10 times.
