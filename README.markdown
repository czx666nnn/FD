# Federated Learning Framework for Defending Model Poisoning Attacks in IoT

## Project Overview
This project implements a **Federated Learning (FL)** framework designed to defend against model poisoning attacks in Internet of Things (IoT) environments. It addresses two primary attack scenarios: **data poisoning** (Scenario 1) and **weight poisoning** (Scenario 2). The framework leverages a dual trust evaluation mechanism to filter out malicious clients, ensuring robust model aggregation. Key components include a convolutional autoencoder for data trust assessment, cosine similarity for model trust evaluation, and a ResNet50-based model for classification tasks. The implementation is optimized for resource-constrained IoT devices through memory-efficient training and modular design.

## Features
- **Dual Trust Evaluation**:
  - **Data Trust (Scenario 1)**: Uses a convolutional autoencoder to detect poisoned data based on reconstruction errors, with robust normalization using median-based trust scores.
  - **Model Trust (Scenario 2)**: Computes cosine similarity between global and client model weights to identify weight poisoning attacks.
- **Customizable Attack Simulation**:
  - **Data Poisoning**: Simulates label flipping and noise injection using a custom `PoisonedCIFAR10` dataset.
  - **Weight Poisoning**: Introduces random noise to model weights to mimic malicious client behavior.
- **Memory Optimization**: Reduces GPU memory fragmentation with environment settings and periodic memory cleanup, suitable for IoT devices.
- **Modular Design**: Separates data loading, model training, trust evaluation, and aggregation for easy extension and experimentation.
- **Robust Aggregation**: Filters untrusted clients based on trust scores and aggregates only reliable models, incorporating global model weights for stability.

## Prerequisites
- **Python**: 3.8 or higher
- **PyTorch**: 1.9 or higher
- **Torchvision**: 0.10 or higher
- **NumPy**: 1.20 or higher
- **tqdm**: For progress bars
- **Hardware**: CUDA-enabled GPU recommended for faster training; CPU supported but slower.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/federated-learning-iot-defense.git
   cd federated-learning-iot-defense
   ```
2. Install dependencies:
   ```bash
   pip install torch torchvision numpy tqdm
   ```
3. Ensure the `./data` directory exists for downloading the CIFAR-10 dataset:
   ```bash
   mkdir data
   ```

## Usage
The main script (`51创新点.py`) runs the federated learning framework for both scenarios. To execute:
```bash
python 51创新点.py
```

### Configuration
Key parameters can be modified in the script:
- **Number of Clients**: `num_clients = 10` (default)
- **Rounds**: `rounds = 10` (default)
- **Data Poisoning Ratio**: `poison_ratio = 0.4` (Scenario 1)
- **Weight Noise Probability**: `fault_prob = 0.4` (Scenario 2)
- **Trust Thresholds**: 
  - Data trust: `0.4` (Scenario 1)
  - Model trust: `0.1` (Scenario 2)
- **Batch Size**: `batch_size = 32` (optimized for memory)
- **Epochs**: `epochs = 5` for client training and autoencoder training

### Output
The script outputs:
- Training progress for each round, including loss and accuracy.
- Trust scores for each client in both scenarios.
- Test accuracy after each round for both scenarios.
- Final accuracy lists (`zero_pad_accs_scenario1` and `zero_pad_accs_scenario2`).

## Project Structure
```
federated-learning-iot-defense/
├── 51创新点.py       # Main script with FL framework
├── data/             # Directory for CIFAR-10 dataset
├── README.md         # This file
```

## Scenarios
### Scenario 1: Data Poisoning
- **Objective**: Defend against poisoned datasets (label flipping and noise injection).
- **Mechanism**: 
  - Trains a convolutional autoencoder to compute reconstruction errors.
  - Calculates trust scores based on median-normalized errors.
  - Filters clients with trust scores below 0.4.
- **Dataset**: Custom `PoisonedCIFAR10` with 40% poisoned data (20% label flipping, 20% noise injection).

### Scenario 2: Weight Poisoning
- **Objective**: Mitigate weight tampering by malicious clients.
- **Mechanism**:
  - Adds random noise to client model weights with a 40% probability.
  - Computes cosine similarity between global and client model weights.
  - Filters clients with similarity scores below 0.1.
- **Dataset**: Standard CIFAR-10 dataset.

## Key Innovations
1. **Dual Trust Mechanism**: Combines data trust (autoencoder-based) and model trust (cosine similarity-based) for comprehensive defense.
2. **Custom Attack Simulation**: Realistic poisoning scenarios with configurable parameters.
3. **Memory Efficiency**: Optimized for IoT with reduced memory usage and periodic cleanup.
4. **Robust Normalization**: Median-based trust score normalization for stable filtering.
5. **Flexible Aggregation**: Incorporates global model weights in aggregation to enhance robustness.

## Limitations
- **Resource Intensive**: Training autoencoders and ResNet50 models may be slow on low-end IoT devices without GPUs.
- **Fixed Thresholds**: Trust thresholds (0.4 for data, 0.1 for models) may need tuning for different datasets or attack types.
- **Dataset Dependency**: Currently tailored for CIFAR-10; adaptation to other datasets requires transform adjustments.

## Future Work
- Support for additional datasets (e.g., ImageNet, custom IoT datasets).
- Dynamic trust threshold adjustment based on attack intensity.
- Integration with lightweight models (e.g., MobileNet) for ultra-low-resource IoT devices.
- Real-time attack detection and mitigation during training.
