# Transformer Model for Bilingual Translation

This project implements a transformer model for bilingual translation using PyTorch. The model is specifically designed to translate sentences from a source language to a target language, leveraging the Opus Books dataset for training and evaluation.

![Transformer Architecture](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*TPfHAcf7Afxip2pX2AL9qg.png)

## Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Components](#components)
  - [Tokenization](#tokenization)
  - [Dataset Preparation](#dataset-preparation)
  - [Model Architecture](#model-architecture)
  - [Training](#training)
- [Configuration](#configuration)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Bilingual Translation**: Translate sentences between two languages using a transformer model.
- **Custom Tokenization**: Implements a WordLevel tokenizer that handles special tokens and padding.
- **Flexible Configuration**: Easily adjustable parameters for sequence length, batch size, and training settings.
- **Training Visualization**: Integrated with TensorBoard for real-time monitoring of the training process.
- **Extensible Architecture**: Modular code structure allows for easy modification and experimentation.

## Dataset

The model uses the **Opus Books** dataset, which contains a variety of bilingual books, providing a rich source of parallel text for training and evaluation. The dataset supports multiple language pairs and is ideal for machine translation tasks.

### Dataset Structure

- **Source Language**: The language from which text is translated (e.g., English).
- **Target Language**: The language into which text is translated (e.g., French).
- The dataset is structured in a JSON format, where each entry contains translations for the specified languages.

## Architecture

The transformer architecture consists of an encoder and a decoder, with the following key components:

- **Multi-Head Self-Attention**: Allows the model to weigh the importance of different words in a sequence.
- **Positional Encoding**: Injects information about the position of tokens in the input sequence.
- **Feed-Forward Neural Networks**: Applies a fully connected feed-forward network to the outputs of the attention layer.
- **Layer Normalization and Residual Connections**: Improves training stability and efficiency.

### Transformer Architecture Diagram

![Transformer Architecture](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/12/Transformer-Architecture.jpg)

## Requirements

- Python 3.7+
- PyTorch 1.7+
- `torchvision` for data manipulation
- `datasets` for loading datasets
- `tokenizers` for tokenization
- `tqdm` for progress bars
- `tensorboard` for visualization

You can install the required packages using:

```bash
pip install torch torchvision datasets tokenizers tqdm tensorboard
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your_username/transformer-bilingual-translation.git
cd transformer-bilingual-translation
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. **Configuration**: Adjust the configuration settings in `config.py` to set your source and target languages, sequence length, batch size, and other parameters.

2. **Run Training**: Start the training process by running the `train.py` script:

   ```bash
   python train.py
   ```

3. **Monitor Training**: Use TensorBoard to monitor the training process. You can start TensorBoard with the following command:

   ```bash
   tensorboard --logdir=logs
   ```

## Components

### Tokenization

The tokenizer is built using the `tokenizers` library, employing a WordLevel model. It normalizes text by lowercasing and removing unnecessary Unicode characters. The tokenizer can be reused across multiple runs unless it is modified.

### Dataset Preparation

The `BilingualDataset` class is implemented to handle the loading and encoding of the bilingual dataset. It prepares input sequences, target sequences, and labels while ensuring the correct padding and special tokens are included.

### Model Architecture

The transformer model architecture is defined in the `transformer.py` file. It consists of an encoder and decoder with multi-head self-attention and feed-forward layers. The model leverages positional encodings to maintain the order of input sequences.

### Training

The training process is handled in the `train.py` file. It defines the training loop, optimizes the model using the Adam optimizer, and computes loss using cross-entropy with label smoothing. The model's progress is tracked and saved periodically.

## Configuration

Configuration settings are managed in the `config.py` file. Key parameters include:

- `source_lang`: Language code for the source language (e.g., "en").
- `target_lang`: Language code for the target language (e.g., "fr").
- `seq_len`: Maximum sequence length for input and target sequences.
- `batch_size`: Number of samples per batch.
- `num_epochs`: Number of training epochs.
- `lr`: Learning rate for the optimizer.
- `model_folder`: Directory to save trained model weights.

## Results

The model's performance can be evaluated on a validation set. After training, you can inspect the saved model weights to perform inference on new sentences.

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request. 
