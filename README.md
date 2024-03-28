# Transformer Model Implementation

This repository contains an implementation(not completed yet due to me being compute poor) of the Transformer model as described in the paper "Attention Is All You Need" by Vaswani et al. The Transformer model is a novel architecture for handling sequential data, particularly in the field of natural language processing (NLP). It introduces the concept of self-attention, allowing the model to weigh the importance of different parts of the input sequence when generating the output.

## Transformer Architecture Overview

The Transformer model consists of an encoder-decoder structure, with each part consisting of multiple layers. The encoder processes the input sequence and maps it to a sequence of continuous representations. These representations are then fed into the decoder, which generates the output sequence. The model is auto-regressive, meaning it consumes previously generated symbols as additional input when generating the next symbol.

### Key Components

- **Encoder**: Transforms the input sequence into a sequence of continuous representations.
- **Decoder**: Generates the output sequence based on the encoder's output and the previously generated symbols.
- **Self-Attention Mechanism**: Allows the model to focus on different parts of the input sequence when generating the output.
- **Positional Encoding**: Adds information about the position of tokens in the sequence, as the Transformer does not inherently understand the order of the sequence.

## Getting Started

To get started with this implementation, clone the repository and follow the instructions in the README file for setting up the environment and running the code.

## Resources

For a deeper understanding of the Transformer model and its components, refer to the following resources:

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)

## Technologies Used

- ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
- ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
- ![AI/ML](https://img.shields.io/badge/AI/ML-F7931E?style=for-the-badge&logo=tensorflow&logoColor=white)
- ![NLP](https://img.shields.io/badge/NLP-007ACC?style=for-the-badge&logo=nlp&logoColor=white)
- ![Transformers](https://img.shields.io/badge/Transformers-53B7DF?style=for-the-badge&logo=transformers&logoColor=white)
