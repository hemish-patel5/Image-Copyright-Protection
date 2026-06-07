# Transformer-Based Invisible Image Watermarking

A deep learning framework for robust image copyright protection using Transformer-based architectures.

## Overview

Digital images are easily copied, edited, and redistributed, creating significant challenges for copyright protection. This project proposes an invisible watermarking approach leveraging **Transformer architectures** to embed copyright information into images. 

Unlike traditional methods that rely on local convolutional features, this system uses self-attention mechanisms to model global relationships across image regions, improving the watermark's robustness against common transformations such as JPEG compression, cropping, resizing, and noise.

## Project Goal
To develop a practical deep learning framework that balances:
* **Imperceptibility:** The watermarked image remains visually identical to the original.
* **Robustness:** The watermark remains detectable after image processing and attacks.
* **Recoverability:** Accurate extraction of the copyright message.

## Methodology

The system follows a structured encoder-decoder pipeline:

1.  **Embedding Network:** Uses Transformer blocks to fuse a binary copyright watermark with the original image.
2.  **Attack Simulation Layer:** Simulates common online distortions (e.g., JPEG compression, blur, cropping) to train the model to be robust.
3.  **Extraction Network:** Learns to predict the hidden binary watermark from the attacked watermarked image.

### Workflow
![Transformer-based invisible watermarking workflow](https://placeholder-link-to-figure-1.png)

## Evaluation Metrics

The framework is evaluated using a combination of image quality and watermark recovery metrics:

| Metric Type | Metric | Description |
| :--- | :--- | :--- |
| **Image Quality** | PSNR, SSIM | Measures visual similarity to the original image. |
| **Watermark Recovery** | BER, Accuracy | Measures the correctness of the extracted watermark. |

## Requirements

* **Language:** Python
* **Frameworks:** PyTorch or TensorFlow/Keras
* **Libraries:** NumPy, OpenCV, Matplotlib, scikit-image
* **Hardware:** Recommended NVIDIA T4 GPU (or equivalent) for training.

## Usage

1.  **Preparation:** Place images in your dataset directory and define the binary watermark message (e.g., 32/64/128 bits).
2.  **Training:** Run the embedding and extraction training scripts. The model uses a total loss function balancing image reconstruction loss, watermark extraction loss, and perceptual quality.
3.  **Inference:** Pass the original image and watermark through the trained Embedding Network to produce the protected image.
4.  **Verification:** Pass the (possibly attacked) image through the Extraction Network to recover the watermark.

## References

* Cox, I. J., et al. (2007). *Digital watermarking and steganography*.
* Hosny, K. M., et al. (2024). Digital image watermarking using deep learning: A survey.
* Liu, Z., et al. (2021). Swin Transformer: Hierarchical vision Transformer using shifted windows.
* Tancik, M., et al. (2020). StegaStamp: Invisible hyperlinks in physical photographs.
* Zhu, J., et al. (2018). HiDDeN: Hiding data with deep networks.

---
