# SMA Breast Tumor Segmentation
## Introduction to SAM

The **Segment Anything Model (SAM)** is a cutting-edge segmentation model capable of handling diverse and complex segmentation tasks. Its versatility and accuracy make it suitable for a wide range of applications, especially in fields requiring precise delineation of features within images.
## Introduction to SAM

The **Segment Anything Model (SAM)**, a novel creation by researchers from Meta, revolutionizes the concept of segmentation in machine learning. Extending beyond traditional language prompting, SAM introduces visual prompting, allowing for zero-shot segmentation with various prompt inputs.

### SAM at a Glance
SAM redefines segmentation tasks with its ability to process prompts that range from points, free text, and boxes, to masks. It's designed to cater to an array of segmentation tasks like semantic segmentation and edge detection, all through intuitive prompting.

![image](https://github.com/AliAmini93/SMA-Breast-Tumor-Segmentation/assets/96921261/33ed7c12-75db-48ab-91b3-727f12231d78)
*Image Source: SAM Paper*

### Key Components of SAM:
- **Image Encoder:** Computes image embeddings. It's used once per image due to its computational intensity.
- **Prompt Encoder:** Lightweight and handles sparse prompts like points and boxes.
- **Convolutional Layer:** Processes dense prompts, such as masks.
- **Mask Decoder:** A vital, lightweight component that predicts the mask, integrating image and prompt embeddings.

SAM's training enables it to generate valid masks for a wide range of prompts, even under ambiguity. This feature not only makes SAM ambiguity-aware but also capable of predicting multiple masks for a single prompt.

## Application in Breast Cancer Segmentation:
In this project, SAM is applied to the challenging domain of **Breast Tumor diagnosis**. By segmenting medical images from the Breast Cancer [dataset](https://huggingface.co/datasets/nielsr/breast-cancer), SAM plays a crucial role in identifying and delineating cancerous tissues.

