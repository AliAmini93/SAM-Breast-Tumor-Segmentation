# SMA Breast Tumor Segmentation
## Introduction to SAM

The **Segment Anything Model (SAM)** is a cutting-edge segmentation model capable of handling diverse and complex segmentation tasks. Its versatility and accuracy make it suitable for a wide range of applications, especially in fields requiring precise delineation of features within images.
## Introduction to SAM

The **Segment Anything Model (SAM)**, a novel creation by researchers from Meta, revolutionizes the concept of segmentation in machine learning. Extending beyond traditional language prompting, SAM introduces visual prompting, allowing for zero-shot segmentation with various prompt inputs.

### SAM at a Glance
SAM redefines segmentation tasks with its ability to process prompts that range from points, free text, and boxes, to masks. It's designed to cater to an array of segmentation tasks like semantic segmentation and edge detection, all through intuitive prompting.

| ![SAM paper architecture](https://github.com/AliAmini93/SMA-Breast-Tumor-Segmentation/assets/96921261/6e659861-0c38-4469-9cd5-3eb16dafcac6) |
| :--: |
| Architecture Overview of SAM ([Paper](https://arxiv.org/pdf/2304.02643.pdf)) |

### Key Components of SAM:
- **Image Encoder:** Computes image embeddings. It's used once per image due to its computational intensity.
- **Prompt Encoder:** Lightweight and handles sparse prompts like points and boxes.
- **Convolutional Layer:** Processes dense prompts, such as masks.
- **Mask Decoder:** A vital, lightweight component that predicts the mask, integrating image and prompt embeddings.

SAM's training enables it to generate valid masks for a wide range of prompts, even under ambiguity. This feature not only makes SAM ambiguity-aware but also capable of predicting multiple masks for a single prompt.

## Application in Breast Cancer Segmentation:
In this project, SAM is applied to the challenging domain of **Breast Tumor diagnosis**. By segmenting medical images from the Breast Cancer [dataset](https://huggingface.co/datasets/nielsr/breast-cancer), SAM plays a crucial role in identifying and delineating cancerous tissues.

## Model Initialization and Visualization Functions

This section of the code involves initializing the SAM model and defining a series of functions for various visualization tasks.

### Initialization:
- **SAM Model and Processor:** We initialize the SAM model and its processor, which are crucial for handling the segmentation tasks.

### Visualization Functions:
- **Display Functions:** A series of functions (`display_mask`, `display_box`, `display_boxes_on_img`, etc.) are defined to visualize the segmentation results. These functions are used to overlay masks, bounding boxes, and points over the images, providing a clear view of the model's segmentation capabilities.

This set of functions forms the backbone of our visualization strategy, allowing us to effectively present and analyze the model's outputs.

## Image Loading, Preprocessing, and Segmentation

This part of the code deals with loading an image, preprocessing it, and performing segmentation using SAM.

### Steps Involved:
- **Image Loading:** An image is loaded from a URL for segmentation tasks.
- **Point Visualization:** Specific points are visualized on the image, which acts as prompts for the SAM model.
- **Preprocessing:** The SAM processor is used to preprocess the image and the points.
- **Segmentation:** The SAM model performs the segmentation based on the processed inputs.
- **Result Visualization:** The segmented masks along with their Intersection Over Union (IOU) scores are displayed on the image.

![image](https://github.com/AliAmini93/SMA-Breast-Tumor-Segmentation/assets/96921261/83a4010f-ae57-47e5-b5a4-650ac4a26e5f)
![image](https://github.com/AliAmini93/SMA-Breast-Tumor-Segmentation/assets/96921261/fa07e4d1-8e63-4aa0-887d-1e12cb215dce)

## Fine Tuning the SAM for Breast Cancer Segmentation 

This segment of the code handles the downloading and initial preparation of the Breast Cancer dataset.

### Key Actions:
- **Dataset Acquisition:** The Breast Cancer dataset is extracted. It is assumed to be in a compressed `.tar.gz` format.
- **Path Loading:** The paths for all images and their corresponding labels within the dataset are loaded.

## Data Generator for Model Fine-Tuning

The `DataGenerator` class plays a crucial role in preparing the data for fine-tuning the SAM model.

### Functionality:
- **Initialization:** Sets up the generator with the dataset path, processor, and paths to the images and labels.
- **Data Processing:** Iterates over each image-label pair, processes them using the SAM processor and prepares the inputs for the model.
- **Bounding Box Generation:** Calculates bounding boxes from the ground truth masks, adding slight perturbations for robustness.

## Training Dataset Creation

The next step in the process is the creation of the TensorFlow dataset for training the SAM model.

### Key Steps:
- **Output Signature Definition:** Specifies the structure and data types of the dataset, ensuring compatibility with the SAM model.
- **Data Generator Instantiation:** The `DataGenerator` class is instantiated with the necessary parameters.
- **Dataset Generation:** A TensorFlow dataset is created from the generator, adhering to the defined output signature.

## Training Dataset Configuration

After creating the training dataset, it's configured for optimal performance during the training process.

### Configuration Steps:
- **Caching:** The dataset is cached to improve data loading speed.
- **Shuffling:** Data is shuffled using a buffer to ensure randomness in the training batches.
- **Batching:** The dataset is divided into batches of a specified size for training.
- **Prefetching:** Data is prefetched using TensorFlow's AUTOTUNE feature for efficient utilization of hardware resources during training.

## DICE Loss Function Implementation

For the model training, we implement the DICE loss function, which is particularly effective for segmentation tasks.

### Implementation Details:
- **Inspiration:** This implementation is inspired by the [MONAI DICE loss](https://docs.monai.io/en/stable/_modules/monai/losses/dice.html#DiceLoss).
- **Functionality:** The DICE loss calculates the overlap between the predicted segmentation and the ground truth. It uses a sigmoid activation on the predictions and computes the intersection and union of the predictions and true values.
- **Batch Handling:** The function is designed to handle both single and batch predictions.

## Model Initialization and Training Step Function

We initialize the SAM model and define the training step function, which is integral to the model's training process.

### Initialization and Configuration:
- **SAM Model:** Initialized from a pre-trained state.
- **Optimizer:** Adam optimizer with a learning rate of `1e-5`.
- **Layer Configuration:** Specific layers, such as the vision and prompt encoders, are set to non-trainable to maintain their pre-trained states.

### Training Step Function:
- **Functionality:** The function takes the inputs, passes them through the SAM model, and calculates the DICE loss.
- **Gradient Update:** It computes gradients and applies them to the trainable variables, facilitating the model's learning process.

## Model Training and Inference

The concluding part of our workflow involves training the [sam-vit-large model](https://huggingface.co/facebook/sam-vit-large) and using it for inference on a new image.

### Training:
- **Loop Over Epochs:** The model undergoes training for a predefined number of epochs.
- **Loss Calculation:** At the end of each epoch, the training loss is calculated and displayed.

### Inference:
- **Image Selection:** An image is selected from the dataset for the inference process.
- **Image Processing:** This image is processed using the SAM processor to prepare it for the model.
- **Model Inference:** The SAM model performs inference on the processed image, generating predicted masks and IoU scores.
- **Visualization:** The results, including the masks and IoU scores, are visualized on the image.

This final step demonstrates the practical application of the trained SAM model in segmenting and analyzing new images.

![image](https://github.com/AliAmini93/SMA-Breast-Tumor-Segmentation/assets/96921261/84b0ab6d-7bb8-4805-8abf-5646a783dc0c)

## Acknowledgements

This project benefited from the insightful notebooks created by [Niels Rogge](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb) and [Younes Belkada](https://github.com/huggingface/notebooks/blob/main/examples/segment_anything.ipynb).



