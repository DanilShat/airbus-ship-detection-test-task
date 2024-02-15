# Airbus Ship Detection Challenge

This repository contains the code for my approach to the Airbus Ship Detection Challenge. The goal of this challenge is to detect ships in satellite images as quickly as possible.

## Approach

After reviewing various solutions, I found that a two-model approach yields the best results for this task. My approach consists of:


1. **Classification Model**: I used a **ResNet50** model and fine-tuned it for this task. This model determines whether an image contains any ships at all. It helps us quickly identify images without ships and focus our resources on images that are likely to contain ships.

2. **U-Net Model**: I used **MobileNetV2** as an encoder to train my U-Net model. This is an image segmentation model that is trained only on images containing ships. It is used to identify the exact location and shape of the ships in the images. I chose MobileNetV2 due to memory restrictions on Kaggle. For even better results, it would be beneficial to fine-tune a larger model like ResNet34 or ResNet50.

## Important Notes

For training my models, I did not use the entire dataset due to memory limitations on Kaggle. However, the results are quite satisfactory given the constraints. 

I have pre-trained these two models and provided an overview of their performance in the `model_inference_show` Jupyter notebook. 

Please note that the paths for the images are from Kaggle and are created for a Kaggle notebook. Due to the large memory usage of the original dataset (over 30GB), I did not download it to my repository.

I only uploaded the model for segmentation since the model for classification is too large to upload to GitHub, model_inference_example.ipynb uses both pre-trained models, I also uploaded these two models that I had already trained to [Kaggle](https://www.kaggle.com/datasets/purrrplehaze/airbus-ship-detection-models)





   
