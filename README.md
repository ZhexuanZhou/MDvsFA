# MDvsFA
PyTorch implementation of  ICCV2019 paper Miss Detection vs. False Alarm: Adversarial Learing for Small Object Segmentation in Infrared Images.

# Guide
1. Creating the following folders:
    - training_results: this folder is to contain all the images of evaluation phases, to visualize the performance of model.
    - test_results: this folder is to contain the images during test phases.
    - logs: this folder is to contain all logs during training.
    - saved_models: to save the weight after each epoch.
    
    The following command is to create fodler under the root of repository:
    
    ```bash
    mkdir training_results test_results logs saved_models
    ``` 
2. Dataset:
    The [official implementation](https://github.com/wanghuanphd/MDvsFA_cGAN) offers the dataset, the structure has to be:
    ```
    root
        data
            test_gt
            test_ort
            training
    ```
3. Using following command to train:
    ```python
    python train.py
    ```
    all the training parameters have default values.
  
4. Using following command to test:
    ```python
    python test.py
    ```   
 
