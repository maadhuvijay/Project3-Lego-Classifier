# Project-3: AI - Powered Lego Classifier
The AI-powered LEGO classifier will use a neural network model to classify the LEGO images into its corresponding design number.

Many households accumulate large collections of LEGO pieces that children lose interest in, while parents are unsure how to repurpose them. A model 
that identifies individual LEGO pieces and recommends sets that can be built using them provides a practical solution, making better use of existing 
LEGO collections.

## Tech Stack

**Machine Learning & Deep Learning**
1. **TensorFlow / Keras:*
   Core deep learning framework for building and training neural networks.

   tensorflow, keras, layers, optimizers, callbacks like EarlyStopping

2. **scikit-learn:**
   Utilities for data preprocessing and model evaluation.

   train_test_split, OneHotEncoder

**Computer Vision & Image Processing**
1. OpenCV (cv2):
   Image loading, manipulation, and preprocessing.

2. PIL (Pillow):
   Additional image handling capabilities.

3. Matplotlib:
   Image visualization and plotting support.

**User Interface & Interaction**
1. **Gradio:**
   For building interactive web UIs for your model (e.g., upload image, get prediction).

2. **pyttsx3:**
   Local text-to-speech engine (offline, unlike OpenAI’s TTS).
   
**Networking / API**

1. **OpenAI API (openai):**
   For integrating advanced services like:
   Text-to-Speech (TTS)

## Dataset installation

1. Link for the google drive shared folder is available in the resources sections. Those images needs to be downloaded used in the code.
2. The Gradiio demo images are available in a zip folder in the Data folder and should be used for uploading the images for predictions. Though these images are from the Lego sites, the size of these images matches with the training data. Bigger sized images might need more size resizing and processing.
   
## Dataset collection

1. Source: Lego images from Kaggle [https://www.kaggle.com/code/stpeteishii/lego-bricks-classify-torch-linear-conv2d]
2. Image selection: Each lego piece has 800 images. Total 1132 Images for 4 lego brick pieces -2357, 3001,3010,3022 [283 each] were chosen
3. Lego images used: The front and top angle of pieces were chosed. 

## Data Visualization
The lego image was printed / plotted to ensure the import was successful.

![image](https://github.com/user-attachments/assets/a0f4e6ec-d309-4c9a-a215-95b2d18cca7f)



## Data Pre-processing

The images were pre-processed implementing the below steps.

#### Re-sizing of the images:

Initially the images were re-sized to 60 x 60 but then there was a difference in the image clarity. The edges of the images were not as clear as the ones with bigger size. Later the image size was changed to 224 x 224 as part of enhancing the image processing

   ![image](https://github.com/user-attachments/assets/3f502e36-169f-4c85-b57c-fd67e4cc6ec6)
   ![image](https://github.com/user-attachments/assets/9875ea42-6ec0-4648-9631-bc51be5eb111)

#### Edge detection and Dilation of edges:


   **Edge Detection** detects sharp changes in intensity and Canny Edge Detection was implemented

   ![image](https://github.com/user-attachments/assets/d0bd86d0-70b2-4e69-9eab-eb39af337b87)

   **Dilation** makes the edges thicker for making edges more “visible” or easier to track.
   
   ![image](https://github.com/user-attachments/assets/c1a99fdf-dd43-4dda-812e-dc45dcfe72eb)

  
#### Normalization of the images 

    The images were converted to array and normalized by dividing by 255 to change the pixel values to be between 0 and 1.   

#### Padding

Padding turned out be an important step in getting more accuracy in this model training

The images were padded by re-sizing it into 128 x 128 within the image frame.This was done because, after augmentation, the images with bigger brick size were not fitting inside the frame and created extra pixels and had part of the image truncated or distorted.

**Before padding** [Note: During the augmentation analysis, I intermittently removed the edge detection and dilating edges transformations from pre-processing and that is the reason the before padding images are different]

   ![image](https://github.com/user-attachments/assets/f9450e9a-2ff6-4b4a-8142-be0f784b7669)
   ![image](https://github.com/user-attachments/assets/7f858ede-bf30-4ea1-82e9-136d0666831b)
   ![image](https://github.com/user-attachments/assets/26256330-4523-4e7e-a396-9e4c64d19b3a)

**After Padding** [These after -padding images were taken once the edge detection and Dilation edges preprocessing steps were added back]


   ![image](https://github.com/user-attachments/assets/11194b5a-144f-4ac5-af83-abf70c1ff81e)
   ![image](https://github.com/user-attachments/assets/fe8aa147-d9af-4514-b9b7-becadd5bafc6)
   ![image](https://github.com/user-attachments/assets/87b1be22-95c9-4fc9-8423-eac1dddf2893)
   ![image](https://github.com/user-attachments/assets/b3bebc23-b2b4-4efd-bccc-144fd4e496ca)


## Image Augmentation
The images were augmented to add more diverse images to the dataset.

Below **random transformations** were initially considered. 

   1. RandomRotation(0.2)
   2. RandomTranslation(0.1, 0.1)
   3. RandomFlip('horizontal'),
   4. Random Zoom

     
Problems identified with Random roatation,Random contrast, brightness and random translation transformations [ Images with extra pixels and blank images]

   ![image](https://github.com/user-attachments/assets/aeee6718-aad3-43f9-b2df-e96930669456)
   ![image](https://github.com/user-attachments/assets/015ae759-b606-4154-8bc8-818c83d299e4)
   ![image](https://github.com/user-attachments/assets/c2d5e244-e9bf-437b-b94a-a768ed007cf2)
   ![image](https://github.com/user-attachments/assets/fce87ab5-a20f-4a44-bb92-4933df8111c1)


**Updated Augmentation:** Random rotation and random translation was removed and Random flip (Vertical) was added instead which resolved these problems 
   
1. Random flip (Horizonatl)
2. Random Zoom
3. Random flip (Vertical)

## Training testing data split
1. The normalized images are the X data.
2. The design ID of the lego piece is the target column (y)

## Converting the y-data to numerical data
1. The y-column with Lego design id's were encoded to numerical data through **One-hot encoding**. 


# Model selection


**CNN model** was built and trained to classify these Lego images due to its capability of classifying images. 

The model was created with 3 Convulation layers, 1 flatten layer and 2 dense layers and one of them being the output layer.

![image](https://github.com/user-attachments/assets/d8e1b5bc-d1b6-45b7-8c44-094928db45b3)



## Model training

The model was trained with

1. Activation funcitons **"relu"** on the convolution layers and **"softmax"** for the output layer as this is a multi-class image classification problem.
2. The model used **"categorical cross entropy"** for the loss function and measured **"Accuracy"** as the validation metric.
3. **Early stopping** is used as a training technique to prevent overfitting in the model.
4. The **learning rate** in the optimizer "Adam" was adjusted. I noticed a lower learning rate achieved better results opposed to higher learning rates. 

## Model Optimization

For optimization and achieve better results, regularization technique was explored. 4 models were built : 1 main model and 3 models to analyze the below optimatization techniques.

1. **L1 regularization** 

   For this analysis, the model was run with L1 regularizer values as 0.01. This optimization technique gave erratic results witht the accuracy and loss curves not close to the expected results. 

  ![image](https://github.com/user-attachments/assets/ca3cf1ac-aec6-4a99-b0b2-ea6dd4b96838)
  ![image](https://github.com/user-attachments/assets/a73e5220-09ca-4283-9609-b1447dfb0aa6)


2. **L2 regularization**

   For this analysis, the model was run with L2 regularizer values as 0.01.The results were analyzed with the accuracy and loss values plotted as curves. In this analysis the accuracy curves seemed to be somewhat in sync and the loss values were less comparative to the L1 regularization.

   ![image](https://github.com/user-attachments/assets/a2d91fe9-8013-4d28-83c5-b4bd6521b0bd)
   ![image](https://github.com/user-attachments/assets/9eabd546-c69d-4c30-97a0-d66116c21692)


4. **Dropout**

   The dropouts model also showed a similar results like L2 regularization technique without any deviated results like L1 regularization

5. For the **Main model**, it was decided 

   1. To add a **dropout layer with value 0.3** after the dense layer
   2. To add **L2 regularizer with value 0.0001** to the layers with higher parameters from the above results. 

## Model evaluation

The model was evaluated by measuring the "Accuracy and the loss function" parameters and by plotting these values as diagnostic curves. 

**Observation 1:**
Immediately after removing the random rotation augmentation function (which caused blank images and distorted images), we could see better diagnostic results. 

![image](https://github.com/user-attachments/assets/8b01c415-0439-46d7-bd5b-fc0f773e4598)
![image](https://github.com/user-attachments/assets/6765b59d-d968-4078-b241-c64161288765)

**Observation 2:**
When I removed all the random transformations and just had the random flip (horizontal), there was no image distortions as the images didn't have much change. Looking at the below graphs there was a gap between the validation loss and the training loss. This meant the training data is insufficient.

![image](https://github.com/user-attachments/assets/5c0e8d5c-76b7-42a7-8bcd-61e5cc1a69ad)
![image](https://github.com/user-attachments/assets/ef4706de-1c56-4178-9ce4-c6075d06aee0)

**Observation 3:**
After the above observation, I increased the training data by increasing the augmentation count for each image. With couple times trying out with different images count, was able to see the below perfect graphs. But since the image is not going through a varied random transformation through this approach, I decided to add more random transformations to the augmentation process. 

![image](https://github.com/user-attachments/assets/c0194011-b177-4139-8554-ef177fd4ce29)
![image](https://github.com/user-attachments/assets/1d43db50-da4b-4fd2-8184-6005fd61a52e)

**Final test results:**

![image](https://github.com/user-attachments/assets/bc568204-5cdb-486e-8acd-f871238e8828)

![image](https://github.com/user-attachments/assets/eb37b00c-c919-4641-8a68-96986e4b01b7)
![image](https://github.com/user-attachments/assets/d51abe78-0376-4505-9110-a87225480811)

## Model prediction

The model was tested with some of the X_test image data and confirmed the predictions are correct. 

![image](https://github.com/user-attachments/assets/d3f4d99d-a493-410a-981a-7190e31ea6a1)

![image](https://github.com/user-attachments/assets/a2bc7128-7184-429e-a678-8d7280206e1d)
![image](https://github.com/user-attachments/assets/98e6d1f3-6885-4d92-8b4f-f46302225517)



## Gradio - User Interface

Gradio app interface was used to upload and test input demo images. The app uploads an image  through the input component and displays the lego probabilities and a text to speech conversion of the predicted lego id as the output components. 

![image](https://github.com/user-attachments/assets/2d3b6549-e60c-42a6-8230-53d24e282479)


**Input image processing**
The input images were also pre-processed with below steps, the same way as the dataset and then fed to the model for prediction.

1. Image resizing.
2. Edge detection.
3. Dilation of edges
4. Padding of images
5. Normalization of images
6. Add the channel dimensions



**OpenAI Text-to-Speech (TTS)** 

As part of enhancing the user interaction experience for the LEGO Classifier Project, OpenAI's Text-to-Speech (TTS) technology is integrated into the Gradio interface to provide natural-sounding audio feedback. This allows the system to verbally announce the identified LEGO bricks making it more engaging and accessible—especially for kids and visually impaired users.

## Reference
1. [https://www.kaggle.com/code/stpeteishii/lego-bricks-classify-torch-linear-conv2d]
2. https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/
3. https://medium.com/intelligentmachines/convolutional-neural-network-and-regularization-techniques-with-tensorflow-and-keras-5a09e6e65dc7
4. https://chatgpt.com/


   
   
   

