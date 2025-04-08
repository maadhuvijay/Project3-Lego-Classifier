# Project-3-AI-Bootcamp


## Dataset collection

1. Source: Lego images from Kaggle [https://www.kaggle.com/code/stpeteishii/lego-bricks-classify-torch-linear-conv2d]
2. Image selection: Each lego piece has 800 images. Total 1132 Images for 4 lego brick pieces -2357, 3001,3010,3022 [283 each] were chosen
3. Lego images used: The front and top angle of pieces were chosed. 

## Data Visualization
The lego image was printed / plotted to ensure the import was successful.

![image](https://github.com/user-attachments/assets/a0f4e6ec-d309-4c9a-a215-95b2d18cca7f)



## Data Pre-processing

Below pre-processing steps were done on the lego images. 

#### Re-sizing of the images:

Initially the images were re-sized to 60 x 60 but then there was a difference in the image clarity. The edges of the images were not as clear as the ones with bigger size. Later the image size was changed to 224 x 224 as part of enhancing the image processing

   ![image](https://github.com/user-attachments/assets/3f502e36-169f-4c85-b57c-fd67e4cc6ec6)
   ![image](https://github.com/user-attachments/assets/9875ea42-6ec0-4648-9631-bc51be5eb111)

#### Edge detection and Dilation of edges:


   **Edge Detection** detects sharp changes in intensity Canny Edge Detection was implemented

   ![image](https://github.com/user-attachments/assets/d0bd86d0-70b2-4e69-9eab-eb39af337b87)

   **Dilation** makes the edges thicker for making edges more “visible” or easier to track.
   
   ![image](https://github.com/user-attachments/assets/c1a99fdf-dd43-4dda-812e-dc45dcfe72eb)

  
#### Normalization of the images 

    To change the image pixel values to be between 0 and 1.   

#### Padding

Padding turned out be an important step in getting more accuracy in this model training

The images were padded by re-sizing it into 128 x 128 within the image frame.This was done because, after augmentation images with bigger brick size were not fitting inside the frame and created extra pixels and had part of the image truncated, distorted.

**Before padding** [Note: During the augmentation analysis, I intermittently removed the edge detection and dilating edges transformations from pre-processing and that is why the before padding images are different]

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

Problems identified with Random roatation, and random translation transformations [ Images with extra pixels and blank images]

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

4 models were built : 1 main model and 3 models to analyze the below optimatization techniques.

1. **L1 regularization** 

   For this analysis, the model was run with L1 regularizer values as 0.01. This optimization technique gave erratic results witht the accuracy and loss curves not close to the expected results. 

  ![image](https://github.com/user-attachments/assets/ca3cf1ac-aec6-4a99-b0b2-ea6dd4b96838)
  ![image](https://github.com/user-attachments/assets/a73e5220-09ca-4283-9609-b1447dfb0aa6)


2. **L2 regularization**

   For this analysis, the model was run with L2 regularizer values as 0.01.The results were analyzed with the accuracy and loss values plotted as curves. In this analysis the accuracy curves seemed to be somewhat in sync and the loss values were lesser comparatively to the L1 regularization.

   ![image](https://github.com/user-attachments/assets/a2d91fe9-8013-4d28-83c5-b4bd6521b0bd)
   ![image](https://github.com/user-attachments/assets/9eabd546-c69d-4c30-97a0-d66116c21692)


4. **Dropout**

   The dropouts model also showed a similar results like L2 regularization technique without any deviated results like L1 regularization

5. For the **Main model**, it was decided 

   1. To add a **dropout layer with value 0.3** after the dense layer
   2. To add **L2 regularizer with value 0.0001** to the layers with higher parameters from the above results. 

## Model evaluation

The model was evaluated by measuring the "Accuracy and the loss function" parameters and by plotting these values as diagnostic curves. 

Observation 1:
Immediately after removing the random rotation augmentation function (which caused blank images and distorted images), there was a difference in the diagnostic results. 

![image](https://github.com/user-attachments/assets/8b01c415-0439-46d7-bd5b-fc0f773e4598)
![image](https://github.com/user-attachments/assets/6765b59d-d968-4078-b241-c64161288765)

Observation 2:




## Model prediction

## Gradio - User Inferface



## Reference


   
   
   

