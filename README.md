# Transformator
## hubert-large-ls960-ft
Is failed because not enough system recruitment. 
# Deep Learing Algorithim
## LSTM
   Accuracy  F1 Score    Recall  Precision   ROC AUC       MCC 
   
0  0.749577  0.749592  0.749577   0.750922  0.894152  0.624828 

Confusion Matrix: 

[[315  60  48] 
 [ 42 272  65] 
 [ 42  39 299]] 
![image](https://github.com/ArtunKARA/MusicEmotionRecognition/assets/76822513/3bad2473-3cad-4bee-b14b-db98919790d3)

ROC AUC Score (Macro): 0.8941684841943457
## LSTM(cross validation)
37/37 [==============================] - 2s 40ms/step

   Accuracy  F1 Score    Recall  Precision   ROC AUC       MCC
   
0  0.762913  0.763886  0.762913   0.776338  0.906839  0.649359

Confusion Matrix:

[[322  75  19]

 [ 42 309  23]
 
 [ 46  75 270]]
 
![image](https://github.com/ArtunKARA/MusicEmotionRecognition/assets/76822513/14d266b9-4117-49cf-aaa7-6ea846716f55)

ROC AUC Score (Macro): 0.9066480363403912
## LSTM(ner1280 epo500 cross validation)
   Accuracy  F1 Score   Recall  Precision   ROC AUC       MCC
   
0   0.84928  0.849185  0.84928   0.849224  0.956962  0.773868

Confusion Matrix:

[[356  31  17]

 [ 33 327  36]
 
 [ 24  37 320]]
 
![image](https://github.com/ArtunKARA/MusicEmotionRecognition/assets/76822513/a2c7c119-09d7-4169-87b5-a7c12e017196)

ROC AUC Score (Macro): 0.9569365871499508

## LSTM(1024-0.5_1024-0.5_cross_validation)

   Accuracy  F1 Score    Recall  Precision   ROC AUC       MCC
   
0  0.847587   0.84753  0.847587   0.850642  0.959864  0.772467 

Confusion Matrix: 

[[362  32  10] 

 [ 38 334  22] 
 
 [ 30  48 305]] 
 ![image](https://github.com/ArtunKARA/MusicEmotionRecognition/assets/76822513/d6323b2b-cf43-4eff-b942-c448320187f6)

ROC AUC Score (Macro): 0.9598399299521306


## CNN_20
Mean Evaluation Metrics: 

Accuracy: 0.7026 ± 0.0317 

F-measure: 0.6905 ± 0.0396 

Recall: 0.6987 ± 0.0308 

Precision: 0.7486 ± 0.0137 

MCC: 0.5736 ± 0.0374 

AUC: 0.5215 ± 0.0138 

![image](https://github.com/ArtunKARA/MusicEmotionRecognition/assets/76822513/e754d04d-a427-479f-9693-b7332b2e83d4)

