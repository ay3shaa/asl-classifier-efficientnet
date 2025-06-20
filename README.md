# ASL Alphabet Classifier

A deep learning project to classify American Sign Language (ASL) alphabet images using **EfficientNetB1**, **TensorFlow**, and **transfer learning techniques**. The model is trained and fine-tuned on an image dataset to recognize 29 hand signs (A-Z, plus "del", "space", and "nothing").


---

## Features

-  ASL hand sign classification using image input
-  Transfer learning with **EfficientNetB1** (pretrained on ImageNet)
-  Fine-tuning to improve accuracy
-  Data augmentation for better generalization
-  Saves the best model using callbacks
-  Automatically stops training when no improvement (EarlyStopping)
