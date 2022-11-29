# Whale-and-Dolphin-Identification
![image](https://user-images.githubusercontent.com/69694512/204549502-a8476fb3-a2d4-4281-a128-7eba53786ed2.png)

In this Projecr, we will build an automated model for image recognition of whales and dolphins in order to reduce recognition time.

🎐This project belongs to computer vision/image retrieval, so the recommended model or library: **EfficientNet/NFNet/Arcface/GeM**
🎐Data: The data of images are more than 15,000 unique individual marine mammals of 30 different species collected from 28 different research institutions. The official training set has about 51,033 pictures, the test set has about 28,000 pictures, and the total data set is 62GB. Ocean researchers have manually identified and assigned individual_id to individuals, and our task is to correctly identify these individuals in the image.
🎐Evaluation criteria: *MAP@5*. For each image in the test set, up to 5 individual_id labels can be predicted. There are some individuals in the test set that did not appear in the training data, and these individuals should be predicted as new_individual.
