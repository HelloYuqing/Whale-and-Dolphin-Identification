# Whale-and-Dolphin-Identification
![image](https://user-images.githubusercontent.com/69694512/204550639-4990a25c-cb34-4e2f-a802-f0eb95445f2c.png)

In this Projecr, we will build an automated model for image recognition of whales and dolphins in order to reduce recognition time.

🎐This project belongs to computer vision/image retrieval, so the recommended model or library: **EfficientNet/NFNet/Arcface/GeM**

🎐Data: The data of images are more than 15,000 unique individual marine mammals of 30 different species collected from 28 different research institutions. The official training set has about 51,033 pictures, the test set has about 28,000 pictures, and the total data set is 62GB. Ocean researchers have manually identified and assigned individual_id to individuals, and our task is to correctly identify these individuals in the image.

🎐Evaluation criteria: *MAP@5*. For each image in the test set, up to 5 individual_id labels can be predicted. There are some individuals in the test set that did not appear in the training data, and these individuals should be predicted as new_individual.

## Data Explanation
https://www.kaggle.com/competitions/happy-whale-and-dolphin

train_images/ - a folder containing training images

train.csv - provides species and individual_id for each training image

test_images/ - a folder containing test images;

sample_submission.csv - a properly formatted sample submission file

# Solution Ideas
Our solution for this project uses the **multi-model fusion of EfficientNet_B6 / EfficientNet_V2_L / NFNet_L2** as we are only focus on the fullbody and backfins of the whale. We use Yolov5 to crop the fullbody and backfins of the original image (refer to Backfin Detection with Yolov5 | Kaggle), and trained **tf_efficientnet_b6_ns, tf_efficientnetv2_l_in21k, eca_nfnet_l2** three models on the cropped picture, then added GeM Pooling and Arcface head, all models will output a 512-dimensional Embedding, we will Concatenated obtained a **512×9=4608** dimensional vector, alculated the distance and selected the nearest neighbor result. We also used new_individual replacement for post-processing.

For the model, we chose allenai/longformer-base-4096 and microsoft/deberta-large versions, and then connected a Dropout layer and a Linear layer.

# Coding Part

## modeling
```
print(hello)
```



































































































