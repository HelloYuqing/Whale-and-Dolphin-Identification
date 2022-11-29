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

# Coding Parts

## Modeling
```
class HappyWhaleModel(nn.Module):
    def __init__(self, model_name, embedding_size, pretrained=True):
        super(HappyWhaleModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained) # 创建模型
        # 获取 in_features，以及置空最后两层
        if 'efficientnet' in model_name:
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
            self.model.global_pool = nn.Identity()
        elif 'nfnet' in model_name:
            in_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Identity()
            self.model.head.global_pool = nn.Identity()

        self.pooling = GeM() # GeM Pooling
        # bn层 + dense层
        self.embedding = nn.Sequential(
                            nn.BatchNorm1d(in_features),
                            nn.Linear(in_features, embedding_size)
                            )
        # arcface
        self.fc = ArcMarginProduct(embedding_size,  # in_features
                                   CONFIG["num_classes"], # out_features
                                   s=CONFIG["s"],  # scale
                                   m=CONFIG["m"],  # margin
                                   easy_margin=CONFIG["easy_margin"],  # easy_margin模式
                                   ls_eps=CONFIG["ls_eps"]) # label smoothing

    def forward(self, images, labels):
        '''
        train/valid
        '''
        features = self.model(images) # backbone 
        pooled_features = self.pooling(features).flatten(1) # gem pooling
        embedding = self.embedding(pooled_features) # embedding
        output = self.fc(embedding, labels) # arcface
        return output

    def extract(self, images):
        '''
        test
        '''
        features = self.model(images) # backbone 
        pooled_features = self.pooling(features).flatten(1) # gem pooling
        embedding = self.embedding(pooled_features) # embedding
        return embedding
```



## Data Augmentation
```
# 数据增强
data_transforms = {
    "train": A.Compose([
        A.Affine(rotate=(-15, 15), translate_percent=(0.0, 0.25), shear=(-3, 3), p=0.5), # 仿射变换
        A.RandomResizedCrop(CONFIG['img_size'], CONFIG['img_size'], scale=(0.9, 1.0), ratio=(0.75, 1.333)), # 随机裁剪 + Resize
        A.ToGray(p=0.1), # 灰度
        A.GaussianBlur(blur_limit=(3, 7), p=0.07), # 高斯模糊
        A.GaussNoise(p=0.07), # 高斯噪音
        A.RandomGridShuffle(grid=(2, 2), p=0.28), # 图像网格随机打乱排版
        A.RandomBrightnessContrast(p=0.5), # 亮度、对比度
        A.HorizontalFlip(p=0.1), # 水平翻转
        # 归一化
        A.Normalize( 
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.),

    "valid": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']), # Resize
        # 归一化
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.
        )

}
```

## ArcFace
```
# Arcface
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, 
                 m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features  # input的维度
        self.out_features = out_features # output的维度
        self.s = s # re-scale
        self.m = m # margin
        self.ls_eps = ls_eps  # label smoothing
        # 初始化权重
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin # easy_margin 模式
        self.cos_m = math.cos(m) # cos margin
        self.sin_m = math.sin(m) # sin margin
        self.threshold = math.cos(math.pi - m) # cos(pi - m) = -cos(m)
        self.mm = math.sin(math.pi - m) * m # sin(pi - m)*m = sin(m)*m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)) # 获得cosθ (vector)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)) # 获得cosθ
        phi = cosine * self.cos_m - sine * self.sin_m # cosθ*cosm – sinθ*sinm = cos(θ + m)
        phi = phi.float() # phi to float
        cosine = cosine.float() # cosine to float
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # 以下代码控制了 θ+m 应该在 range[0, pi]
            # if cos(θ) > cos(pi - m) means θ + m < math.pi, so phi = cos(θ + m);
            # else means θ + m >= math.pi, we use Talyer extension to approximate the cos(θ + m).
            # if fact, cos(θ + m) = cos(θ) - m * sin(θ) >= cos(θ) - m * sin(math.pi - m)
            phi = torch.where(cosine > self.threshold, phi, cosine - self.mm) # https://github.com/ronghuaiyang/arcface-pytorch/issues/48
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        # 对label形式进行onehot转换，假设batch为2、有3类的话，即将label从[1,2]转换成[[0,1,0],[0,0,1]]
        one_hot = torch.zeros(cosine.size(), device=CONFIG['device'])
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # label smoothing
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) #验证是否匹配正确 
        # 进行re-scale
        output *= self.s

        return output

```
# Summary
1. Use Yolov5 to split fullbody data and backfins data;
2. Use tf_efficientnet_b0_ns + ArcFace as Baseline, train fullbody 512size, use kNN to search;
3. Add new_individual post-processing;
4. Training backfins 512size;
5. Use the fullbody 768size image, and adjust the data enhancement;
6. Fusion of fullbody 768size model and backfins 768size model;
7. Training tf_efficientnet_b6_ns , and fine-tuning all the above features;
8. Training tf_efficientnetv2_l_in21k, and fine-tuning all the above functions;
9. Training eca_nfnet_l2, and fine-tuning all the above functions;
10. Combine the 5Fold of the above three models, select the one with high cv, and fuse it;
11. KNN search parameters;
