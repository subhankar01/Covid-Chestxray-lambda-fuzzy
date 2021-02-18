# Covid-Chestxray-lambda-fuzzy

[![Watch a video](https://github.com/subhankar01/Covid-Chestxray-lambda-fuzzy/blob/main/assets/covid_poster.mp4)](https://github.com/subhankar01/Covid-Chestxray-lambda-fuzzy/blob/main/assets/covid_poster.mp4)


Our solution for [ICIAR 2018 Grand Challenge dataset on BreAst Cancer Histology images](https://iciar2018-challenge.grand-challenge.org/)

In the present work, we have proposed an approach for breast cancer image classification,implemented using Tensorflow and Keras, which at first uses five fine-tuned, pre-trained deep learning models for classification breast cancer histology im-ages. Then a fuzzy ensemble approach is introduced where the confidencescores of the five models are fused using Choquet integral, Coalition game theory and Information theory. The dataset used for evaluating the proposed model is the ICIAR 2018 Grand Challenge on Breast Cancer Histology (popularly known as BACH) images.  We have considered both 2-class (Malignant and Benign) and 4-class (Benign, In-situ carcinoma,Invasive carcinoma, and Normal tissue). To the best of our knowledge,our experimental results outperform many state-of-the-art methods.


## Table of Contents

- [Team Members](#1)
- [Reference Paper](#2)
- [Method Overview](#3)
- [Dataset](#4)
- [Results](#5)
- [Dependencies](#6)
- [Contact](#7)



## Team Members<a name="1"></a>
- Subhankar Sen  [LinkedIn](https://www.linkedin.com/in/subhankar-sen-a62457190lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BP2gUaNhAT0uL2etYJDiGqw%3D%3D) 
- Pratik Bhowal  [LinkedIn](https://www.linkedin.com/in/pratik-bhowal-1066aa198?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3B%2BqgwqwxJRIep5K454MTQ6w%3D%3D)
- Prof. Juan D. Velasquez Silva,University of Chile [LinkedIn](https://www.linkedin.com/in/jdvsilva/) ,[Google Scholar](https://scholar.google.com/citations?user=2xSehG8AAAAJ&hl=es)
- Associate Prof. Ram Sarkar,Jadavpur University,Kolkata [LinkedIn](https://www.linkedin.com/in/ram-sarkar-0ba8a758?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BvwKX%2Frm5RNSySsSaIQTiVQ%3D%3D)    , [Google Scholar](https://scholar.google.com/citations?hl=en&user=bDj0BUEAAAAJ&view_op=list_works&citft=1&citft=2&citft=3&email_for_op=subhankarsen2001%40gmail.com&gmla=AJsN-F5CKj5MB0jIcLJssFUKVVcxdf5jt8CBMbzSZf6W9RJvYUYp61X3OC6sXa_lzg1FHW7A8BpuLWwkMtDLWxJje2eowsNWqllMazckf90f5PsxhFZ2D1PcmhyhjJ8OT5q2-3Pc3DcwNuIj4E0s2LfWgQVOZBVVGs76xTjTPWNSKVvqBhvA-u05tkPXamKiItj8RSd_vApWN6jtmvYA9JcJ4ObPprLRFPV10T5a0A4nmrQVxyniapy6XIgng1L8D1qTtb2oFAow)


## Reference Paper<a name="2"></a>
If you find this work useful for your publications, please consider citing:
## Method Overview<a name="3"></a>

In this paper, our objective is to perform whole image wise classification of histology breast cancer images, unlike some previous works in this field which employed the technique of patch extraction for the classification purpose. [Fig 1](#8) shows the DCNN feature extraction and [Fig 2](#9) illustrates our proposed method. First, the H\&E stained breast cancer histology images are resized to 512x512 pixels using bicubic interpolation. The microscopy images are further pre-processed using stain normalization technique proposed . Image augmentation procedures are applied in order to counter the problem of limited training data and enhance the classification performance of the DCNN models.

Fine-tuned, well established deep CNN architectures, containing millions of parameters and pre-trained over the popular ImageNet dataset (a large research training dataset consisting of 1.4M images and 1000 classes) such as VGG16,VGG19, Xception, InceptionV3 and InceptionResnetV2 are used. Here, sparse and high-level bottleneck features of low dimensionality are extracted from each augmented image using the proposed DCNN models. This dimensionality reduction step reduces the likeliness of  overfitting. These features are then fed as input to an MLP classifier which performs the 2-class and 4-class classification of the breast cancer histology images. The confidence scores obtained per image, across the 5 DCNN models used, are then combined into a confidence matrix. This confidence matrix is then sent as input to a fuzzy ensemble of the five DCNN models, using Choquet integral and Coalition game theory, which leverages the classification accuracy of our proposed approach.

By employing ensemble technique, the overall classification performance of our proposed method gets further refined, outperforming the existing research studies found in the literature,thereby becoming one of the most novel approaches in the field of breast cancer histology classification.

## Fig 1:<a name="8"></a>

<img src="https://github.com/subhankar01/Breast-Cancer-Histology-Classification-using-deep-learning-and-Fuzzy-Ensembling/blob/main/assets/VGG19.png" width="750">

## Fig 2:Flowchart of the proposed method<a name="9"></a>

<img src="https://github.com/subhankar01/Breast-Cancer-Histology-Classification-using-deep-learning-and-Fuzzy-Ensembling/blob/main/assets/Method%20Flowchart.png" width="500">

## Dataset<a name="4"></a>

The dataset used for evaluating the proposed model is the ICIAR 2018 Grand Challenge on BreAst Cancer Histology (popularly known as BACH) images.The ICIAR 2018 Grand Challenge BACH dataset is publicly available underthe CC BY-NC-ND license and can be publicly accessed at [its official website](https://iciar2018-challenge.grand-challenge.org/). The datasetconsists of 400 annotated H&E stained histological breast tissue images of veryhigh resolution (2048x1536 pixels). Each image belongs to one of the four classesnamely: i) Benign tissue, ii) In-situ carcinoma, iii) Invasive carcinoma, and iv)Normal tissue, per class containing 100 images each.Each image is in RGB color mode and is stored in Tagged Image File Format (TIFF) with a magnification factor of 200 and pixel scale of 0.42μm x 0.42μm. The dataset is divided into training and test sets consisting of 300 and 100(25 randomly selected images taken from each category) breast cancer imagesrespectively.[Table 1](#10) shows the class distribution of the dataset before and afterimage augmentation.

<img src="https://github.com/subhankar01/Breast-Cancer-Histology-Classification-using-deep-learning-and-Fuzzy-Ensembling/blob/main/assets/images.PNG" width="750">
Examples of microscopic biopsy images in the dataset: (A) normal; (B) benign; (C) in situ carcinoma; and (D) invasive carcinoma

### Table 1: Dataset Overview<a name="10"></a>
<img src="https://github.com/subhankar01/Breast-Cancer-Histology-Classification-using-deep-learning-and-Fuzzy-Ensembling/blob/main/assets/dataset.png" width="500">


## Results<a name="5"></a>

Our experiment is implemented in Python using Keras package with Tensorflow as the deep learning framework backend and run on Google Colaboratoryhaving the following system specifications: Nvidia Tesla T4 with 13 GB GPUmemory,1.59GHz GPU Memory Clock and 12.72 GB RAM.In our method, we have first trained the five classification models and recordedtheir validation and test accuracies. The validation accuracies have been usedfor determining the weights as mentioned before.Our method has been used for both the 2-class and the 4-class classificationproblems of the breast cancer histology images.  [Table 2](#11) records the 2-classvalidation and test accuracies for each classifier, and the 2-class test accuracy ofthe ensemble method. [Table 3](#12) records the 4-class test and validation accuracies of each classifier, and the 4-class test accuracy of the ensemble method.


### Table 2: Results of 2-class classification<a name="11"></a>
| Classifier/Ensemble | Validation Accuracy |Test Accuracy |
| ------------- | ------------- | ------------- | 
| VGG16  | 100  |89|
| VGG19  | 99.8  |94|
| Xception| 100 | 95|
| Inception V3|100|94|
| InceptionResnetV2| 99.7| 93|
|**Ensemble**|**-**|**96**|


### Table 3: Results of 4-class classification<a name="12"></a>
| Classifier/Ensemble | Validation Accuracy |Test Accuracy |
| ------------- | ------------- | ------------- | 
| VGG16  | 97  |86|
| VGG19  | 98  |83|
| Xception| 99 | 91|
| Inception V3|99|90|
| InceptionResnetV2| 99| 91|
|**Ensemble**|**-**|**95**|

## Dependencies<a name="6"></a>
- [Python3](https://www.python.org/)
- [Tensorflow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/index.html#)

## Contact<a name="7"></a>

In case of doubt or further collaboration, feel free to email us ! 😊
- [Subhankar Sen (subhankarsen2001@gmail.com) ](mailto:subhankarsen2001@gmail.com)
- [Prof. Ram Sarkar (ramjucse@gmail.com)](mailto:ramjucse@gmail.com)
- [Pratik Bhowal (pratikbhowal1999@gmail.com)](mailto:pratikbhowal1999@gmail.com)
- [Juan D. Velasquez Silva (jvelasqu@dii.uchile.cl)](mailto:jvelasqu@dii.uchile.cl)


