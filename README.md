# Covid-Chestxray-lambda-fuzzy
<img src="https://github.com/subhankar01/Covid-Chestxray-lambda-fuzzy/blob/main/assets/covid_poster.gif" width="1000" height="500" />

Our solution for [Novel COVID-19 Chestxray Database](https://www.kaggle.com/subhankarsen/novel-covid19-chestxray-database)

In this project, we have applied Choquet integral for ensemble of deep CNN models and propose a novel method for the evaluation of fuzzy measures using Coalition Game Theory, Information Theory and Lambda fuzzy approximation. Three different sets of Fuzzy Measures are calculated using three different weighting schemes along with information theory and coalition game theory. Using these three sets of fuzzy measures three Choquet Integrals are calculated and their decisions are finally combined.To the best of our knowledge,our experimental results outperform many state-of-the-art methods.


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
- Associate Prof. Ram Sarkar,Jadavpur University,Kolkata [LinkedIn](https://www.linkedin.com/in/ram-sarkar-0ba8a758?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BvwKX%2Frm5RNSySsSaIQTiVQ%3D%3D)    , [Google Scholar](https://scholar.google.com/citations?hl=en&user=bDj0BUEAAAAJ&view_op=list_works&citft=1&citft=2&citft=3&email_for_op=subhankarsen2001%40gmail.com&gmla=AJsN-F5CKj5MB0jIcLJssFUKVVcxdf5jt8CBMbzSZf6W9RJvYUYp61X3OC6sXa_lzg1FHW7A8BpuLWwkMtDLWxJje2eowsNWqllMazckf90f5PsxhFZ2D1PcmhyhjJ8OT5q2-3Pc3DcwNuIj4E0s2LfWgQVOZBVVGs76xTjTPWNSKVvqBhvA-u05tkPXamKiItj8RSd_vApWN6jtmvYA9JcJ4ObPprLRFPV10T5a0A4nmrQVxyniapy6XIgng1L8D1qTtb2oFAow)


## Reference Paper<a name="2"></a>
If you find this work useful for your publications, please consider citing:
## Method Overview<a name="3"></a>
In the present work, we have proposed a lambda fuzzy based ensemble model of DCNN architectures for screening of COVID-19 from CXR images. At first, the CXR images have been preprocessed. Then fine-tuned, well-established DCNN architectures, pretrained over the [ImageNet dataset](http://www.image-net.org/) namely VGG16, Xception and InceptionV3 have been used for feature extraction.These image descriptors are then fed as input into an Multi-layer Perceptron (MLP) classifier with softmax output for 3-class classification problem (COVID-19, Pneumonia and Normal). The confidence scores obtained per image, across the three DCNN models used, are then combined using Choquet integral into a confidence matrix. The fuzzy measures required for the evaluation of the Choquet integral and the Choquet integral itself are calculated as follows.We calculate the Shapley values, using coalition game theory and information theory, which become the fuzzy measures of the single classifier set. We introduce three different weighting schemes to calculate the Shapley values better. We then use lambda fuzzy to calculate the fuzzy measures of the other subsets of classifiers whose cardinality is greater than 1, and then use Choquet integral for aggregation.  Three aggregations done with respect to the three weighting schemes are combined at the end. 

## Fig 1:<a name="8"></a>

<img src="https://github.com/subhankar01/Breast-Cancer-Histology-Classification-using-deep-learning-and-Fuzzy-Ensembling/blob/main/assets/VGG19.png" width="750">

## Fig 2:Flowchart of the proposed method<a name="9"></a>

<img src="https://github.com/subhankar01/Breast-Cancer-Histology-Classification-using-deep-learning-and-Fuzzy-Ensembling/blob/main/assets/Method%20Flowchart.png" width="500">

## Dataset<a name="4"></a>
We have used the [Novel COVID-19 Chestxray Database](https://github.com/subhankar01/Novel-COVID-19-Chestxray-Database) for evaluation of our proposed methodology. We have also used our code to show our method performance over the popular [COVIDx datasset](https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md). Information  about  the  Novel  COVID-19  Chestxray  Database  and  its  parent  image  repositories  isprovided  in [Table 1](#tab1)

### Table 1: Dataset Description<a name="tab1"></a>
| Dataset| COVID-19 |Pneumonia | Normal |
| ------------- | ------------- | ------------- | 
| [COVID Chestxray set](https://github.com/ieee8023/covid-chestxray-dataset) | 521 |239|218|
| [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) | 219 |1341|1345|
| [Actualmed COVID chestxray dataset](https://github.com/agchung/Actualmed-COVID-chestxray-dataset)| 12 |0|80|
| **Total**|**752**|**584**|**1693**|


## Results<a name="5"></a>

Our experiment is implemented in Python using Keras package with Tensorflow as the deep learning framework backend and run on Google Colaboratoryhaving the following system specifications: Nvidia Tesla T4 with 13 GB GPUmemory,1.59GHz GPU Memory Clock and 12.72 GB RAM.In our method, we have first trained the five classification models and recordedtheir validation and test accuracies. The validation accuracies have been usedfor determining the weights as mentioned before.Our method has been used for both the 2-class and the 4-class classificationproblems of the breast cancer histology images.  [Table 2](#11) records the 2-classvalidation and test accuracies for each classifier, and the 2-class test accuracy ofthe ensemble method. [Table 3](#12) records the 4-class test and validation accuracies of each classifier, and the 4-class test accuracy of the ensemble method.






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

## Contact<a name="7"></a>

In case of doubt or further collaboration, feel free to email us ! 😊
- [Subhankar Sen (subhankarsen2001@gmail.com) ](mailto:subhankarsen2001@gmail.com)
- [Prof. Ram Sarkar (ramjucse@gmail.com)](mailto:ramjucse@gmail.com)
- [Pratik Bhowal (pratikbhowal1999@gmail.com)](mailto:pratikbhowal1999@gmail.com)


