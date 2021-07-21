# Covid-Chestxray-lambda-fuzzy
<img src="https://github.com/subhankar01/Covid-Chestxray-lambda-fuzzy/blob/main/assets/covid_poster.gif" width="1000" height="500" />

Our solution for [Novel COVID-19 Chestxray Repository](https://www.kaggle.com/subhankarsen/novel-covid19-chestxray-repository)

In this project, we have applied Choquet integral for ensemble of deep CNN models and propose a novel method for the evaluation of fuzzy measures using Coalition Game Theory, Information Theory and Lambda fuzzy approximation. Three different sets of Fuzzy Measures are calculated using three different weighting schemes along with information theory and coalition game theory. Using these three sets of fuzzy measures three Choquet Integrals are calculated and their decisions are finally combined.To the best of our knowledge,our experimental results outperform many state-of-the-art methods.


## Table of Contents

- [Team Members](#1)
- [Reference Paper](#2)
- [Method Overview](#3)
- [Dataset](#4)
- [Results](#5)
- [Contact](#6)



## Team Members<a name="1"></a>
- Subhankar Sen  [LinkedIn](https://www.linkedin.com/in/subhankar-sen-a62457190lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BP2gUaNhAT0uL2etYJDiGqw%3D%3D) 
- Pratik Bhowal  [LinkedIn](https://www.linkedin.com/in/pratik-bhowal-1066aa198?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3B%2BqgwqwxJRIep5K454MTQ6w%3D%3D),[Github](https://github.com/prat1999)
- Jin Hee Yoon [LinkedIn](https://www.linkedin.com/in/jin-hee-yoon-2418a069), [Google Scholar](https://scholar.google.com/citations?user=Rq_TQc0AAAAJ&hl=en)
-  Zong Woo Geem [LinkedIn](https://www.linkedin.com/in/zong-woo-geem-66273113), [Google Scholar](https://scholar.google.com/citations?hl=en&user=Je3-B2YAAAAJ)
- Prof. Ram Sarkar,Jadavpur University,Kolkata [LinkedIn](https://www.linkedin.com/in/ram-sarkar-0ba8a758?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BvwKX%2Frm5RNSySsSaIQTiVQ%3D%3D)    , [Google Scholar](https://scholar.google.com/citations?hl=en&user=bDj0BUEAAAAJ&view_op=list_works&citft=1&citft=2&citft=3&email_for_op=subhankarsen2001%40gmail.com&gmla=AJsN-F5CKj5MB0jIcLJssFUKVVcxdf5jt8CBMbzSZf6W9RJvYUYp61X3OC6sXa_lzg1FHW7A8BpuLWwkMtDLWxJje2eowsNWqllMazckf90f5PsxhFZ2D1PcmhyhjJ8OT5q2-3Pc3DcwNuIj4E0s2LfWgQVOZBVVGs76xTjTPWNSKVvqBhvA-u05tkPXamKiItj8RSd_vApWN6jtmvYA9JcJ4ObPprLRFPV10T5a0A4nmrQVxyniapy6XIgng1L8D1qTtb2oFAow)


## Journal Paper<a name="2"></a>
If you find this work useful for your publications, please consider citing:
```
```
## Installation
1. Make sure you have python3 setup on your system
2. Clone the repo
```
git clone https://github.com/subhankar01/Covid-Chestxray-lambda-fuzzy
```
3. Install requirements
```
pip install -r requirements.txt
```

## Method Overview<a name="3"></a>
In the present work, we have proposed a lambda fuzzy based ensemble model of DCNN architectures for screening of COVID-19 from CXR images. At first, the CXR images have been preprocessed. Then fine-tuned, well-established DCNN architectures, pretrained over the [ImageNet dataset](http://www.image-net.org/) namely VGG16, Xception and InceptionV3 have been used for feature extraction.In [Fig. 1](#fig1) VGG16  has been  used  for  extraction  of  discriminating  features  from  theinput  CXR  images.These image descriptors are then fed as input into an Multi-layer Perceptron (MLP) classifier with softmax output for 3-class classification problem (COVID-19, Pneumonia and Normal). The confidence scores obtained per image, across the three DCNN models used, are then combined using Choquet integral into a confidence matrix. The fuzzy measures required for the evaluation of the Choquet integral and the Choquet integral itself are calculated as follows.We calculate the Shapley values, using coalition game theory and information theory, which become the fuzzy measures of the single classifier set. We introduce three different weighting schemes to calculate the Shapley values better. We then use lambda fuzzy to calculate the fuzzy measures of the other subsets of classifiers whose cardinality is greater than 1, and then use Choquet integral for aggregation.  Three aggregations done with respect to the three weighting schemes are combined at the end.[Fig. 2](#fig2) demonstrates the flowchart of our proposed methodology.

## Fig 1:<a name="fig1"></a>

<img src="https://github.com/subhankar01/Covid-Chestxray-lambda-fuzzy/blob/main/assets/VGG16_extraction.png" width="750">

## Fig 2:Flowchart of the proposed method<a name="fig2"></a>

<img src="https://github.com/subhankar01/Covid-Chestxray-lambda-fuzzy/blob/main/assets/Covid-19%20flowchart.png" width="500">

## Dataset<a name="4"></a>
We have used the [Novel COVID-19 Chestxray Repository](https://www.kaggle.com/subhankarsen/novel-covid19-chestxray-repository) for evaluation of our proposed methodology. We have also used our code to show our method performance over the popular [COVIDx dataset](https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md). Information  about  the  Novel  COVID-19  Chestxray  Database  and  its  parent  image  repositories  is provided  in [Table 1](#tab1)

### Table 1: Dataset Description<a name="tab1"></a>

| Dataset| COVID-19 |Pneumonia | Normal |
| ------------- | ------------- | ------------- | -------------|
| [COVID Chestxray set](https://github.com/ieee8023/covid-chestxray-dataset) | 521 |239|218|
| [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) | 219 |1345|1341|
| [Actualmed COVID chestxray dataset](https://github.com/agchung/Actualmed-COVID-chestxray-dataset)| 12 |0|80|
| **Total**|**752**|**584**|**1639**|


## Results<a name="5"></a>
To implement the proposed method, we have considered Python using Keras package with Tensorflow used as the deep learning framework backend and run on Google Colaboratory having the following system specifications: Nvidia Tesla T4 with 13 GB GPU memory, 1.59GHz GPU Memory Clock and 12.72 GB RAM.We have performed 3-class classification of the CXR images which are COVID-19 affected lungs, Pneumonia affected lungs and Normal lungs. We have used three pretrained models, namely, VGG16, Xception and InceptionV3, and then ensembled the decision of the three models using Choquet Integral. The fuzzy measures are calculated using Coalition game theory and Lambda fuzzy approximation. The parameters used for training the deep learning models are as follows. Adam optimizer, with a learning rate of 0.001 and hyperparameters beta_1 and beta_2 set equal to 0.6 and 0.8 respectively, are used for training the MLP classifier using the extracted image descriptors. The learning rate and hyperparameter values are experimentally inferred to be the most optimal values obtained using Grid search technique for model tuning and optimization. The batch size is set to 32, and the models are trained for 1000 epochs. Weights are initialized from the weights obtained by training ImageNet dataset for all DCNNs.In [Table 2](#tab2), we have recorded the validation accuracy, test accuracy, precision and recall of each of the three models, and the final results obtained after applying the ensemble method.In [Fig. 3](#fig3), we plot ROC of the 3 DCNN models and proposed ensemble method..In[Fig. 4](#fig4) we plot the multi-labeled ROC curve for the proposed ensemble method. In [Fig. 5](#fig5) we plot the Confusion Matrix for the proposed ensemble method.



### Table 2: Results of 3-class classification<a name="tab2"></a>
| Classifier/Ensemble | Validation Accuracy(in %) |Test Accuracy(in %) |Precision(Avg)|Recall(Avg)|AUC|
| ------------- | ------------- | ------------- | -------------|-------------|-------------|
| VGG16  | 96.71 |91.22|0.92|0.92|0.92|0.92
| Xception|97.02|92.98|0.93|0.93|0.92
| InceptionV3|97.49|93.48|0.94|0.94|0.94|
|Choquet Integral (Weight 1)|97.74|94.23|0.94|0.94|-|
|Choquet Integral (Weight 2)|98.24|94.23|0.94|0.94|-|
|Choquet Integral (Weight 3)| 97.49|93.73|0.95 |0.95 |-|
|**Ensemble**|**98.99**|**95.49** |**0.96**|**0.96**|**0.97**|


##  Fig 3:ROC of the 3 DCNN models and proposed ensemble method<a name="fig3"></a>

<img src="https://github.com/subhankar01/Covid-Chestxray-lambda-fuzzy/blob/main/assets/roc.png" width="350">

##  Fig 4:Multi-labelled ROC curve of the proposed ensemble method<a name="fig4"></a>

<img src="https://github.com/subhankar01/Covid-Chestxray-lambda-fuzzy/blob/main/assets/multiclass_roc.png" width="350">

##  Fig 5:Confusion Matrix of the proposed method<a name="fig5"></a>

<img src="https://github.com/subhankar01/Covid-Chestxray-lambda-fuzzy/blob/main/assets/confmatrix.png" width="350">



## Contact<a name="6"></a>

In case of doubt or further collaboration, feel free to email us ! 😊
- [Subhankar Sen (subhankarsen2001@gmail.com) ](mailto:subhankarsen2001@gmail.com)
- [Pratik Bhowal (pratikbhowal1999@gmail.com)](mailto:pratikbhowal1999@gmail.com)
- [Prof. Ram Sarkar (ramjucse@gmail.com)](mailto:ramjucse@gmail.com)


