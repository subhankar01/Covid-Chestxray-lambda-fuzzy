# Covid-Chestxray-lambda-fuzzy
<img src="https://github.com/subhankar01/Covid-Chestxray-lambda-fuzzy/blob/main/assets/covid_poster.gif" width="1000" height="500" />

Our solution for [Novel COVID-19 Chestxray Repository](https://www.kaggle.com/subhankarsen/novel-covid19-chestxray-repository)

In this project, we have applied Choquet integral for ensemble of deep CNN models and propose a novel method for the evaluation of fuzzy measures using Coalition Game Theory, Information Theory and Lambda fuzzy approximation. Three different sets of Fuzzy Measures are calculated using three different weighting schemes along with information theory and coalition game theory. Using these three sets of fuzzy measures three Choquet Integrals are calculated and their decisions are finally combined.To the best of our knowledge,our experimental results outperform many state-of-the-art methods.


## Table of Contents

- [Team Members](#1)
- [Journal Paper](#2)
- [Installation](#3)
- [Dependencies](#4)
- [Method Overview](#5)
- [Dataset](#6)
- [Results](#7)
- [Contact](#8)



## Team Members<a name="1"></a>
- Subhankar Sen  [LinkedIn](https://www.linkedin.com/in/subhankar-sen-a62457190lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BP2gUaNhAT0uL2etYJDiGqw%3D%3D) 
- Pratik Bhowal  [LinkedIn](https://www.linkedin.com/in/pratik-bhowal-1066aa198?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3B%2BqgwqwxJRIep5K454MTQ6w%3D%3D),[Github](https://github.com/prat1999)
- Prof. Jin Hee Yoon, faculty of the Dept. of Mathematics and Statistics at Sejong University, Seoul, South Korea [LinkedIn](https://www.linkedin.com/in/jin-hee-yoon-2418a069), [Google Scholar](https://scholar.google.com/citations?user=Rq_TQc0AAAAJ&hl=en)
- Prof. Zong Woo Geem, faculty of College of IT Convergence at Gachon University, South Korea [LinkedIn](https://www.linkedin.com/in/zong-woo-geem-66273113), [Google Scholar](https://scholar.google.com/citations?hl=en&user=Je3-B2YAAAAJ)
- Prof. Ram Sarkar,  Professor at Dept. of Computer Science Engineering, Jadavpur Univeristy Kolkata, India [LinkedIn](https://www.linkedin.com/in/ram-sarkar-0ba8a758?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BvwKX%2Frm5RNSySsSaIQTiVQ%3D%3D)    , [Google Scholar](https://scholar.google.com/citations?hl=en&user=bDj0BUEAAAAJ&view_op=list_works&citft=1&citft=2&citft=3&email_for_op=subhankarsen2001%40gmail.com&gmla=AJsN-F5CKj5MB0jIcLJssFUKVVcxdf5jt8CBMbzSZf6W9RJvYUYp61X3OC6sXa_lzg1FHW7A8BpuLWwkMtDLWxJje2eowsNWqllMazckf90f5PsxhFZ2D1PcmhyhjJ8OT5q2-3Pc3DcwNuIj4E0s2LfWgQVOZBVVGs76xTjTPWNSKVvqBhvA-u05tkPXamKiItj8RSd_vApWN6jtmvYA9JcJ4ObPprLRFPV10T5a0A4nmrQVxyniapy6XIgng1L8D1qTtb2oFAow)


## Journal Paper<a name="2"></a>
If you find this work useful for your publications, please consider citing:
```
@ARTICLE{9534669,
  author={Bhowal, Pratik and Sen, Subhankar and Yoon, Jin Hee and Geem, Zong Woo and Sarkar, Ram},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Choquet Integral and Coalition Game-based Ensemble of Deep Learning Models for COVID-19 Screening from Chest X-ray Images}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/JBHI.2021.3111415}}
  ```
## Installation<a name="3"></a>
1. Make sure you have python3 setup on your system
2. Clone the repo
```
git clone https://github.com/subhankar01/Covid-Chestxray-lambda-fuzzy
```
3. Install requirements
```
pip install -r requirements.txt
```

## Dependencies<a name="4"></a>
Our project is built using Python 3.8.6 and the following packages 
```
numpy==1.19.5
pandas==1.1.5
matplotlib==3.2.2
seaborn==2.5.0
opencv-python==4.1.2
tensorflow==2.5.0
```
## Method Overview<a name="5"></a>
## Fig 1:Flowchart of the proposed method<a name="fig2"></a>

<img src="https://github.com/subhankar01/Covid-Chestxray-lambda-fuzzy/blob/main/assets/Covid-19%20flowchart.png" width="500">

## Dataset<a name="6"></a>
We have used the [Novel COVID-19 Chestxray Repository](https://www.kaggle.com/subhankarsen/novel-covid19-chestxray-repository) for evaluation of our proposed methodology. We have also used our code to show our method performance over the popular [COVIDx dataset](https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md). Information  about  the  Novel  COVID-19  Chestxray  Database  and  its  parent  image  repositories  is provided  in [Table 1](#tab1)

### Table 1: Dataset Description<a name="tab1"></a>

| Dataset| COVID-19 |Pneumonia | Normal |
| ------------- | ------------- | ------------- | -------------|
| [COVID Chestxray set](https://github.com/ieee8023/covid-chestxray-dataset) | 521 |239|218|
| [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) | 219 |1345|1341|
| [Actualmed COVID chestxray dataset](https://github.com/agchung/Actualmed-COVID-chestxray-dataset)| 12 |0|80|
| **Total**|**752**|**1584**|**1639**|


## Results<a name="7"></a>

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


##  Fig 2:ROC of the 3 DCNN models and proposed ensemble method<a name="fig3"></a>

<img src="https://github.com/subhankar01/Covid-Chestxray-lambda-fuzzy/blob/main/assets/roc.png" width="350">

##  Fig 3:Multi-labelled ROC curve of the proposed ensemble method<a name="fig4"></a>

<img src="https://github.com/subhankar01/Covid-Chestxray-lambda-fuzzy/blob/main/assets/multiclass_roc.png" width="350">

##  Fig 4:Confusion Matrix of the proposed method<a name="fig5"></a>

<img src="https://github.com/subhankar01/Covid-Chestxray-lambda-fuzzy/blob/main/assets/confmatrix.png" width="350">



## Contact<a name="8"></a>

In case of doubt or further collaboration, feel free to email us ! 😊
- [Subhankar Sen (subhankarsen2001@gmail.com) ](mailto:subhankarsen2001@gmail.com)
- [Pratik Bhowal (pratikbhowal1999@gmail.com)](mailto:pratikbhowal1999@gmail.com)


