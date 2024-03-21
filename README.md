# UML: Uncertainty-Informed Mutual Learning for Joint Medical Image Classification and Segmentation

> This repository provides the code for our accepted **MICCAI2023 paper** "Uncertainty-Informed Mutual Learning for
> Joint Medical Image Classification and Segmentation". 
> 
> Official implementation: [**UML**](https://link.springer.com/chapter/10.1007/978-3-031-43901-8_4). 
> The structure of this repository is as follows:
>
> ```python
> UML/
> ├── images # All images used in this repository.
>     ├── UML_Framework.jpg # The Framework image.
> ├── datasets
>     ├── datasets_preprocess
>         ├── ispy_preprocess.py # The preprocess code of I-SPY1 dataset.
>         ├── refuge_preprocess.py # The preprocess code of REFUGE Glaucoma dataset.
>         └── isic_preprocess.py # The preprocess code of ISIC dataset. Updating 🚀.
>     ├── ispy_dataset.py # The torch.Dataset of I-SPY1 dataset.
>     ├── refuge_dataset.py # The torch.Dataset of Refuge dataset.
>     └── isic_dataset.py # The torch.Dataset of Refuge dataset. Updating 🚀.
> ├── models
> └── config.py # The Config file, set all hyperparameters here.
> ```



## Introduction

Classification and segmentation are crucial in medical image analysis as they enable accurate diagnosis and disease
monitoring. However, current methods often prioritize the mutual learning features and shared model parameters, while
neglecting the reliability of features and performances. In this paper, we propose a novel Uncertainty-informed Mutual
Learning (UML) framework for reliable and interpretable medical image analysis. Our UML introduces reliability to joint
classification and segmentation tasks, leveraging mutual learning with uncertainty to improve performance. To achieve
this, we first use evidential deep learning to provide image-level and pixel-wise confidences. Then, an uncertainty
navigator is constructed for better using mutual features and generating segmentation results. Besides, an uncertainty
instructor is proposed to screen reliable masks for classification. Overall, UML could produce confidence estimation in
features and performance for each link (classification and segmentation). The experiments on the public datasets
demonstrate that our UML outperforms existing methods in terms of both accuracy and robustness. Our UML has the
potential to explore the development of more reliable and explainable medical image analysis models.

<img src="./images/UML_Framework.jpg" alt="UML_Framework " style="zoom:60%;" />



## Dataset Acquisition

We use **3 Datasets** to test our UML network. You can **DOWNLOAD** the raw dataset from the following links. 

- **I-SPY1 Trail Dataset**. Could be downloaded from [**HERE**](https://www.kaggle.com/datasets/saarthakkapse/ispy1-trail-dataset) ! 
- **Refuge Glaucoma**. Could be downloaded from [**HERE**](https://pan.baidu.com/s/1DE8a3UgwGJY85bsr4U7tdw?pwd=2023) ! 
- **ISIC Challenge 2017**. Could be downloaded from [**HERE**](https://challenge.isic-archive.com/data/#2017) ! 



## Data Pre-Process and `torch.Dataset`

After downloading the datasets following the **Dataset Acquisition**, data preprocessing is needed which is to reformat the directory structure  of datasets. We have released Pre-Process code for datasets, please read them carefully and follow the guidelines in the comment ! Also we released `torch.Dataset` code for datasets,

- **I-SPY1 Trail Dataset**. 
  - The Pre-Process code is in `ispy_preprocess.py`, [**HERE**](https://github.com/KarryRen/UML/blob/main/dataset/dataset_preprocess/ispy_preprocess.py) !
  - The  `torch.Dataset` code is in `ispy_dataset.py`, [**HERE**](https://github.com/KarryRen/UML/blob/main/dataset/ispy_dataset.py) !
- **Refuge Glaucoma**. 
  - The Pre-Process code is in `refuge_preprocess.py`, [**HERE**](https://github.com/KarryRen/UML/blob/main/dataset/dataset_preprocess/refuge_preprocess.py) !
  - The  `torch.Dataset` code is in `refuge_dataset.py`, [**HERE**](https://github.com/KarryRen/UML/blob/main/dataset/refuge_dataset.py) !
- **ISIC Challenge 2017**. 
  - The Pre-Porcess code is updating 🚀 !
  - The  `torch.Dataset` code is updating 🚀 !





