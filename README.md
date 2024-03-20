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
> ├── dataset
>   ├── dataset_preprocess
>     ├── ispy_preprocess.py # The preprocess code of I-SPY1 dataset.
>   ├── ispy_dataset.py # The torch.Dataset of I-SPY1 dataset.
> └── 
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

<img src="./images/UML_Framework.jpg" alt="UML_Framework " style="zoom:70%;" />



## Dataset Acquisition

We use **3 datasets** to test our UML network. You can download the raw dataset from the following links, we also write the Pre-Process code. Please read them carefully and follow the guidelines !

- **I-SPY1 Trail Dataset**. Could be downloaded from [**HERE**](https://www.kaggle.com/datasets/saarthakkapse/ispy1-trail-dataset) ! The Pre-Process code is in `ispy_preprocess.py`.
- **Refuge Glaucoma**. Could be downloaded from [**HERE**](https://pan.baidu.com/s/1DE8a3UgwGJY85bsr4U7tdw?pwd=2023) !
- **ISIC Challenge 2017**. Could be downloaded from [**HERE**](https://challenge.isic-archive.com/data/#2017) ! The Pre-Porcess code is updating 🚀 !



