# Face-Recognition-Using-MATLAB

This repository discusses the new Face Recognition Method that is based on Fisher's Discriminant Analysis (FLDA) and Support Vector Machines (SVM). The FLDA projects the high dimensional image space into a relatively low-dimensional space to acquire most discriminant features among the different classes.

Recently, SVM has been used as a new technique for pattern classification and recognition. This repository uses SVM as a classifier, which classifies the images based on the extracted features.

## Introduction

Face Recognition is one of the important areas in the field of pattern recognition and artificial intelligence. It has been used in wide range of applications such as biometrics, law enforcement, identity authentication and surveillance. The image data are always high dimensional in the face recognition area, and it require considerable amount of computing time for recognition. That’s why the feature selection is very important for improving classifier’s accuracy and reducing the running time for classification.

### Using Fisher's Linear Discriminant Analysis

Principal component analysis (PCA) and Fisher’s linear discriminant analysis (FLDA) are two powerful methods used for data reduction as well as feature extraction in appearance-based face recognition approaches. The PCA method is based on linearly projecting the image space into a low dimensional feature space for dimensionality reduction. It yields projection directions that maximize the total scatter across all classes, i.e., across all images of all training faces. Thus, PCA retains unwanted variations due to lighting and facial expression. The FLDA technique projects the face images from high-dimensional image space to a relatively lowdimensional space linearly by maximizing the ratio of between-class scatter to that of within-class scatter. It is generally believed that, when it comes to solving problem of pattern classification, FLDA-based algorithms perform better than PCA-based ones, since the former optimizes the low-dimensional representation of the objects with focus on the most discriminant feature extraction while the latter achieves simply object reconstruction.

### Using Support Vector Machine

Support Vector Machine (SVM) is a popular classification tool, which is applied for pattern recognition as well as computer vision domains recently. A SVM is used to find the hyperplane that separates the largest fraction of points of the same class on the same side, while maximizing the distance from the either class to the hyperplane. This hyperplane is called Optimal Separating Hyperplane [10], which minimizes the risk of misclassification in the training as well as unknown test set. In this new method, to achieve higher performance in face recognition, I basically focused on two criteria. Firstly, the features are extracted from a facial image. This feature extracting technique gives the best discriminant features among classes rather
than data. Secondly, I choose a classifier, which basically trained and learned those face image and then finally classify test face images based on the extracted features. This classifier helps to achieve better generalization capability.
