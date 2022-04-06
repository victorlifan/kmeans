# Project Title
Kmeans from A to Z

## by Fan Li

## Table of Contents
1. [Description](#description)
2. [Workflow](#Workflow)
	1. [Kmeans](#km)
	2. [Kmeans ++](#km+)
	3. [Applications](#app)
	4. [Advanced topic: RF + Kmeans](#rf+km)
	5. [Limitations](#mf)
3. [References](#ref)


<a name="description"></a>
## Description
For this project, I implemented the Kmeans as well as the kmeans++ algorithm from scratch. I used five data sets to showcase some applications and results of those algorithms. Further, after uncovering the drawbacks of Kmeans, I implemented a ‘Spectral clustering’ using the random forest (RF) technique paired with kmeans++ to overcome the ‘discontinuity of clusters’ issue. Lastly, I addressed some limitations and possible improvements for future research reference.


<a name="Workflow"></a>
## Workflow:
<a name="km"></a>
##### 1. Kmeans
> Procedure:

1. Initialize centroids꞉
Randomly initialize k number of data points from the original X data. The number of k depends on how many clusters we want to end up with.

2. Compute distance꞉
Here I used Euclidean distance to measure the distance from each of the remaining data points to each of the centroids we initialized in step 1, assigning each of the remaining data points to the ‘closest ’ centroids.

3. Update centroids꞉
Within each cluster, compute the average distance of all the data points to that centroid FEATURE WISE. This average distance will be the new centroids’ ‘coordinate’ in that cluster. Intuitively speaking, this means we are correcting the centroids to be the ‘center’ of that cluster. This means our final centroids will most likely not be members of the dataset. The reason we picked data points from the dataset as initial centroids is simply to assign a starting point.

4. Reassign data point
Finally, compute distance, reassign data points according to the new centroids we updated in step 3, update centroids. Iterate the above process until the centroids’ ‘coordinates’ don’t change any more.

ISSUES:
> Each rounds’ initial centroids are different due to the mechanism of random initialization. This goes without saying, but the algorithm produces slightly different final centroids, labels, and clusters. The good news is all the final norms are 0, which means those clusters make perfect sense in each of their own round’s ‘world’.


<a name="km+"></a>
##### 2. Kmeans ++

> Procedure:

1. Initialize centroids꞉
Randomly initialize k number of data points from the original X data. The number of k depends on how many clusters we want to end up with.
2. Compute distance꞉
Here I used Euclidean distance to measure the distance from each of the remaining data points to each of the centroids we initialized in step 1, assigning each of the remaining data points to the ‘closest ’ centroids.
3. Update centroids꞉
Within each cluster, compute the average distance of all the data points to that centroid FEATURE WISE. This average distance will be the new centroids’ ‘coordinate’ in that cluster. Intuitively speaking, this means we are correcting the centroids to be the ‘center’ of that cluster. This means our final centroids will most likely not be members of the dataset. The reason we picked data points from the dataset as initial centroids is simply to assign a starting point.

4. Reassign data point
Finally, compute distance, reassign data points according to the new centroids we updated in step 3, update centroids. Iterate the above process until the centroids’ ‘coordinates’ don’t change any more.

<a name="app"></a>
##### 3. Applications
> Synthetic data set

![alt test](https://raw.githubusercontent.com/victorlifan/kmeans/main/img/1.png)

> numerical features
* Num of distinct artist
* Total length
* Num of songs played
* Num of 404 status received
* Customer lifetime
* page count

> vectors assemble and feature normalization
* 1 userId column
* 1 label column
* 83 feature columns

<a name="mol"></a>
##### 4. Modeling
Split the full dataset into train, test, and validation sets. Test out several of the machine learning methods. Evaluate the accuracy of the various models, tuning parameters as necessary. Determine the winning model based on test accuracy and report results on the validation set.

> Initial model selection
* Random Forest
* GBT
* LinearSVC

>Hyperparameter tune (GBT)
* maxDepth
* maxIter

>Feature importance

<a name='drm'></a>
##### 5. Dimension reduction
The original dataframe `final_df` has 83 features, we can see from the  `Weights for top 10 most predictive features` plot that 10 features already carry well over 60% of weights. Futhermore, `fp` dataframe shows 30 features carry over 98% of weights.

Reducing dimension to top 30 features will be cheaper in terms of time consume and computing power while model still remains promising output.

<img src="ima/fimp.png" alt="Dot Product" width="800">


<a name="Dataset"></a>
## Dataset

* `mini_sparkify_event_data.zip`: usage information details. Size: 10.2MB (due to file size limitation, zip file is uploaded)
* `medium-sparkify-event-data.zip`: usage information details. For works performed on IBM Watson Studio. Size: 19.6MB (due to file size limitation, zip file is uploaded)
* Full 12G data set is not included (For works performed on AWS)

<a name="summary"></a>
## Summary of Project

1. Since the churned users are a fairly small subset, the data is imbalanced, using F1 score as the metric is the fair call.

A quick summary of our initial models:

* Random Forest F1 score 0.6410174880763115.
* GBT F1 score 0.7456568331672422.
* Linear SVC F1 score 0.625548726953468.

2. Because GBT outperformed the other two models, I choose it as the base model. After tuning parameters the best model reached over 0.74 as f1 score. Parameters specification:
* maxDepth: 5
* maxIter: 10

3. The feature reduced model takes less time to train and predict, it decreased time by 12.4% versus the full feature model. Surprisingly, the feature reduced model also increased f1 score by a small amount of 3.13%. This is a classic example of bias-variance tradeoff. More features yield a better performance on the training set, as it generalizes worse on the test set. In other words the full feature model might suffer from overfitting.

<img src="ima/freduce.png" alt="Dot Product" height="200" width="500">

<a name="About"></a>
## Files In The Repository
+ [`Jupyter Notebook file`](https://github.com/victorlifan/Sparkify--Pyspark-Big-Data-Project/blob/master/Sparkify.ipynb): workspace where I performed the works.
+ [`data_zip`](https://github.com/victorlifan/Sparkify--Pyspark-Big-Data-Project/tree/master/data_zip): a folder contains dataset zip files
+ [`ima`](https://github.com/victorlifan/Sparkify--Pyspark-Big-Data-Project/tree/master/ima): png files were displayed in READMEs

<a name="Software"></a>
## Software used
+ Jupyter Notebook
+ Atom
+ Python 3.7
> + Numpy
> + Pandas
> + Matplotlib
> * Seaborn
+ Spark 3.0
>* Pyspark SQL
>* Pyspark ML



## Credits
* [How to Install and Run PySpark in Jupyter Notebook on Windows](https://changhsinlee.com/install-pyspark-windows-jupyter/#comment-4302741820)
* [Feature Selection Using Feature Importance Score - Creating a PySpark Estimator](https://www.timlrx.com/2018/06/19/feature-selection-using-feature-importance-score-creating-a-pyspark-estimator/)
+ Data provided by: [DATA SCIENTIST NANODEGREE PROGRAM](https://www.udacity.com/course/data-scientist-nanodegree--nd025)
+ Instruction and assist: [DATA SCIENTIST NANODEGREE PROGRAM](https://www.udacity.com/course/data-scientist-nanodegree--nd025)
