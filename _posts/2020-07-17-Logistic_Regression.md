---
layout: post
title: Logistic Regression
subtitle: 
cover-img: /assets/img/bgimg3.jpg
thumbnail-img: 
share-img: 
tags: [Logistic regression, Logistic Function, Gradient Descent]
---


Suppose we want to predict the risk of developing a disease to a person or we want to predict whether a student will pass or fail in an exam by using data like how many hours he/she spends on daily study. Here comes the need of Logistic Regression. Then let’s get started !



## Introduction
Logistic Regression actually denotes a classification algorithm. It is a statistical model that is used to predict a class or category based on the other features. It finds a relationship between the features and the predicted class in which they can fit in.

Suppose you want to predict whether a person with a tumor will survive or not and the given features include the age, sex, size of the tumor, and other factors regarding the person. Logistic regression is used to predict the probability of survival by using the given features. 

![Crepe](/assets/img/3.1.png){: .center-block :}

Generally Logistic Regression can be classified into two categories :
1. Binary logistic regression
2. Multinomial logistic regression

Binary consists of two output classes whereas multinomial consists of more than two classes.



## But why not Linear Regression ?

The question arises that can we use linear regression to predict the classes using the given features.

Suppose we want to predict whether a tumor is malignant (“1”) or benign (“0”) based on the size. So let's fit a linear model on size vs class ( 0 or 1 ).

![Crepe](/assets/img/3.2.png){: .center-block :}

The threshold that can be set to classify the output can be 0.5.
If h<sub>Θ</sub>(x) >= 0.5 then predict y=1.
If h<sub>Θ</sub>(x) < 0.5 then predict y=0.

But there is a problem with this approach. If we have another tumour size that is surely malignant and then we fit the linear model including this size too then the line shifts towards this new included point and the boundary also shifts. This makes the point that was earlier classified as malignant to be predicted as benign and this is not at all good :(

![Crepe](/assets/img/3.3.png){: .center-block :}

There is also another problem. For classification, we need values of y i.e. the prediction to be between 0 and 1 as the probability cannot be less than 0 and greater than 1. But in linear regression the values less than zero and greater than 1 are also possible.



## Logistic Function
To deal with the problems stated in using linear regression we use a different function for logistic regression. With linear regression, we had h<sub>Θ</sub>(x) = Θ<sup>T</sup>x , but for the logistic regression we take h<sub>Θ</sub>(x) = g(Θ<sup>T</sup>x) where the 

![Crepe](/assets/img/3.4.png){: .center-block :}

g(z) is known as sigmoid or logistic function. By using this function we get g(z) → 0 as z → −∞, g(z) = 0.5 if z=0 and g(z) → 1 as z → ∞. And thus for Logistic Regression we get 0 < h<sub>Θ</sub>(x) < 1.

![Crepe](/assets/img/3.5.png){: .center-block :}



## Interpretation of h<sub>Θ</sub>(x)
In terms of conditional probability, h<sub>Θ</sub>(x) denotes P(y=1 | x) i.e. the probability of y being 1 given the features x.

Suppose we have x = [x<sub>0</sub> x<sub>1</sub>] = [1 S<sub>tumour</sub>] and h<sub>Θ</sub>(x)=0.7 then this means that there is 70% chance of tumour being malignant given the size S of the tumour. 

However there is an alternate notation for the h<sub>Θ</sub>(x). 

<p align="center">h<sub>Θ</sub>(x)=P(y=1 | x; Θ) </p>

This denotes the probability of y being 1 given the features x and parameterized by Θ. 

The possible values for y can only be 0 and 1. By law of probability we can write that 

<p align="center">P(y=1 | x; Θ) + P(y=0 | x; Θ) = 1</p>



## Decision Boundary
Decision boundary is a function of the regression parameters Θ, derived from training data.
Let us understand this by an example. Suppose we have some data points like given in the figure

![Crepe](/assets/img/3.6.png){: .center-block :}

After training the Logistic model on the dataset, we have Θ to be [-3 1 1]<sup>T</sup>. Now we know from the logistic function that h<sub>Θ</sub>(x) = 1 if Θ<sup>T</sup>x > 0 i.e.
 -3 + 1.x<sub>1</sub> +1.x<sub>2</sub> > 0 or x<sub>1</sub> +x<sub>2</sub> > 3.

![Crepe](/assets/img/3.7.png){: .center-block :}

Decision boundaries are not always clear cut. That is, the transition from one class in the feature space to another is not discontinuous, but gradual.

But suppose we have a dataset that looks like this

![Crepe](/assets/img/3.8.png){: .center-block :}

Here we cannot separate these by a linear decision boundary but it can be done by a non-linear decision boundary.

For that let 

![Crepe](/assets/img/3.9.png){: .center-block :}

This means we are transforming our 2D space to a 5D feature space.
Assigning Θ<sub>0</sub> = -1, Θ<sub>1</sub> = 0, Θ<sub>2</sub> = 0, Θ<sub>3</sub> = 1, Θ<sub>4</sub> = 1. Then, h<sub>Θ</sub>(x) = 1 if -1+ x<sub>1</sub><sup>2</sup> + x<sub>2</sub><sup>2</sup> ≥ 0 => x<sub>1</sub><sup>2</sup> + x<sub>2</sub><sup>2</sup> ≥ 1. 

![Crepe](/assets/img/3.10.png){: .center-block :}

However more complex decision boundaries are also possible depending on our chosen feature space.  



## Cost Function
We previously saw that we cannot use linear regression for classification problem but can we use the cost function of linear regression for our logistic function. So let’s see what happens if we use linear regression cost function 

![Crepe](/assets/img/3.11.png){: .center-block :}

What we get is this

![Crepe](/assets/img/3.12.png){: .center-block :}

We cannot use gradient descent on this non-convex function because it may lead to getting stuck in a local minimum.

The solution for this problem is to define a new cost function for Logistic Regression

![Crepe](/assets/img/3.13.png){: .center-block :}

We can see that if our y=1 but predicted y is close to zero then we get a very huge error so this cost function is good and we will use gradient descent to decrease this error.

![Crepe](/assets/img/3.14.png){: .center-block :}

Now rewriting the cost function we get

![Crepe](/assets/img/3.15.png){: .center-block :}

Now we can apply gradient descent on this cost function to get the best fit parameters.

![Crepe](/assets/img/3.16.png){: .center-block :}



But logistic regression is not just limited to data points. We can even classify images and audio based on the pixels and audio samples. Scikit-Learn is a python library which allows us to implement Logistic Regression with just 2-3 lines of code. 

So get started and thanks for reading :)
