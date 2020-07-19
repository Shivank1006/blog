---
layout: post
title: Polynomial Regression
subtitle: 
cover-img: /assets/img/bg2.jpg
thumbnail-img: 
share-img: /assets/img/bg2.jpg
tags: [polynomial regression, overfitting, bias-variance tradeoff]
---
Many times the data we have, cannot be fitted linearly and we need a higher degree polynomial ( like quadratic or cubic ) to fit the data. We can use a polynomial model when the relationship between outcome and explanatory variables is curvilinear as can be seen in the given figure. 

![Crepe](https://shivank1006.github.io/blog/assets/img/2.0.png){: .mx-auto.d-block :}

This article requires the understanding of linear regression and the mathematics behind it. If you are not aware about it then you can refer to my previous article on [Linear Regression](https://shivank1006.github.io/blog/2020-03-14-Linear_Regression/).
<br><br>
## Introduction

Polynomial regression is a form of regression analysis in which the relationship between the independent variable x and the dependent variable y is modelled as an nth degree polynomial in x. However there can be two or more independent variables or features also.

![Crepe](https://shivank1006.github.io/blog/assets/img/2.12.png){: .mx-auto.d-block :}

Although polynomial regression is technically a special case of multiple linear regression, the interpretation of a fitted polynomial regression model requires a somewhat different perspective. It is often difficult to interpret the individual coefficients in a polynomial regression fit, since the underlying monomials can be highly correlated. For example, x and x<sup>2</sup> have correlation around 0.97 when x is uniformly distributed on the interval (0, 1). 

<br>
## Linear Algebra Review 

### Rank of a matrix
Suppose we are given a mxn matrix <b>A</b> with its columns to be [ a<sub>1</sub>, a<sub>2</sub>, a<sub>3</sub> …...a<sub>n</sub> ]. The column a<sub>i</sub> is called linearly dependent if we can write it as a linear combination of other columns i.e.
a<sub>i</sub> = w<sub>1</sub>a<sub>1</sub> + w<sub>2</sub>a<sub>2</sub> + ……. + w<sub>i-1</sub>a<sub>i-1</sub> + w<sub>i+1</sub>a<sub>i+1</sub> +..... + w<sub>n</sub>a<sub>n</sub>
where at least one w<sub>i</sub> is non-zero.
Then we define the rank of the matrix as the number of independent columns in that matrix.
Rank(<b>A</b>) = number of independent columns in <b>A</b> 

However there is another interesting property that the number of linearly independent columns is equal to the number of independent rows in a matrix.([proof](https://en.wikibooks.org/wiki/Linear_Algebra/Row_and_column_spaces#Proof))

Hence Rank(<b>A</b>) ≤ min(m, n)
A matrix is called full rank if Rank(<b>A</b>) = min(m, n) and is called rank deficient if Rank(<b>A</b>) < min(m, n).

### Pseudo-Inverse of a matrix
A nxn square matrix <b>A</b> has an inverse <b>A<sup>-1</sup></b> if and only if <b>A</b> is a full rank matrix. However a rectangular mxn  matrix <b>A</b> does not have an inverse. 
If <b>A<sup>T</sup></b> denotes the transpose of matrix <b>A</b> then <b>A<sup>T</sup>A</b> is a square matrix and Rank of (<b>A<sup>T</sup>A</b>) = Rank (<b>A</b>) ([proof](https://math.stackexchange.com/questions/349738/prove-operatornamerankata-operatornameranka-for-any-a-in-m-m-times-n)).

Therefore if <b>A</b> is a full-rank matrix then the inverse of <b>A<sup>T</sup>A</b> exists. And <b>(A<sup>T</sup>A)<sup>-1</sup>A<sup>T</sup></b> is called the pseudo-inverse of <b>A</b>. We'll see soon why it is called so.

<br>
## Polynomial Fitting 
In general, non-linear fitting is more difficult than linear fitting. However we will be using a simple version.
In Polynomial regression, the original features are converted into Polynomial features of required degree (2,3,..,n) and then modeled using a linear model.
Suppose we are given n data points <b>p<sub>i</sub></b> = [ x<sub>i1</sub> ,x<sub>i2</sub> ,……, x<sub>im</sub> ]<sup>T</sup> , 1 ≤ i ≤ n , and their corresponding values <b>v<sub>i</sub></b> . Here m denotes the number of features that we are using in our polynomial model. Our goal is to find a nonlinear function f that minimizes the error 
![Crepe](https://shivank1006.github.io/blog/assets/img/2.2.png){: .mx-auto.d-block :}
Hence <b>f</b> is nonlinear over <b>p<sub>i</sub></b> .

So let's take an example of a quadratic function i.e. with n data points and 2 features. Then the function would be

![Crepe](https://shivank1006.github.io/blog/assets/img/2.3.png){: .mx-auto.d-block :}

The objective is to learn the coefficients. Hence we have n points <b>p<sub>i</sub></b> and their corresponding values <b>v<sub>i</sub></b> ; we have to minimize

![Crepe](https://shivank1006.github.io/blog/assets/img/2.4.png){: .mx-auto.d-block :}

For each data point we can write equations as 

![Crepe](https://shivank1006.github.io/blog/assets/img/2.5.png){: .mx-auto.d-block :}

Hence we can form the following matrix equation

<p align='center'><b>Da = v</b></p>

where


<img src='https://shivank1006.github.io/blog/assets/img/2.6.png' align='center'>



However the equation is nonlinear with respect to the data points <b>p<sub>i</sub></b> , it is linear with respect to the coefficients <b>a</b>. So, we can solve for <b>a</b> using the linear least square method.

We have

<p align='center'><b>Da = v</b></p>

multiply <b>D<sup>T</sup></b> on both sides

<p align='center'><b>D<sup>T</sup>Da = D<sup>T</sup>v</b></p>

Suppose <b>D</b> has a full rank, that is when the columns in <b>D</b> are linearly independent, then <b>D<sup>T</sup>D</b> has an inverse.Therefore

<p align='center'><b>(D<sup>T</sup>D)<sup>-1</sup>(D<sup>T</sup>D)a = (D<sup>T</sup>D)<sup>-1</sup>D<sup>T</sup>v</b></p>

We now have

<p align='center'><b>a = (D<sup>T</sup>D)<sup>-1</sup>D<sup>T</sup>v</b></p>

Comparing it with <b>Da = v</b>, we can see that <b>(D<sup>T</sup>D)<sup>-1</sup>D<sup>T</sup></b> acts like the inverse of <b>D</b>. So it is called the pseudo-inverse of <b>D</b>.

The above used quadratic polynomial function can be generalised to a polynomial function of order or degree m.
<p align='center'><img src='https://shivank1006.github.io/blog/assets/img/2.7.png' align='centre'></p>

## Underfitting and Overfitting

Underfitting and Overfitting are the most common problems in a machine learning model and lead to poor performance. 


Underfitting refers to a condition when a model does not learn the features of training data and also cannot perform good on unseen data. Like if we want to fit a line in data with curvilinear pattern then the model will definitely underfit and cannot be used.


<p align='center'><img src='https://shivank1006.github.io/blog/assets/img/2.11.png' align='centre'></p>

Overfitting refers to a condition when a model learns the data and the noise present in it so well, that it declines the performance of the model on the unseen data. Using a higher degree polynomial, say of degree 4 or 5, then it surely fit the training data very well but will produce high error for unseen data. Then the model will be overfit and again cannot be used.



## Bias-Variance Tradeoff

<p align='center'><img src='https://shivank1006.github.io/blog/assets/img/2.10.png' align='centre'></p>
Bias is how far are the predicted values from the actual values. If there is so much difference in the predicted values and the actual values then we say the bias is high.


Variance tells us how scattered are the predicted values from the actual values. High variance causes overfitting which means that our model is learning the noise present in the data.

<p align='center'><img src='https://shivank1006.github.io/blog/assets/img/2.8.png' align='centre'></p>
<p align='center'>Bias-Variance Tradeoff</p>

This figure depicts the bias-variance tradeoff in a very good way. It can be seen that as the complexity of the model increases i.e. if we use so many features or use a higher degree polynomial model to fit the data then the variance increases and bias decreases.
<br><br>

I hope now that you are clear with the working of polynomial regression and the mathematics behind. Next step will be to implement a polynomial model on a dataset using scikit-learn library. So get started.

Thanks for reading :)

