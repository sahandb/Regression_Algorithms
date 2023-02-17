# Regression Algorithms
Implementation of four Regression Model from scratch

At first I create 4 data model with instructions and then use them in our regression models
Experiments on regression problems use 4 different sets of synthetic data with 𝑛 data points {𝑥𝑖 }𝑖-𝑛 with 𝑝 dimensions.

Stack these data points in a 𝑝 × 𝑛 matrix of 𝑋. The univariate response variable 𝒚 depends only on a particular set of features. The 𝑗th feature is denoted by 𝑋𝑗: (𝑗th row of 𝑋) and the 𝑖th data point is denoted by 𝑋:𝑖 (𝑖th column of 𝑋).

Regression A: This regression model is defined as:

![image](https://user-images.githubusercontent.com/24508376/219643938-4cb1b901-52b2-48a9-af17-7ce592671720.png)

where 𝑋:𝑖 ∼ 𝑁(0, 𝐼4) is a four-dimensional input vector and 𝜀 ∼ 𝑁(0,1) is the normal additive Gaussian noise. In this model only, the first two features are related to the response variable 𝒚.

Regression B: The second regression model is as follows:

![image](https://user-images.githubusercontent.com/24508376/219643989-40fe5304-c7d3-43c8-99a6-b4228aa85d7f.png)

where 𝑋:𝑖 ∼ 𝑁(0, 𝐼10) $ and 𝜀 ∼ 𝑁(0,1) is the independent noise. The dimension of the true space is 1 and the noise is multiplicative rather than additive in this case.

Regression C: The third regression model is defined as follows:

![image](https://user-images.githubusercontent.com/24508376/219644221-f60b4fe5-c441-4feb-91b2-bf849caab38f.png)


where 𝑋 ∼ 𝑁(0, 𝐼4) is as it is defined in regression (A), and 𝜀 ∼ 𝑁(0,1).

Regression D: The final regression model is as follows:

![image](https://user-images.githubusercontent.com/24508376/219644389-56402b01-abcf-4329-b0cb-c4a7e92c0d20.png)


in which 𝑋 = (𝑋1, . . . , 𝑋10)𝑇 ∼ 𝑁(0, 𝐼10), and 𝜀 ∼ 𝑁(0,1).

Samples of size 𝑛 = 100 drawn out of each regression model and in each set 70%, of the data is used for training and the remaining 30% is used as testing data. The average root mean square error (RMSE) of estimating the test data response variable for different regression models should be reported in each case for fifty times.

After constructing these regression models, you are required to implement the following four algorithms.

I. Linear Regression

First objective model is multiple linear regression


![image](https://user-images.githubusercontent.com/24508376/219645018-60d17bb0-cd0c-4a8d-80e5-b874a502212b.png)


II. Ridge Regression (Regression with 𝑳𝟐-regularization )

Second one is Ridge regression as we know lambda is regularization penalty


As you know, when a regularization penalty (scaled by λ) is added to the objective function of linear regression, we call it ridge regression, and its objective function is:

![image](https://user-images.githubusercontent.com/24508376/219645218-c97f2835-d6c8-4672-8c07-b88b61be0a5d.png)

which can be solved in closed-form as:

![image](https://user-images.githubusercontent.com/24508376/219645280-e2c76a22-0b03-47cf-bba6-464ad633330d.png)


  III. Coordinate Descent for Lasso
  
  Third is Lasso regression and we don’t have any closed form bcz we cant differentiate from objective function then for that then we use coordinate decent, in which at each step we optimize over one component of the unknown parameter vector and fix all other unknown components. Aka shooting algorithm.

  
  The Lasso optimization problem can be formulated as:
  
  ![image](https://user-images.githubusercontent.com/24508376/219645404-220b7b2c-0fca-474d-b7b4-93964cd32d0a.png)


in which
![image](https://user-images.githubusercontent.com/24508376/219645453-c4f988c3-2ae8-45ea-8c9a-5f0e60d06e9d.png)


Since the 𝐿1-regularization term in the objective function is non-differentiable, it is not clear how gradient descent or SGD could be used to solve this optimization problem, directly. Another approach to solve an optimization problem is coordinate descent, in which at each step we optimize over one component of the unknown parameter vector and fix all other unknown components. The descent path is a sequence of steps, each of which is parallel to a coordinate axis in 𝑅𝑑 .

This gives us the following algorithm, known as the shooting algorithm:

![image](https://user-images.githubusercontent.com/24508376/219645590-71e56871-5b85-4f99-911e-187ea1fcf6df.png)

(Source: Murphy, Kevin P. Machine learning: a probabilistic perspective. MIT press, 2012.)

The “soft thresholding” function is defined as:

𝑠𝑜𝑓𝑡(𝑎, 𝛿) = 𝑠𝑖𝑔𝑛(𝑎)(|𝑎| − 𝛿)+

where (|𝑎| − 𝛿)+ = max ((|𝑎| − 𝛿), 0) is the positive part of (|𝑎| − 𝛿).

IV. Kernelized Ridge Regression

Four is kernel ridge regression that we multiple all sample with a kernel (gaussian or poly) over all samples and then find parameter (a) and then with that a we predict the test samples.


We replace all data points with their feature vector: 𝑥𝑖 → Φ𝑖 = Φ(𝑥𝑖 ). In this case, the number of dimensions can be much higher, or even infinitely higher, than the number of data points. There is a neat trick that allows us to perform ridge regression in high-dimensional space as follows:

![image](https://user-images.githubusercontent.com/24508376/219645782-72738094-4df1-4547-a9a3-96605019f6fd.png)

where 𝐾(𝑥, 𝑥𝑖 ) = 𝛷(𝑥)𝛷(𝑥𝑖 )𝑇.

I applied four mentioned methods on four generated data sets and report the RMSE for different values of the regularization parameter 𝜆 = {0.5, 1, 10, 100, 1000}.


# Some important questions here:

Explain advantages and disadvantages of kernel ridge regression? Under what circumstances is the kernel regression better than the other methods?
That’s a common approach to map samples to a high dimensional space using a nonlinear mapping, and then learn the model in the high dimensional space and disadvantage is computational complexity
Under this circumstance :Learning non-linear relationships between predictor variables and responses is a fundamental problem in machine learning .One state-of-the-art method is Kernel Ridge Regression (KRR)

What if 𝜆 is set to an extremely large value? Explain the role of the regularization parameter 𝜆 on training phase.
To solve some ill-posed problems or prevent overfitting, L2 regularization is used, as formulated in Ridge Regression. λ is a positive parameter that controls w: the larger is λ, the smaller is ∥w∥2

In Kernel ridge regression, which kernel is better than the other, and why?
Gaussian is better because it has sigma parameter and lambda that leads to more complex model than polynomial and it has more generalization

When is lasso worse than ridge regression?
Ridge tends to give small but well distributed weights, because the l2 regularization cares more about driving big weight to small weights, instead of driving small weights to zeros. If you only have a few predictors, and you are confident that all of them should be really relevant for predictions, try Ridge as a good regularized linear regression method.

Compare four mentioned methods, in terms of, noise, outlier, the linearly separable data, the non-linearly separable data, number of samples, and number of features. What is the effect of 𝜀 on each regression algorithm?






