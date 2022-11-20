# Improvements & Debugging

## Relevant Terms

> Underfitting is called "Simplifying assumption" (Model is HIGHLY
> BIASED towards its assumption). your model will think linear
> hyperplane is good enough to classify your data which may not
> be true. consider you are shown a picture of cat 1000 times,
> Now you are blindfolded, No matter Whatever you are shown the
> 1001th time, probability that you will say cat is very high
> (You are HIGHLY BIASED that the next picture is also gonna
> be a cat). Its because you believe its gonna be a cat anyway.
> Here you are simplifying assumptions
> 
> In statistics, Variance informally means how far your data
> is spread out. Overfitting is you memorise 10 qns for your exam
> and on the next day exam, only one question has been asked in the
> question paper from that 10 you read. Now you will answer that one
> qn correctly just like in the book, but you have no idea what the
> remaining questions are(Question are HIGHLY VARIED from what
> you read). In overfitting, model will memorise the entire train
> data such that it will give high accuracy on train but will
> suck in test. Hope its helps
>
> -- https://stackoverflow.com/a/2002150


## Possible Improvements

- More training examples
  - fix high variance in learning/validation curve
    (big distance between the lines)
- Smaller sets of features
  - fix high variance in learning/validation curve
    (big distance between the lines)
- Additional features
  - fix high bias (oversimplified model)
- Polynomial features
  - fix high bias (oversimplified model)
- Decreasing regularization
  - fix high bias (oversimplified model)
- Increasing regularization
  - fix high variance in learning/validation curve
    (big distance between the lines)

## Choose Model

If a model is choosen based on the accuracy of different approaches there
should be a cross validation set additionally to the training and test set.
A typical split is 40% training data, 20% training set and 20% CV set.
The highest accuracy is then calculated based on the CV set and the accuracy
calculated based on the chosen model trained with the training data and
accuracy then measured using the test data.

## Bias/Variance

High bias (underfitting): both J_train and J_CV will be high.Also, J_CV≈J_train.

High variance (overfitting): J_train will be low and J_CV will be much greater
than J_train.

## Regularization

Checkout [this script](./snippets/regularization.py)
for a practical implementation of regularization.

## Validation & Learning Curve

To look at the validation and learning curve checkout:
- [Learning Curve](./snippets/learning_courve.py)
- [Validation Curve](./snippets/validation_curve.py)
