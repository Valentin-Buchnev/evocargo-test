from logistic_regression import LogisticRegression
import torch
from sklearn import datasets, model_selection


# everywhere seed is equal to 42
# 0.9666 - accuracy on test set

def test_passing():
    
    X, Y = datasets.make_classification(n_samples=100, random_state=42)

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=42)

    LR = LogisticRegression(20, seed=42)
    LR.fit(X_train, Y_train)

    Y_pred = LR.predict(X_test)
    Y_pred = (Y_pred > 0.5).clone().detach().type(torch.float32)

    Y_test = torch.tensor(Y_test, dtype=torch.float32)
    Y_test = torch.reshape(Y_test, (-1, 1))

    accuracy = 1 - torch.mean(torch.abs(Y_pred - Y_test)).item()

    assert abs(accuracy - 0.9666) < 0.01