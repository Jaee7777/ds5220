import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def problem1():
    data = np.array([[1, 52], [2, 59], [4, 67], [6, 81], [8, 90]])

    # using closed form solution
    x = data[:, 0]
    y = data[:, 1]
    n = len(data)
    w1 = (n * sum(x * y) - sum(x) * sum(y)) / (n * sum(x**2) - (sum(x) ** 2))
    print(w1)
    w0 = sum(y - w1 * x) / n
    print(w0)

    y_3 = w0 + w1 * 3
    y_5 = w0 + w1 * 5

    print(f"Score after studying 3 hours is {y_3}")
    print(f"Score after studying 5 hours is {y_5}")

    # using sklearn
    X = np.array([[i] for i in data[:, 0]])
    y = np.array([[i] for i in data[:, 1]])

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Intercept: {model.intercept_}")
    print(f"Slope: {model.coef_}")
    return


def problem2():
    return


def quadratic_regression_normal_equation(X, y):
    X = np.array(X)
    y = np.array(y)

    # Construct the design matrix with polynomial features
    X_b = np.column_stack((np.ones(len(X)), X, X**2))

    # Calculate the coefficients using the normal equation
    theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    return theta_best


def problem3():
    data = np.array([[-2, 0], [-1, 0], [0, 1], [1, 0], [2, 0]])
    X = data[:, 0]
    y = data[:, 1]
    coef = quadratic_regression_normal_equation(X, y)
    print(coef)
    return


def problem5():
    df = pd.read_csv("data/housing.csv")

    print("\t 1. Summary of Data:")
    print(df.info())
    print(df.describe())

    print("\t 2. Correlation of median_house_value:")
    df_float = df.drop("ocean_proximity", axis=1)
    corr_matrix = df_float.corr()
    print(corr_matrix["median_house_value"])

    print("\t 4. Check NaN:")
    print(df.isna().any())

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.xticks(rotation=45, ha="right")
    plt.title("5. Correlation Matrix of Features")
    plt.tight_layout()
    plt.show()
    return


def main():
    problem1()
    # problem3()
    # problem5()
    return


if __name__ == "__main__":
    main()
