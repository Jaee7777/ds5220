import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score


def problem1():
    print("\t Problem 1:")
    data = np.array([[1, 52], [2, 59], [4, 67], [6, 81], [8, 90]])

    # using closed form solution
    x = data[:, 0]
    y = data[:, 1]
    n = len(data)
    w1 = (n * sum(x * y) - sum(x) * sum(y)) / (n * sum(x**2) - (sum(x) ** 2))
    print(f"w1 is {w1}")
    w0 = sum(y - w1 * x) / n
    print(f"w0 is {w0}")

    MSE = sum((w0 + w1 * x - y) ** 2) / n
    print(f"MSE is {MSE}")

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

    # plot line
    xx = np.linspace(1, 8, 100)
    yy = w0 * np.ones(len(xx)) + w1 * xx

    plt.figure(figsize=(10, 8))
    plt.plot(x, y, "o")
    plt.plot(xx, yy)
    plt.title("Parabola Fit")
    plt.tight_layout()
    plt.show()
    return


def problem3():
    print("\t Problem 3:")
    data = np.array([[-2, 0], [-1, 0], [0, 1], [1, 0], [2, 0]])
    x = data[:, 0]
    y = data[:, 1]
    n = len(x)

    # create X and compute w using normal equation
    X = np.column_stack((np.ones(n), x, x**2))
    w = np.linalg.inv(X.transpose() @ X) @ X.transpose() @ y

    print(f"Coefficients are {w}")

    # plot parabola
    xx = np.linspace(-2, 2, 100)
    yy = w[0] * np.ones(len(xx)) + w[1] * xx + w[2] * xx**2

    plt.figure(figsize=(10, 8))
    plt.plot(x, y, "o")
    plt.plot(xx, yy)
    plt.title("Parabola Fit")
    plt.tight_layout()
    plt.show()
    return


def problem5():
    print("\t Problem 5:")
    df = pd.read_csv("data/housing.csv")

    # part 1
    print("\t 1. Summary of Data:")
    print(df.info())
    print(df.describe())

    ohc = OneHotEncoder(sparse_output=False)
    encoded = ohc.fit_transform(df[["ocean_proximity"]])
    encoded_df = pd.DataFrame(
        encoded, columns=ohc.get_feature_names_out(["ocean_proximity"])
    )
    df = pd.concat([df, encoded_df], axis=1)

    # part 2
    print("\t 2. Correlation of median_house_value:")
    df_float = df.drop("ocean_proximity", axis=1)
    corr_matrix = df_float.corr()
    print(corr_matrix["median_house_value"])

    # part 4
    print(df.head())
    vt = VarianceThreshold(threshold=0.05)
    vt.fit(df_float)
    print(
        pd.DataFrame(
            {"variance": vt.variances_, "select_feature": vt.get_support()},
            index=df_float.columns,
        )
    )
    df_float = df_float.drop("ocean_proximity_ISLAND", axis=1)

    print("\t Check NaN:")
    print(df_float.isna().any())
    print("\t Check number of NaN:")
    print(df_float.isnull().sum())

    df_float = df_float.fillna(df["total_bedrooms"].median())

    print("We substitude NaN in total_bedrooms to its median:")
    print(df_float.isna().any())

    print(df_float.head())

    target_feature = "median_house_value"
    features = [col for col in df_float.columns if col != target_feature]

    y = df_float[target_feature]
    X = df_float[features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # part 5
    selector = SelectKBest(score_func=f_regression, k=2)
    selector.fit(X_train, y_train)

    top_features = selector.get_feature_names_out(X.columns)
    print(f"Top 2 correlated features: {top_features}")

    X_train_top2 = X_train_scaled[:, selector.get_support()]
    X_test_top2 = X_test_scaled[:, selector.get_support()]

    # Part 6
    model = LinearRegression()
    model.fit(X_train_top2, y_train)

    y_pred = model.predict(X_train_top2)

    rmse_train = mean_squared_error(y_train, y_pred, squared=False)
    print(f"Train set RMSE: {rmse_train}")
    r2_train = r2_score(y_train, y_pred)
    print(f"Train set R2: {r2_train}")

    y_pred = model.predict(X_test_top2)

    rmse_test = mean_squared_error(y_test, y_pred, squared=False)
    print(f"Test set RMSE: {rmse_test}")
    r2_test = r2_score(y_test, y_pred)
    print(f"Test set R2: {r2_test}")

    # part 7
    alpha_list = [10000, 1000, 100, 10, 1, 0.1, 0.01]
    for alpha in alpha_list:
        model = Ridge(alpha=alpha)
        model.fit(X_train_top2, y_train)

        y_pred = model.predict(X_test_top2)

        print(f"\t Alpha = {alpha}")
        rmse_test = mean_squared_error(y_test, y_pred, squared=False)
        print(f"Test set with Regularization RMSE: {rmse_test}")
        r2_test = r2_score(y_test, y_pred)
        print(f"Test set with Regularization R2: {r2_test}")

    # part 8
    model = DecisionTreeRegressor()
    model.fit(X_train_top2, y_train)

    y_pred = model.predict(X_test_top2)

    print(f"\t DecisionTreeRegressor")
    rmse_test = mean_squared_error(y_test, y_pred, squared=False)
    print(f"Test set with Regularization RMSE: {rmse_test}")
    r2_test = r2_score(y_test, y_pred)
    print(f"Test set with Regularization R2: {r2_test}")
    return


def main():
    problem1()
    problem3()
    problem5()
    return


if __name__ == "__main__":
    main()
