import numpy as np
import matplotlib.pyplot as plt


def yh(y, h):
    check = y == h
    return check * 2 - 1


def error(Y, H, W):
    result = 0
    for i, y in enumerate(Y):
        check = y == H[i]
        if check is False:
            result += 1 * W[i]
    return result


def alpha(e, eta=0.5):
    result = eta * np.log((1 - e) / e)
    return result


def Z(Y, H, alpha, t=1):
    result = 0
    for i, y in enumerate(Y):
        result += D(Y, H, alpha, t) * np.exp(-alpha * yh(y, H[i]))
    return result


def solver(X, Y, D, split_point, condition="smaller or equal"):
    print(f"\t Condition: x {condition} to {split_point}")
    if condition == "smaller or equal":
        H = [int(x <= split_point) * 2 - 1 for x in X]
    elif condition == "greater":
        H = [int(x > split_point) * 2 - 1 for x in X]
    error_1 = error(Y, H, D)
    print("Error: ", error_1)

    alpha_1 = alpha(error_1)
    print("Alpha: ", alpha_1)

    D_new = D
    for i, y in enumerate(Y):
        D_new[i] = D[i] * np.exp(-alpha(error_1) * yh(y, H[i]))
    Z = sum(D_new)
    print("Z: ", Z)
    D_new = D_new / Z

    print("Weight: ", D_new)
    print("\n")

    for i, _ in enumerate(test):
        test[i] += alpha_1 * H[i]
    return error_1, test, D_new


def plotit(X, Y):
    plt.scatter(X[0, 0], X[0, 1], marker="o", color="red")
    plt.scatter(X[1, 0], X[1, 1], marker="+", color="blue")
    plt.scatter(X[2, 0], X[2, 1], marker="+", color="blue")
    plt.scatter(X[3, 0], X[3, 1], marker="o", color="red")
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.xlabel("x_1")
    plt.ylabel("x_2")


if __name__ == "__main__":
    np.set_printoptions(legacy="1.25")

    # example from ppt.
    X = np.array(
        [
            [3, 4],
            [3.2, 2],
            [3.7, 1],
            [4, 4.5],
            [4.5, 6],
            [5, 4],
            [5.1, 7],
            [5.5, 5.5],
            [6.5, 2],
            [7, 6],
        ]
    )
    Y = [1, 1, -1, -1, 1, -1, 1, 1, -1, -1]
    D = [1 / len(Y) for _ in Y]
    test = [0 for _ in Y]

    error_1, test, D = solver(
        X[:, 0], Y, D, split_point=3.5, condition="smaller or equal"
    )
    error_1, test, D = solver(
        X[:, 0], Y, D, split_point=6, condition="smaller or equal"
    )
    error_1, test, D = solver(X[:, 1], Y, D, split_point=5, condition="greater")
    print(f"Y values: {Y}")
    print(f"Final Hypothesis: {test}")

    # hw problem.
    print("\n HW problem.")
    X = np.array(
        [
            [0, -1],
            [1, 0],
            [-1, 0],
            [0, 1],
        ]
    )
    Y = [-1, 1, 1, -1]
    D = [1 / len(Y) for _ in Y]
    test = [0 for _ in Y]
    error_e = 1
    error_1, test, D = solver(
        X[:, 0], Y, D, split_point=-0.5, condition="smaller or equal"
    )
    error_e = error_e * 2 * np.sqrt(error_1 * (1 - error_1))

    error_1, test, D = solver(X[:, 0], Y, D, split_point=0.5, condition="greater")
    error_e = error_e * 2 * np.sqrt(error_1 * (1 - error_1))

    error_1, test, D = solver(
        X[:, 1], Y, D, split_point=-0.5, condition="smaller or equal"
    )
    error_e = error_e * 2 * np.sqrt(error_1 * (1 - error_1))

    error_1, test, D = solver(X[:, 1], Y, D, split_point=0.5, condition="greater")
    error_e = error_e * 2 * np.sqrt(error_1 * (1 - error_1))

    print(f"Y values: {Y}")
    print(f"Final Hypothesis: {test}")

    print(f"Error of ensemble: {error_e}")

    plotit(X, Y)
    plt.plot([-0.5, -0.5], [-1.5, 1.5], label="t=1")
    plt.plot([0.5, 0.5], [-1.5, 1.5], label="t=2")
    plt.plot([-1.5, 1.5], [-0.5, -0.5], label="t=3")
    plt.plot([-1.5, 1.5], [0.5, 0.5], label="t=4")
    plt.legend()
    plt.show()
