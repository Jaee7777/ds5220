def manhatten(a, b):
    if len(a) != len(b):
        print("input length does not match")
        return
    result = 0
    for i in range(len(a)):
        result += abs(a[i] - b[i])
    return result


if __name__ == "__main__":
    table = [
        [1, 1, 0, -1, 1, -1, 0, 1],
        [4, 2, 0, 4, 0, -1, -4, 0],
        [1, 3, 1, 0, -2, 1, 0, -3],
    ]
    y = [1, 1, 1, 1, 0, 0, 0, 0]
    x = []
    for i in range(8):
        x.append([table[0][i], table[1][i], table[2][i]])

    target = [1, 0, 1]
    distance_result = []

    for i in range(8):
        distance_result.append(manhatten(x[i], target))

    print(distance_result)
