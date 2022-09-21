def matmul(M1, M2):
    n = matshape(M1)[0]
    m = matshape(M2)[1]
    d = matshape(M1)[1]
    assert(d == matshape(M2)[0])

    res = [[0]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(d):
                res[i][j] += M1[i][k] * M2[k][j]
    return res

def matshape(M):
    return len(M), len(M[0])

def relu(x):
    for i, x_i in enumerate(x):
        x[i] = 0 if x_i <= 0 else x_i