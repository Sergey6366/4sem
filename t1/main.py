import numpy as np


# finds the determinant of matrix C

n = int(input())
a = np.empty((n, n))
c = np.zeros((n, n))

for i in range(n):
    for j in range(i + 1):
        a[i][j] = n - i
        a[j][i] = a[i][j]

# HOLESSKY (Cholesky) decomposition - for positive defined matrix
for j in range(n):
    s1 = 0
    for k in range(0, j):
        s1 += c[j][k] * c[j][k]
    if a[j][j] - s1 < 0:
        print("matrix is not positive definite")
        break
    c[j][j] = np.sqrt(a[j][j] - s1)

    for i in range(j + 1, n):
        s2 = 0
        for k in range(0, j):
            s2 += c[i][k] * c[j][k]
        if c[j][j] == 0:
            print("matrix is not positive definite")
            break
        c[i][j] = (a[i][j] - s2) / c[j][j]

# print(f"A=\n{a}\nC=\n{c}\nC*CT=\n{c.dot(c.transpose())}")

detH = 1
for i in range(n):
    detH *= c[i][i] * c[i][i]

# LU decomposition - for strictly regular matrix
u = a
e = np.empty((n, n))
for j in range(n):
    e = np.eye(n)
    for i in range(j, n):
        if u[j][j] == 0:
            print("matrix isn't strictly regular")
            break
        e[i][j] = -u[i][j] / u[j][j]
    u = e.dot(u)

detLU = 1
for i in range(n):
    detLU *= u[i][i]

detOrig = np.linalg.det(a)
print(f"kind of original: {detOrig}")
print("HOLESSKY det:", detH)
print("LU det:", detLU)


inaccH = detH/detOrig*100 - 100
inaccLU = detLU/detOrig*100 - 100
print("\nHOLESSKY inaccuracy:", "%.4f" % inaccH)
print("LU inaccuracy:", "%.4f" % inaccLU)
