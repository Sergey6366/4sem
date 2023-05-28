import numpy as np
import matplotlib.pyplot as plt

# solves linear system of algebraic equations and does 2 graphs: norm depending on #iterations
# second Gauss transformation: {A*A^T * y = b; x = A^T * y}
# here L is Lotkin matrix

n = int(input())
L = np.zeros((n, n))
b = np.empty(n)
xPrev1 = np.zeros(n)
xOne = np.zeros(n)
xTwo = np.zeros(n)

epsn = 1e-8
limK = 1e4

for i in range(n):
    b[i] = pow(-1, i % 2) * (i + 1)
    for j in range(n):
        if i == 0:
            L[0][j] = 1
            continue
        L[i][j] = 1 / (i + j + 1)

L1 = L.dot((L.transpose()))

# Gauss-Seidel method
k = 0
LowD = np.tril(L1)
Up = np.triu(L1) - np.diagflat(np.diag(L1))
normC = max(abs(np.linalg.eig((np.linalg.inv(LowD)).dot(Up))[0])) if n > 1 else 0.5
normCoef = normC / (1 - normC)

normVec = np.inf
normOne = list()

if normC >= 1:
    normCoef = np.inf
while (normCoef * normVec > epsn) & (k <= limK):
    xPrev1 = xOne.copy()
    s1 = 0
    for i in range(n):
        for j in range(n):
            if j != i:
                s1 += L1[i][j] * xOne[j]
        if L1[i][i] == 0:
            k *= -1
            break
        xOne[i] = (b[i] - s1) / L1[i][i]
        s1 = 0
    normVec = max(abs(np.linalg.eig(np.tile(xOne - xPrev1, (n, 1)))[0]))
    k += 1
    normOne.append(np.linalg.norm(b - L1.dot(xOne), 1))

np.set_printoptions(precision=9)

print(f"k={k}\nnorm=%.9f" % np.linalg.norm(b - L1.dot(xOne), 1))
print(f"other={normCoef * normVec}\n")
xOrig = L.transpose().dot(xOne)
print(f"x={xOrig}")

# Conjugate Gradients method
r = b - L1.dot(xTwo)
rN = r.copy()
s = r.copy()
k1 = 0

n1 = np.linalg.norm(b - L1.dot(xTwo), 2) / min(abs(np.linalg.eig(L1)[0]))
n2 = n1.copy()
normTwo = list()
o = 1

kk1 = 0
while (n1 > epsn) & (k1 <= limK):
    g = L1.dot(s)  # s(k)
    sdg = s.dot(g)
    if sdg == 0:
        k1 *= -1
        break
    rdr = r.dot(r)
    t = rdr / sdg  # tau(k), r(k), s(k)
    xTwo = xTwo + t * s  # x(k+1), x(k), s(k)
    rN = r - t * g  # r(k+1), r(k)
    if rdr == 0:
        k1 *= -1
        break
    v = (rN.dot(rN)) / rdr  # v(k)
    s = rN + v * s

    r = rN

    n2 = np.linalg.norm(b - L1.dot(xTwo), 2) / min(abs(np.linalg.eig(L1)[0]))  # ???????
    o += 1 if n2 > n1 else 0
    n1 = n2

    k1 += 1
    if k1 % n == 0:
        kk1 += 1
        normTwo.append(np.linalg.norm(b - L1.dot(xTwo), 1))

print(f"\n\nk={k1}\nnorm=%.9f\nother=%.9f\n" % (np.linalg.norm(b - L1.dot(xTwo), 1), n1))
xOrig1 = L.transpose().dot(xTwo)
print(f"x={xOrig1}")
if k1 < 0:
    print(f"norm increased {o} times")
print(f"\n\nOriginal solution:\n{np.linalg.solve(L, b)}")


# PLOT
ar1, ar2 = np.empty(abs(k)), np.empty(kk1)
for i in range(k):
    ar1[i] = i + 1
for i in range(kk1):
    ar2[i] = i + 1

fig1 = plt.figure(1)
fig2 = plt.figure(2)
ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)
green = '#008000'
blue = '#0000ff'
black = '#000000'
lime = '#008000'
purple = '#800080'

ax1.plot(ar1, normOne, color=green)
ax2.plot(ar2, normTwo, color=black)
if n > 2:
    ax1.semilogy()
    ax2.semilogy()
plt.show()
