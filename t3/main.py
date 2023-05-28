import numpy as np


# searching for the eigenvalues of matrix M (symmectrical)

def delTa(matr, dim):
    ans = 0
    s = 0
    for k in range(dim):
        for m in range(dim):
            if m != k:
                s += abs(matr[k][m])
        ans = max(s, ans)
    return ans


maxk = 1000
eps = 1e-8
n = int(input())
M = np.empty((n, n))

for i in range(n):
    for j in range(i + 1):
        M[i][j] = n - i
        M[j][i] = M[i][j]

# Jacobi Rotation Method
k2 = 0
B = M.copy()
eig2 = []
mx2 = np.inf
while mx2 > eps and k2 < maxk:
    if n == 1:
        B[0][0] = M[0][0]
        break

    cur = k2 % (n-1)
    mx = 0
    q = 0
    for j in range(cur + 1, n):
        if mx < abs(B[cur][j]):
            q = j
            mx = abs(B[cur][j])

    tau = (B[cur][cur] - B[q][q])/(2*B[cur][q])
    t = 1/(tau + np.sign(tau)*np.sqrt(tau*tau + 1)) if tau != 0 else 1
    c = 1/np.sqrt(t*t + 1)
    s = t*c

    G = np.eye(n)
    G[cur][cur], G[q][q], G[cur][q], G[q][cur] = c, c, -s, s
    B = (G.transpose()).dot(B).dot(G)

    mx2 = delTa(B, n)
    k2 += 1

for i in range(n):
    eig2.append(B[i][i])

np.set_printoptions(precision=2)
srt2 = np.array(eig2)
srt2[::-1].sort()
print(srt2, '\n', k2, '\n')


# QR algorithm. works for positive defined matricies
k1 = 0
A = M.copy()
eig1 = []
mx1 = np.inf
while mx1 > eps and k1 < maxk:
    MOD = A[k1 % n][k1 % n] * np.eye(n)
    Q, R = np.linalg.qr(A - MOD)
    A = R.dot(Q) + MOD
    mx1 = delTa(A, n)
    k1 += 1

for i in range(n):
    eig1.append(A[i][i])

srt1 = np.array(eig1)
srt1[::-1].sort()
print(srt1, '\n', k1)


eig3 = np.empty(n)
for i in range(1, n+1):
    eig3[i-1] = 1/(2*(1-np.cos((2*i-1)*np.pi/(2*n+1))))

print(f"\ncomparing 1 and 2 results to original:\n{np.linalg.norm(srt1-eig3, np.inf)}"
      f"\n{np.linalg.norm(srt2-eig3, np.inf)}")
print(f"comp:\n{np.linalg.norm(srt1-srt2, np.inf)}\n")


# eigenvectors for this M (it is possible)
vec = np.zeros((n, n))
norms = []
for i in range(n):
    lamb = srt1[i]
    vec[i][0] = 1
    if n > 1:
        vec[i][1] = (lamb-1)/lamb  # det == 1 => lamb != 0
    for j in range(2, n):
        vec[i][j] = ((2*lamb-1)*vec[i][j-1] - lamb*vec[i][j-2])/lamb
    vec[i] = vec[i]/(np.sqrt(vec[i].dot(vec[i])))
    norms.append(np.linalg.norm(M.dot(vec[i]) - lamb*vec[i], np.inf))

print(f"eigenvectors:\n{vec}\n\nmax norm:{max(norms)}")
