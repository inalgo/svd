import numpy as np

def c_bidiag(a,b):
    f = np.diagflat(a)
    f += np.diagflat(b, 1)
    return f

def bidiag(A):
    transposed = False
    if (A.shape[1] > A.shape[0]):
        A = A.transpose()
        transposed = True

    m, n = A.shape

    b = np.zeros((n + 1, 1))
    a = np.zeros((n + 1, 1))
    U = np.zeros((n + 1, m))
    V = np.zeros((n + 1, n))

    V[1, 0] = 1
    stop = False
    for k in range(1, n):
        U[k] = np.dot(A, V[k]) - b[k - 1]*U[k - 1]
        a[k] = np.linalg.norm(U[k])
        if (np.abs(a[k]) < 10**-4):
            stop = True
            break;
        U[k] /= a[k]
        V[k + 1] = np.dot(A.transpose(), U[k]) - a[k]*V[k]
        b[k] = np.linalg.norm(V[k + 1])
        if (np.abs(b[k]) < 10**-4):
            stop = True
            break;
        V[k + 1] /= b[k]

    if (not stop):
        U[n] = np.dot(A, V[n]) - b[n - 1]*U[n - 1]
        a[n] = np.linalg.norm(U[n])
        U[n] /= a[n]

    B = c_bidiag(a[1:], b[1:-1])
    if (not transposed):
        return [U[1:].transpose(), B, V[1:]]
    else:
        return [V[1:].transpose(), B.transpose(), U[1:]]

def f_char(k, l, B):
    n = B.shape[0]
    f0 = B[n-1, n-1] - l
    if (k == 0):
        return f0
    f1 = (B[n-2, n-2] - l)*f0 - B[n-1, n-2]**2
    if (k == 1):
        return f1
    f = 0
    for i in range(n-3, n - k - 2, -1):
        f = (B[i, i] - l)*f1 - B[i + 1, i]**2*f0
        f0 = f1
        f1 = f
    return f/f0

def eig(B, t0, k):
    n = B.shape[0]
    for i in range(0, k):
        t0 = f_char(n - 1, t0, B)
        print i, " | ", t0
    return t0

def eig1(B, k):
    v0 = np.ones(B.shape[1])
    for i in range(0, k):
        v1 = np.dot(A, v0)
        mu = np.linalg.norm(v1)
        v0 = v1/mu
        print mu, " | ", v0

    return [mu, v0]

def found_det(B, l):
    n = B.shape[0]
    eps = 10**-5
    e = B[0, 0] - l
    if (e == 0):
        e = eps**2 * B[1, 0]**2
    nu = 0
    nu1 = 1/e
    si = 0
    si1 = 0
    e1 = 0
    neg_count = 0
    if (e < 0):
        neg_count += 1
    for i in range(1, n):
        e = B[i, i] - l - B[i - 1, i]**2/e1
        if (e == 0):
            e = (B[i, i] - l)*eps**2
        if (e < 0):
            neg_count += 1
        nu2 = 1/e*((B[i, i] - l)*nu1 + 1 - (B[i - 1, i]**2/e1)*nu)
        si2 = 1/e*((B[i, i] - l)*si1 + 2*nu1 - (B[i - 1, i]**2/e1)*si)
        e1 = e
        nu = nu1
        nu1 = nu2
        si = si1
        si1 = si2
    return [nu1, si1, neg_count]

m = 3
n = 3


A = np.random.randn(m,n) #very bad at uniformy distributioned random

A = np.array([0.38049728,  0.49039663,  0.73064887,  0.46382454,  0.9332573 ,
  0.92895418,  0.93233151,  0.10202367,  0.63185114,  0.95555188,
  0.2972574 ,  0.14169525,  0.45809888,  0.89940305,  0.00112226,
  0.04622205,  0.49898289,  0.61711454,  0.64726465,  0.28850816]
  ).reshape((4,5)) #.transpose()

A = np.array(
[[0, 3, 6],
[1, 4, 7],
[2, 5, 8]]
)

#A = np.random.randn(m,n)

print A.round(4)
[U, B, V] = bidiag(A)

print "\n U = \n"
print U.round(4)
print "\n B = \n"
print B.round(4)
print "\n V = \n"
print V.round(4)

D = np.dot(U, np.dot(B, V))

#print '\n reconstructed \n'
#print D.round(4)


BB = np.dot(B, B.transpose())
[q1,q2,q3] = np.linalg.svd(B);
print "\n BB singluar values =\n"
print q2**2

print "\n BB U singluar values =\n"
print q1

print "\n BB V singluar values =\n"
print q3



print '\n Difference \n'
print np.linalg.norm(A - D)
