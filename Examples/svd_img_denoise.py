import matplotlib.pyplot as plt
import numpy as np

path = 'D:/python_projects/test/'
name = 'noised_lena.png'

img = plt.imread(path + name)

epsilon = 0.55


def de_noise(m, epsilon):
    [u, s, v] = np.linalg.svd(m, False)
    l = np.count_nonzero(np.abs(s) > epsilon)

    print "Number of used singular values = %i" % l
    s = np.diag(s)
    s[l:, l:] = 0
    return np.dot(u, np.dot(s, v))

def img_de_noise(img, epsilon):
    img1 = np.ones(img.shape)
    for i in range(0, 3): #img.shape[2]):
        img1[:,:,i] = de_noise(img[:,:,i], epsilon)
    """
    if image has alpha channel with all ones
    then svd reconstruction of alpha channel is unuseful
    """
    return img1

denoised = img_de_noise(img, epsilon)
plt.imsave(path + 'denoised_' + name, denoised)