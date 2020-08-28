# package
import copy as cp
import pywt as pw
import numpy as np
import skimage.io
import skimage.metrics
import scipy.fftpack

# quadrature mirror filter
QMF = 'dmey'

# decomposition level
DLV = 1

# quantization table
QTZ_TBL = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99]])

# sign table
SGN_TBL = np.array([
    [ True, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False]])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def bdct2(img):

    row = img.shape[0]
    col = img.shape[1]
    cff = skimage.util.view_as_blocks(img, (8, 8))
    cff = scipy.fftpack.dct(cff, axis=2, norm='ortho')
    cff = scipy.fftpack.dct(cff, axis=3, norm='ortho')
    cff = np.reshape(cff.swapaxes(1, 2), (row, col))
    return cff


def bidct2(cff):

    row = cff.shape[0]
    col = cff.shape[1]
    img = skimage.util.view_as_blocks(cff, (8, 8))
    img = scipy.fftpack.idct(img, axis=2, norm='ortho')
    img = scipy.fftpack.idct(img, axis=3, norm='ortho')
    img = np.reshape(img.swapaxes(1, 2), (row, col))
    return img


def thresh_swt2(cff, val):

    c = cp.deepcopy(cff)
    for d in range(DLV):
        c[d] = list(c[d])
        c[d][0] = pw.threshold(c[d][0], val)
        c[d][1] = pw.threshold(c[d][1], val)

    return c


def proj_dct2(c, u, l):

    return np.minimum(np.maximum(c, l), u)


def fienup(con_upp, con_low):

    # initial guess
    rec_vec = bidct2((con_upp + con_low) / 2)

    for i in range(1024):
        # copy
        cpy_vec = rec_vec

        # thresholding
        rec_vec = pw.swt2(rec_vec, QMF, DLV, norm=True)
        rec_vec = thresh_swt2(rec_vec, 1)
        rec_vec = pw.iswt2(rec_vec, QMF, DLV)

        # projection
        rec_vec = bdct2(rec_vec)
        rec_vec = proj_dct2(rec_vec, con_upp, con_low)
        rec_vec = bidct2(rec_vec)

        # acceleration
        rec_vec = rec_vec + (i - 1) * (rec_vec - cpy_vec) / (i + 2)

    return rec_vec

if __name__ == '__main__':

    # original image
    org_img = skimage.io.imread('data/camera.png')
    
    # image size
    row = org_img.shape[0]
    col = org_img.shape[1]

    # replicate quantization table
    rep_qtz_tbl = np.tile(QTZ_TBL, [row >> 3, col >> 3])

    # replicate sign table
    rep_sgn_tbl = np.tile(SGN_TBL, (row >> 3, col >> 3))

    # index of known sign bit
    ind_sbt = np.where(rep_sgn_tbl == True)

    # index of unknown sign bit
    ind_sbf = np.where(rep_sgn_tbl == False)

    # original coefficient
    org_cff = bdct2(org_img)

    # quantization and dequantization
    deg_cff = np.round(org_cff / rep_qtz_tbl) * rep_qtz_tbl

    # reduction of sign bit
    deg_cff[ind_sbf] = abs(deg_cff[ind_sbf])

    # constraint (upper)
    con_upp = np.zeros([row, col])
    con_upp[ind_sbt] = +deg_cff[ind_sbt]
    con_upp[ind_sbf] = +deg_cff[ind_sbf] 

    # constraint (lower)
    con_low = np.zeros([row, col])
    con_low[ind_sbt] = +deg_cff[ind_sbt]
    con_low[ind_sbf] = -deg_cff[ind_sbf]

    # sign retrieval via Fienup algorithm
    rec_img = fienup(con_upp, con_low)

    # clip
    rec_img = np.round(rec_img)
    rec_img = np.maximum(rec_img, 0)
    rec_img = np.minimum(rec_img, 255)
    rec_img = np.array(rec_img, np.uint8)
    skimage.io.imsave('rec.png', rec_img)

    # PSNR
    print("rec: {:4.2f}".format(skimage.metrics.peak_signal_noise_ratio(org_img, rec_img)))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # for comparison

    # random image 
    rnd_sgn = np.sign(np.random.normal(0, 1, [row, col]))
    rec_img = deg_cff
    rec_img[ind_sbf] = rec_img[ind_sbf] * rnd_sgn[ind_sbf]
    rec_img = bidct2(rec_img)
    
    rec_img = np.round(rec_img)
    rec_img = np.maximum(rec_img, 0)
    rec_img = np.minimum(rec_img, 255)
    rec_img = np.array(rec_img, np.uint8)
    skimage.io.imsave('rnd.png', rec_img)

    print("rnd: {:4.2f}".format(skimage.metrics.peak_signal_noise_ratio(org_img, rec_img)))

    # partial image 
    rec_img = np.zeros([row, col])
    rec_img[ind_sbt] = deg_cff[ind_sbt]
    rec_img = bidct2(rec_img)
    
    rec_img = np.round(rec_img)
    rec_img = np.maximum(rec_img, 0)
    rec_img = np.minimum(rec_img, 255)
    rec_img = np.array(rec_img, np.uint8)
    skimage.io.imsave('par.png', rec_img)

    print("par: {:4.2f}".format(skimage.metrics.peak_signal_noise_ratio(org_img, rec_img)))

    # jpg image
    rec_img = np.round(org_cff / rep_qtz_tbl) * rep_qtz_tbl
    rec_img = bidct2(rec_img)
    
    rec_img = np.round(rec_img)
    rec_img = np.maximum(rec_img, 0)
    rec_img = np.minimum(rec_img, 255)
    rec_img = np.array(rec_img, np.uint8)
    skimage.io.imsave('sdh.png', rec_img)

    print("sdh: {:4.2f}".format(skimage.metrics.peak_signal_noise_ratio(org_img, rec_img)))
