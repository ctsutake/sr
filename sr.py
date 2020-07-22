import copy as cp
import pywt as pw
import numpy as np
import scipy.fftpack
import skimage.io
import skimage.metrics

# quadrature mirror filter
QMF = 'sym6'

# decomposition level
DLV = 2

# quantization table
QTZ_TBL = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99]]
    ) * 1.0

# sign table
SGN_TBL = np.array([
    [ True,  True,  True, False, False, False, False, False],
    [ True,  True, False, False, False, False, False, False],
    [ True, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False]]
    )
#SGN_TBL = np.ones([8, 8])
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


def sum_swt2(a, b):

    c = cp.deepcopy(a)

    for d in range(DLV):
        a[d] = list(a[d])
        b[d] = list(b[d])
        c[d] = list(c[d])
        c[d][0] = np.array(a[d][0]) + np.array(b[d][0])
        c[d][1] = np.array(a[d][1]) + np.array(b[d][1])

    return c


def dif_swt2(a, b):

    c = cp.deepcopy(a)

    for d in range(DLV):
        a[d] = list(a[d])
        b[d] = list(b[d])
        c[d] = list(c[d])
        c[d][0] = np.array(a[d][0]) - np.array(b[d][0])
        c[d][1] = np.array(a[d][1]) - np.array(b[d][1])

    return c


def thresh_swt2(cff, t):

    c = cp.deepcopy(cff)

    for d in range(DLV):
        c[d] = list(c[d])
        c[d][0] = np.sign(c[d][0]) * np.maximum(np.abs(c[d][0]) - t, 0)
        c[d][1] = np.sign(c[d][1]) * np.maximum(np.abs(c[d][1]) - t, 0)

    return c

def proj(c, u, l):
    
    return np.minimum(np.maximum(c, l), u)

def PhaseMax(anc_vec, con_upp, con_low, reg_prm, pen_prm):

    # image size
    row = anc_vec.shape[0]
    col = anc_vec.shape[1]

    # zero buffer
    zero_array = np.zeros([row, col])
    #zero_array = np.random.normal(0, 1.0, [row, col])
    zero_tuple = pw.swt2(zero_array, QMF, DLV, norm=True)

    # variables
    prim = zero_array
    aux1 = zero_array
    aux2 = zero_tuple
    lag1 = zero_array
    lag2 = zero_tuple

    # begin iteration
    for i in range(50):

        # update prim
        tmp0 = bidct2(aux1 - lag1)
        prim = dif_swt2(aux2, lag2)
        prim = pw.iswt2(prim, QMF, DLV)
        prim = (tmp0 + prim + anc_vec / pen_prm) / 2.0
        
        # update aux1
        tmp1 = bdct2(prim)
        aux1 = tmp1 + lag1
        aux1 = proj(aux1, con_upp, con_low)
        
        # update aux2
        tmp2 = pw.swt2(prim, QMF, DLV, norm=True)
        aux2 = sum_swt2(tmp2, lag2)
        aux2 = thresh_swt2(aux2, reg_prm / pen_prm)
        
        # update lag1
        lag1 = lag1 + tmp1 - aux1

        # Update lag2
        lag2 = sum_swt2(lag2, dif_swt2(tmp2, aux2))

    return prim

def WirtingerFlow(anc_vec, con_upp, con_low, reg_prm):
    
    # image size
    row = anc_vec.shape[0]
    col = anc_vec.shape[1]
    
    # initial guess
    rec = cp.deepcopy(anc_vec)

    # magnitude of cff
    mag = np.abs(bidct2(anc_vec)) ** 2

    # step size
    stp = 0.23 / np.average(mag)    

    for i in range(250):
        
        # gradient descent 
        tmp = bdct2(rec)
        grd = bidct2((tmp ** 2 - mag) * tmp)
        grd = grd / row / col
        rec = rec - stp * grd

        # threhsolding 
        tmp = pw.swt2(rec, QMF, DLV, norm=True)
        tmp = thresh_swt2(tmp, 0.02)
        rec = pw.iswt2(tmp, QMF, DLV)

    return rec

if __name__ == '__main__':

    # original image
    org_img = pw.data.camera()

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

    # anchor vector
    anc_vec = np.zeros([row, col])
    anc_vec[ind_sbt] = deg_cff[ind_sbt]
    anc_vec = bidct2(anc_vec)

    # constraint (upper)
    con_upp = np.zeros([row, col])
    con_upp[ind_sbt] = +deg_cff[ind_sbt] + 0.5 * rep_qtz_tbl[ind_sbt]
    con_upp[ind_sbf] = +deg_cff[ind_sbf] + 0.5 * rep_qtz_tbl[ind_sbf]

    # constraint (lower)
    con_low = np.zeros([row, col])
    con_low[ind_sbt] = +deg_cff[ind_sbt] - 0.5 * rep_qtz_tbl[ind_sbt]
    con_low[ind_sbf] = -deg_cff[ind_sbf] - 0.5 * rep_qtz_tbl[ind_sbf]

    # sign retrieval via PhaseMax
    rec_img = PhaseMax(anc_vec, con_upp, con_low, 100, 0.5)

    # sign retrieval via WirtingerFlow
    #rec_img = WirtingerFlow(anc_vec, con_upp, con_low, 150)

    anc_vec = np.round(anc_vec)
    anc_vec = np.maximum(anc_vec, 0)
    anc_vec = np.minimum(anc_vec, 255)
    anc_vec = np.array(anc_vec, np.uint8)
    skimage.io.imsave('anc.png', anc_vec)

    rec_img = np.round(rec_img)
    rec_img = np.maximum(rec_img, 0)
    rec_img = np.minimum(rec_img, 255)
    rec_img = np.array(rec_img, np.uint8)
    skimage.io.imsave('rec.png', rec_img)

    # PSNR
    print("{:4.2f}".format(skimage.metrics.peak_signal_noise_ratio(org_img, anc_vec)))
    print("{:4.2f}".format(skimage.metrics.peak_signal_noise_ratio(org_img, rec_img)))
