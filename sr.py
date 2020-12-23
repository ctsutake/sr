# package
import copy as cp
import pywt as pw
import numpy as np
import skimage.io
import skimage.metrics
import scipy.fftpack

# quality factor
QF = 50

# quadrature mirror filter
QMF = 'sym12'

# decomposition level
DLV = 1

# outer iteration
OUT_ITR = 3

# inner iteration
INN_ITR = 200

# regularization parameter
REG_PRM = 1.0

# penalty parameter
PEN_PRM = 0.01

# quntization table
QTZ_TBL = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99]])

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

def fienup(anc_vec, con_upp, con_low):

    # initial guess
    rec_vec = anc_vec

    for inn_itr in range(INN_ITR):

        # copy
        cpy_vec = cp.deepcopy(rec_vec)

        # thresholding
        rec_vec = pw.swt2(rec_vec, QMF, DLV, norm=True)

        for d in range(DLV):
            # LL
            rec_vec[d] = list(rec_vec[d])
            rec_vec[d][0] = pw.threshold(rec_vec[d][0], REG_PRM)

            # LH, HL, and HH
            rec_vec[d][1] = list(rec_vec[d][1])
            rec_vec[d][1][0] = pw.threshold(rec_vec[d][1][0], REG_PRM)
            rec_vec[d][1][1] = pw.threshold(rec_vec[d][1][1], REG_PRM)
            rec_vec[d][1][2] = pw.threshold(rec_vec[d][1][2], REG_PRM)

        rec_vec = pw.iswt2(rec_vec, QMF, DLV)

        # proximal mapping for inner product
        rec_vec = rec_vec + PEN_PRM * anc_vec

        # projection
        rec_vec = bdct2(rec_vec)
        rec_vec = np.minimum(rec_vec, con_upp)
        rec_vec = np.maximum(rec_vec, con_low)
        rec_vec = bidct2(rec_vec)

        # acceleration
        rec_vec = rec_vec + (inn_itr - 1) * (rec_vec - cpy_vec) / (inn_itr + 2)

    return rec_vec


if __name__ == '__main__':

    # original image
    org_img = skimage.io.imread('0323.png')

    # image size
    row = org_img.shape[0]
    col = org_img.shape[1]

    # make table
    if QF < 50:
        QTZ_TBL = np.round(((5000 / QF) * QTZ_TBL + 50) / 100)
    else:
        QTZ_TBL = np.round(((200 - 2 * QF) * QTZ_TBL + 50) / 100)

    # replicate quantization table
    rep_qtz_tbl = np.tile(QTZ_TBL, [row >> 3, col >> 3])

    # original coefficient
    org_cff = bdct2(org_img)

    # quantization index in JPEG
    qtz_ind_jpg = np.array(np.round(org_cff / rep_qtz_tbl), np.int64)

    # quantization index in our method
    qtz_ind_our = np.abs(qtz_ind_jpg)

    # inverse quantization
    deg_cff_jpg = np.array(qtz_ind_jpg, np.float) * rep_qtz_tbl
    deg_cff_our = np.array(qtz_ind_our, np.float) * rep_qtz_tbl

    # upper constraint
    con_upp = +np.array(deg_cff_our, np.float)

    # lower constraint
    con_low = -np.array(deg_cff_our, np.float)
    con_low[0:row:8, 0:col:8] = -con_low[0:row:8, 0:col:8]

    # anchor vector
    anc_vec = bidct2((con_upp + con_low) / 2)

    # sign retrieval via cascaded Fienup algorithm
    for out_itr in range(OUT_ITR):
        rec_img = fienup(anc_vec, con_upp, con_low)
        anc_vec = rec_img

    # original sign
    org_sgn = np.sign(qtz_ind_jpg)
    org_sgn[0:row:8, 0:col:8] = 0

    # recovered sign
    rec_sgn = np.sign(bdct2(rec_img))
    rec_sgn[np.where(rec_sgn == 0)] = 1
    rec_sgn[0:row:8, 0:col:8] = 0
    rec_sgn[np.where(qtz_ind_our == 0)] = 0

    # bit plane to be transmitted
    bit_pln = org_sgn * rec_sgn

    # probability of residual
    num_pos = np.count_nonzero(bit_pln == +1)
    num_neg = np.count_nonzero(bit_pln == -1)
    prb_pos = num_pos / (num_pos + num_neg)
    prb_neg = num_neg / (num_pos + num_neg)

    # residual bits per significant index
    bit_ind = -(prb_pos * np.log2(prb_pos) + prb_neg * np.log2(prb_neg))
    print("Entropy of residual was {0:6.4f} [bits].".format(bit_ind))

    # reconstructed image 
    rec_img = np.round(rec_img)
    rec_img = np.maximum(rec_img, 0)
    rec_img = np.minimum(rec_img, 255)
    rec_img = np.array(rec_img, np.uint8)
    skimage.io.imsave('rec.png', rec_img)
    
    # random image
    rec_img = np.sign(np.random.randn(row, col))
    rec_img = rec_img * deg_cff_jpg
    rec_img[0:row:8, 0:col:8] = deg_cff_jpg[0:row:8, 0:col:8]
    rec_img = bidct2(rec_img)
    rec_img = np.round(rec_img)
    rec_img = np.maximum(rec_img, 0)
    rec_img = np.minimum(rec_img, 255)
    rec_img = np.array(rec_img, np.uint8)
    skimage.io.imsave('rnd.png', rec_img)

    # JPEG
    rec_img = bidct2(deg_cff_jpg)
    rec_img = np.round(rec_img)
    rec_img = np.maximum(rec_img, 0)
    rec_img = np.minimum(rec_img, 255)
    rec_img = np.array(rec_img, np.uint8)
    skimage.io.imsave('jpg.png', rec_img)
