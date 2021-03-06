-----------------------------------------------------------------------------

                    An Efficient Compression Method 
        for Sign Information of DCT Coefficients via Sign Retrieval            
 
-----------------------------------------------------------------------------

Written by  : Chihiro Tsutake
Affiliation : Nagoya University
E-mail      : ctsutake "at" nagoya-u.jp
Created     : Dec 2020

-----------------------------------------------------------------------------
    Contents
-----------------------------------------------------------------------------

sr.py       : Main algorithm file
0***.png    : Original image

-----------------------------------------------------------------------------
    Usage
-----------------------------------------------------------------------------

1) Change skimage.io.imread('0***.png') in main function.
2) Running `python3 sr.py' generates the following images.

    -- rec.png (reconstructed image by using our Fienup algorithm)
    -- rnd.png (reconstructed image by using randomized sign)
    -- jpg.png (JPEG)

-----------------------------------------------------------------------------
    Feedback
-----------------------------------------------------------------------------

If you have any questions, please contact me.
