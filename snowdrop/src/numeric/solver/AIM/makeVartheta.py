"""This function calculates the Vartheta matrix used in the AMA algorithm."""

# import numpy package
import numpy as np 


def makeVartheta(phi, F, psi, upsilon):

    # Ensure that psi and upsilon are in matrix form
    psi = np.matrix(psi)
    upsilon = np.matrix(upsilon)

    # Store the dimensions of phi, F, and psi
    phirows, phicols = phi.shape
    frows, fcols = F.shape
    psirows, psicols = psi.shape

    # Calculate the Kronecker Product and its dimensions
    krnprt = np.kron(upsilon.T, F)
    krnrows, krncols = krnprt.shape

    bigun = np.eye(krnrows)-krnprt
    productPhiPsi = phi * psi
    bigvec = productPhiPsi.T.reshape((phirows*psicols,1))
    
    resultProduct = bigun.I * bigvec
    varthetaTranspose = resultProduct.reshape((phirows,psicols))
    vartheta = varthetaTranspose.T

    return vartheta
