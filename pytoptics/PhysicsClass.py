
import numpy as np
from .Physics import *

import torch
from .Global import *
torch.set_default_tensor_type(torchPrecision)
device = torchdevice


class snell_refraction_vector_physics():
    """snell_refraction_vector_physics.
    """


    def __init__(self):
        """__init__.
        """
        pass

    def calculate(self, s1_norm, Nsurf_norm, n1, n2, empty1, empty2, empty3, empty4, Secuen):
    
        return calculate2(s1_norm, Nsurf_norm, n1, n2, empty1, empty2, empty3, empty4, Secuen)


# @jit(nopython=True)
def calculate2(s1_norm, Nsurf_norm, n1, n2, empty1, empty2, empty3, empty4, Secuen):
    """calculate.

    Parameters
    ----------
    s1_norm :
        s1_norm
    Nsurf_norm :
        Nsurf_norm
    n1 :
        n1
    n2 :
        n2
    empty1 :
        empty1
    empty2 :
        empty2
    empty3 :
        empty3
    empty4 :
        empty4
    Secuen :
        Secuen
r = rayo
r'=reflejado
S=normal superficie
    """


    Nv = Nsurf_norm
    Iv = s1_norm


    """ checking if the normal to the surface is in the
    direction of space where the ray is coming from and
    not where it is going """

    cos=torch.dot(Nv, Iv)

    if (cos < (- 1.0)):
        ang = 180.0
    else:
        ang = torch.rad2deg(torch.arccos(cos))

    if (ang <= 90.0):
        Nv = - Nv

    """----------------------------------------------"""


    Nsurf_Cros_s1 = torch.cross(Nv, Iv)
    SIGN = 1.0

    if (n2 == - 1.0):
        n2 = - n1
        SIGN = -1.0

    NN = (n1 / n2)
    d22=torch.dot(Nsurf_Cros_s1, Nsurf_Cros_s1)
    R = (NN * NN) * d22

    if (Secuen == 1.0):
        R = 2.0

    if (R > 1.0):
        n2 = - n1
        NN = n1 / n2
        # R = (NN * NN) * d22
        SIGN = - 1.0

    c1 = torch.dot(Nv,Iv)
    if c1 < 0.0 :
        c1 = torch.dot(-Nv,Iv)
    IP = ((NN**2)*(1-(c1**2.0)))

    c2 = torch.sqrt(1.0-IP)
    T = NN * Iv + ((( NN * c1 ) - c2)) * Nv

    return T, np.abs(n2), SIGN


# @jit(nopython=True)
# def calculate2(s1_norm, Nsurf_norm, n1, n2, empty1, empty2, empty3, empty4, Secuen):
#     """calculate.

#     Parameters
#     ----------
#     s1_norm :
#         s1_norm
#     Nsurf_norm :
#         Nsurf_norm
#     n1 :
#         n1
#     n2 :
#         n2
#     empty1 :
#         empty1
#     empty2 :
#         empty2
#     empty3 :
#         empty3
#     empty4 :
#         empty4
#     Secuen :
#         Secuen
#     """


#     # Nsurf_norm = 1. * np.asarray(Nsurf_norm1)
#     # s1_norm = 1. * np.asarray(s1_norm1)


#     N1 = [Nsurf_norm[0], Nsurf_norm[1], Nsurf_norm[2]]
#     S1 = [s1_norm[0], s1_norm[1], s1_norm[2]]

#     cos=(N1[0]*S1[0]) + (N1[1]*S1[1]) + (N1[2]*S1[2])

#     # cos = np.dot(N1, S1)

#     if (cos < (- 1.0)):
#         ang = 180.0
#     else:
#         ang = np.rad2deg(np.arccos(cos))

#     if (ang <= 90.0):
#         Nsurf_norm = (- Nsurf_norm)

#     Nsurf_Cros_s1 = np.cross(Nsurf_norm, s1_norm)
#     SIGN = 1.0

#     if (n2 == (- 1.0)):
#         n2 = (- n1)
#         SIGN = (- 1.0)

#     NN = (n1 / n2)
#     d22=np.dot(Nsurf_Cros_s1, Nsurf_Cros_s1)
#     R = ((NN * NN) * d22)

#     if (Secuen == 1.0):
#         R = 2.0

#     if (R > 1.0):
#         n2 = (- n1)
#         NN = (n1 / n2)
#         R = ((NN * NN) * d22)
#         SIGN = (- 1.0)

#     s2 = ((NN * np.cross(Nsurf_norm, np.cross((- Nsurf_norm), s1_norm))) - (Nsurf_norm * np.sqrt((1.0 - ((NN * NN) * d22)))))


#     # cos = np.dot([0.0, 0.0, 1.0], [s2[0], s2[1], s2[2]])
#     # cos = s2[2]



#     # ang = np.rad2deg(np.arccos(cos))


#     return s2, np.abs(n2), SIGN





class paraxial_exact_physics():
    """paraxial_exact_physics.
    """


    def __init__(self):
        """__init__.
        """
        pass

    def calculate(self, s1_norm, Nsurf_norm, n1, n2, empty1, empty2, empty3, empty4, empty5):
        """calculate.

        Parameters
        ----------
        s1_norm :
            s1_norm
        Nsurf_norm :
            Nsurf_norm
        n1 :
            n1
        n2 :
            n2
        empty1 :
            empty1
        empty2 :
            empty2
        empty3 :
            empty3
        empty4 :
            empty4
        empty5 :
            empty5
        """
        return (Nsurf_norm, 1, 1)





class diffraction_grating_physics():
    """diffraction_grating_physics.
    """


    def __init__(self):
        """__init__.
        """
        pass

    def calculate(self, S, R, N, Np, P, Ord, d, W, Secuent):
        Ang=np.rad2deg(np.arccos( np.dot(R,S)))
        # print(S, R, Ord, Ang)


        if np.abs(Ang) < 90:
            R = -R


        """calculate.

        Parameters
        ----------
        S :
            S
        R :
            R
        N :
            N
        Np :
            Np
        D :
            D
        Ord :
            Ord
        d :
            d
        W :
            W
        Secuent :
            Secuent
        """


        D = np.cross(R, P)



        lamb = W
        RefRef = ((- 1.) * np.sign(Np))
        SIGN = 1.

        # print(RefRef, Np)
        if (Np == (- 1.)):
            Np = np.abs(N)

##########################################
        mu = (N / Np)
        T = ((Ord * lamb) / (Np * d))
##########################################

        V = (mu * np.dot(R,S))
        W = ((((mu ** 2.0) - 1.) + (T ** 2.0)) - (((2.0 * mu) * T) * np.dot(D,S)))




        Q1 = ( np.sqrt((V**2 - W))) - V
        Q2 = (-np.sqrt((V**2 - W))) - V

        if RefRef == 1:
            if Q1 > Q2:
                Q = Q1
            else:
                Q = Q2

        if RefRef == -1:
            if Q1 < Q2:
                Q = Q1
            else:
                Q = Q2


        S = np.asarray(S)
        Sp = (((mu * S) - (T * D)) + (Q * R))

        SIGN = SIGN*-1*RefRef
        return (Sp, np.abs(Np), SIGN)

