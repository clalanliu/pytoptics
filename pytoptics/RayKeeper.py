
import numpy as np
import pyvista as pv
import torch
from .Global import *
torch.set_default_tensor_type(torchPrecision)
device = torchdevice

class raykeeper():
    """raykeeper.
    """


    def __init__(self, System):
        """__init__.

        Parameters
        ----------
        System :
            System
        """
        self.SYSTEM = System
        self.clean()

    def valid(self):
        """valid.
        """
        z = np.argwhere((self.vld == 1))
        return z

    def push(self):
        """push.
        """
        self.nelements = self.SYSTEM.n
        if (self.SYSTEM.val == 0):
            self.invalid_vld = np.append(self.vld, 0)
            self.invalid_SURFACE.append(torch.asarray(self.SYSTEM.SURFACE))
            self.invalid_NAME.append(torch.asarray(self.SYSTEM.NAME))
            self.invalid_GLASS.append(torch.asarray(self.SYSTEM.GLASS))
            self.invalid_S_XYZ.append(torch.asarray(self.SYSTEM.S_XYZ))
            self.invalid_T_XYZ.append(torch.asarray(self.SYSTEM.T_XYZ))
            self.invalid_XYZ.append(torch.asarray(self.SYSTEM.XYZ))
            self.invalid_OST_XYZ.append(torch.asarray(self.SYSTEM.OST_XYZ))
            self.invalid_S_LMN.append(torch.asarray(self.SYSTEM.S_LMN))
            self.invalid_LMN.append(torch.asarray(self.SYSTEM.LMN))
            self.invalid_R_LMN.append(torch.asarray(self.SYSTEM.R_LMN))
            self.invalid_N0.append(torch.asarray(self.SYSTEM.N0))
            self.invalid_N1.append(torch.asarray(self.SYSTEM.N1))
            self.invalid_WAV.append(torch.asarray(self.SYSTEM.WAV))
            self.invalid_G_LMN.append(torch.asarray(self.SYSTEM.G_LMN))
            self.invalid_ORDER.append(torch.asarray(self.SYSTEM.ORDER))
            self.invalid_GRATING.append(torch.asarray(self.SYSTEM.GRATING))
            self.invalid_DISTANCE.append(torch.asarray(self.SYSTEM.DISTANCE))
            self.invalid_OP.append(torch.asarray(self.SYSTEM.OP))
            self.invalid_TOP_S.append(torch.asarray(self.SYSTEM.TOP_S))
            self.invalid_TOP.append(torch.asarray(self.SYSTEM.TOP))
            self.invalid_ALPHA.append(torch.asarray(self.SYSTEM.ALPHA))
            self.invalid_BULK_TRANS.append(torch.asarray(self.SYSTEM.BULK_TRANS))
            self.invalid_RP.append(torch.asarray(self.SYSTEM.RP))
            self.invalid_RS.append(torch.asarray(self.SYSTEM.RS))
            self.invalid_TP.append(torch.asarray(self.SYSTEM.TP))
            self.invalid_TS.append(torch.asarray(self.SYSTEM.TS))
            self.invalid_TTBE.append(torch.asarray(self.SYSTEM.TTBE))
            self.invalid_TT.append(torch.asarray(self.SYSTEM.TT))
        else:
            self.vld = np.append(self.vld, 1)
            self.valid_vld = np.append(self.vld, 0)
            self.valid_SURFACE.append(torch.asarray(self.SYSTEM.SURFACE))
            self.valid_NAME.append(np.asarray(self.SYSTEM.NAME))
            self.valid_GLASS.append(np.asarray(self.SYSTEM.GLASS))
            self.valid_S_XYZ.append(torch.vstack(self.SYSTEM.S_XYZ))
            self.valid_T_XYZ.append(torch.vstack(self.SYSTEM.T_XYZ))
            self.valid_XYZ.append(torch.vstack(self.SYSTEM.XYZ))
            self.valid_OST_XYZ.append(torch.vstack(self.SYSTEM.OST_XYZ))
            self.valid_S_LMN.append(torch.vstack(self.SYSTEM.S_LMN))
            self.valid_LMN.append(torch.vstack(self.SYSTEM.LMN))
            self.valid_R_LMN.append(torch.vstack(self.SYSTEM.R_LMN))
            self.valid_N0.append(torch.asarray(self.SYSTEM.N0))
            self.valid_N1.append(torch.asarray(self.SYSTEM.N1))
            self.valid_WAV.append(torch.asarray(self.SYSTEM.WAV))
            self.valid_G_LMN.append(torch.vstack(self.SYSTEM.G_LMN))
            self.valid_ORDER.append(torch.asarray(self.SYSTEM.ORDER))
            self.valid_GRATING.append(torch.asarray(self.SYSTEM.GRATING))
            self.valid_DISTANCE.append(torch.asarray(self.SYSTEM.DISTANCE))
            self.valid_OP.append(torch.asarray(self.SYSTEM.OP))
            self.valid_TOP_S.append(torch.asarray(self.SYSTEM.TOP_S))
            self.valid_TOP.append(torch.asarray(self.SYSTEM.TOP))
            self.valid_ALPHA.append(torch.asarray(self.SYSTEM.ALPHA))
            self.valid_BULK_TRANS.append(torch.asarray(self.SYSTEM.BULK_TRANS))
            self.valid_RP.append(torch.asarray(self.SYSTEM.RP))
            self.valid_RS.append(torch.asarray(self.SYSTEM.RS))
            self.valid_TP.append(torch.asarray(self.SYSTEM.TP))
            self.valid_TS.append(torch.asarray(self.SYSTEM.TS))
            self.valid_TTBE.append(torch.asarray(self.SYSTEM.TTBE))
            self.valid_TT.append(torch.asarray(self.SYSTEM.TT))
        self.nrays = (self.nrays + 1)


        self.RayWave.append(self.SYSTEM.Wave)
        self.CC.append(self.SYSTEM.ray_SurfHits)



        self.SURFACE.append(torch.asarray(self.SYSTEM.SURFACE))
        self.NAME.append(np.asarray(self.SYSTEM.NAME))
        self.GLASS.append(np.asarray(self.SYSTEM.GLASS))
        self.S_XYZ.append(torch.vstack(self.SYSTEM.S_XYZ))
        self.T_XYZ.append(torch.vstack(self.SYSTEM.T_XYZ))
        self.XYZ.append(torch.vstack(self.SYSTEM.XYZ))
        self.OST_XYZ.append(torch.vstack(self.SYSTEM.OST_XYZ))
        self.S_LMN.append(torch.vstack(self.SYSTEM.S_LMN))
        self.LMN.append(torch.vstack(self.SYSTEM.LMN))
        self.R_LMN.append(torch.vstack(self.SYSTEM.R_LMN))
        self.N0.append(torch.asarray(self.SYSTEM.N0))
        self.N1.append(torch.asarray(self.SYSTEM.N1))
        self.WAV.append(torch.asarray(self.SYSTEM.WAV))
        self.G_LMN.append(torch.vstack(self.SYSTEM.G_LMN))
        self.ORDER.append(torch.asarray(self.SYSTEM.ORDER))
        self.GRATING.append(torch.asarray(self.SYSTEM.GRATING))
        self.DISTANCE.append(torch.asarray(self.SYSTEM.DISTANCE))
        self.OP.append(torch.asarray(self.SYSTEM.OP))
        self.TOP_S.append(torch.asarray(self.SYSTEM.TOP_S))
        self.TOP.append(torch.asarray(self.SYSTEM.TOP))
        self.ALPHA.append(torch.asarray(self.SYSTEM.ALPHA))
        self.BULK_TRANS.append(torch.asarray(self.SYSTEM.BULK_TRANS))
        self.RP.append(torch.asarray(self.SYSTEM.RP))
        self.RS.append(torch.asarray(self.SYSTEM.RS))
        self.TP.append(torch.asarray(self.SYSTEM.TP))
        self.TS.append(torch.asarray(self.SYSTEM.TS))
        self.TTBE.append(torch.asarray(self.SYSTEM.TTBE))
        self.TT.append(torch.asarray(self.SYSTEM.TT))

    def clean(self):
        """clean.
        """
        self.vld = torch.asarray([])
        self.nrays = 0
        self.RayWave = []
        self.CC =[]
        self.SURFACE = []
        self.NAME = []
        self.GLASS = []
        self.S_XYZ = []
        self.T_XYZ = []
        self.XYZ = []
        self.OST_XYZ = []
        self.S_LMN = []
        self.LMN = []
        self.R_LMN = []
        self.N0 = []
        self.N1 = []
        self.WAV = []
        self.G_LMN = []
        self.ORDER = []
        self.GRATING = []
        self.DISTANCE = []
        self.OP = []
        self.TOP_S = []
        self.TOP = []
        self.ALPHA = []
        self.BULK_TRANS = []
        self.RP = []
        self.RS = []
        self.TP = []
        self.TS = []
        self.TTBE = []
        self.TT = []
        self.valid_RayWave = []
        self.valid_CCC = pv.MultiBlock()
        self.valid_SURFACE = []
        self.valid_NAME = []
        self.valid_GLASS = []
        self.valid_S_XYZ = []
        self.valid_T_XYZ = []
        self.valid_XYZ = []
        self.valid_OST_XYZ = []
        self.valid_S_LMN = []
        self.valid_LMN = []
        self.valid_R_LMN = []
        self.valid_N0 = []
        self.valid_N1 = []
        self.valid_WAV = []
        self.valid_G_LMN = []
        self.valid_ORDER = []
        self.valid_GRATING = []
        self.valid_DISTANCE = []
        self.valid_OP = []
        self.valid_TOP_S = []
        self.valid_TOP = []
        self.valid_ALPHA = []
        self.valid_BULK_TRANS = []
        self.valid_RP = []
        self.valid_RS = []
        self.valid_TP = []
        self.valid_TS = []
        self.valid_TTBE = []
        self.valid_TT = []
        self.invalid_RayWave = []
        self.invalid_CCC = pv.MultiBlock()
        self.invalid_SURFACE = []
        self.invalid_NAME = []
        self.invalid_GLASS = []
        self.invalid_S_XYZ = []
        self.invalid_T_XYZ = []
        self.invalid_XYZ = []
        self.invalid_OST_XYZ = []
        self.invalid_S_LMN = []
        self.invalid_LMN = []
        self.invalid_R_LMN = []
        self.invalid_N0 = []
        self.invalid_N1 = []
        self.invalid_WAV = []
        self.invalid_G_LMN = []
        self.invalid_ORDER = []
        self.invalid_GRATING = []
        self.invalid_DISTANCE = []
        self.invalid_OP = []
        self.invalid_TOP_S = []
        self.invalid_TOP = []
        self.invalid_ALPHA = []
        self.invalid_BULK_TRANS = []
        self.invalid_RP = []
        self.invalid_RS = []
        self.invalid_TP = []
        self.invalid_TS = []
        self.invalid_TTBE = []
        self.invalid_TT = []

    def pick(self, N_ELEMENT=(- 1)):
        """pick.

        Parameters
        ----------
        N_ELEMENT :
            N_ELEMENT
        """

        gls = self.SYSTEM.SDT[N_ELEMENT].Glass
        if gls == "NULL":
            print("NULL surface has been chosen, the return values correspond to those of the previous surface")

        self.numsup = (self.nelements - 1)
        self.xyz = self.valid_XYZ
        self.lmn = self.valid_LMN
        self.s = self.valid_SURFACE
        if ((N_ELEMENT < 0) or (N_ELEMENT > self.numsup)):
            N_ELEMENT = self.numsup
        else:
            N_ELEMENT = N_ELEMENT
        AA = []
        BB = []
        for k in self.s:
            aa = (k == N_ELEMENT).nonzero()
            aa = torch.squeeze(aa)
            AA.append(aa)
            BB.append(torch.numel(aa))
        #AA = torch.vstack(AA).detach().cpu().numpy()
        BB = np.array(BB)
        if (N_ELEMENT != 0):
            BB = np.argwhere((BB == 1))
        else:
            BB = np.argwhere((BB == 0))
        X = []
        Y = []
        Z = []
        L = []
        M = []
        N = []
        for c in BB:
            for d in c:
                ray0 = self.xyz[d]

                [x1, y1, z1] = ray0[N_ELEMENT]
                X.append(x1)
                Y.append(y1)
                Z.append(z1)
                ray1 = self.lmn[d]
                if (N_ELEMENT != 0):
                    el = (N_ELEMENT - 1)
                else:
                    el = 0
                [l1, m1, n1] = ray1[el]
                L.append(l1)
                M.append(m1)
                N.append(n1)
        return (torch.asarray(X), torch.asarray(Y), torch.asarray(Z), torch.asarray(L), torch.asarray(M), torch.asarray(N))


