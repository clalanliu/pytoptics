
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
        self.valid_set = []
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
            self.valid_set = np.append(self.valid_set, 0)
            self.invalid_vld = np.append(self.vld, 0)
            
        else:
            self.valid_set = np.append(self.valid_set, 1)
            self.vld = np.append(self.vld, 1)
            self.valid_vld = np.append(self.vld, 0)
            
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
        #self.OP.append(np.asarray(self.SYSTEM.OP))
        #self.TOP_S.append(np.asarray(self.SYSTEM.TOP_S))
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
        self.valid_set = []
        self.vld = torch.asarray([])
        self.nrays = 0
        self.RayWave = []
        self.CC = []
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
        #self.OP = []
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
        self.xyz = [self.XYZ[i[0]] for i in np.argwhere(self.valid_set)]
        self.lmn = [self.LMN[i[0]] for i in np.argwhere(self.valid_set)]
        self.s = [self.SURFACE[i[0]] for i in np.argwhere(self.valid_set)]
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

        try:
            X_ = torch.stack(X)
            Y_ = torch.stack(Y)
            Z_ = torch.stack(Z)
            L_ = torch.stack(L)
            M_ = torch.stack(M)
            N_ = torch.stack(N)
            
            return (X_, Y_, Z_, L_,M_, N_)

        except:
            return None


