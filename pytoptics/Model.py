from .SurfClass import *
from .RayKeeper import *
from .SurfTools import surface_tools as SUT
from .Prerequisites3D import *
from .Physics import *
from .HitOnSurf import *
from .InterNormalCalc import *
from .MathUtil import *
from .Display import *
from .ConstraintManager import ConstraintManager

from enum import Enum
import KrakenOS as Kos
import matplotlib.pyplot as plt
import torch
from .Global import *
torch.set_default_tensor_type(torchPrecision)
device = torchdevice

class FieldType(Enum):
    """
    ANGLE

    OBJECT_HEIGHT
    """
    ANGLE = 1
    OBJECT_HEIGHT = 2

class ApertureType(Enum):
    """
    ENTRANCE_PUPIL_DIAMETER

    NUMERICAL_APERTURE_OBJECT
    """
    ENTRANCE_PUPIL_DIAMETER = 1
    NUMERICAL_APERTURE_OBJECT = 2


class OpticalSystem(torch.nn.Module):
    def __init__(self, surface_num):
        super().__init__()
        self.surfaces = torch.nn.ModuleList()
        self.constraintManager = None
        for i in range(surface_num):
            self.surfaces.append(surf())


    def Initialize(self):
        config_ = Kos.Setup()
        self.__init_system(self.surfaces, config_)

    def SetAperture(self, aperture_type: ApertureType, aperture_value: float):
        """Setting Aperture of the system

        aperture_type: ApertureType
        aperture_value: value
        """
        self.aperture_type = aperture_type
        self.aperture_value = aperture_value

    def SetFields(self, field_type:FieldType, fields:list, field_sample_num:float = 3) -> None:
        """Setting fields of the system

        field_type: FieldType
        fields: [[x, y, weight], ...]
        field_sample_num: Number of samples in a direction
        """
        self.field_type = field_type 
        self.fields = fields
        self.field_sample_num = field_sample_num

    def SetWavelength(self, wavelengths: list):
        """Setting wavelength of the system

        wavelengths: list of used wavelength
        """
        self.wavelengths = wavelengths

    def AddConstraint(self, constraints:list):
        self.constraintManager = ConstraintManager(constraints)

    def Optimize_GradientDescent(self, lr=10, MNC=5, MXC=99999, IMP=0.0001, loss_criteria=0, print_variable = False):
        """Optimize system using gradient descent (SGD optimizer)

        lr: learning rate
        MNC: minimal number of cycles
        MXC: maximal number of cycles
        IMP: fractional improvement. If the improvements less than this amount five times continuously, optimization will be stopped
        """
        self.optimizer = torch.optim.Adagrad(self.parameters(), lr=lr)
        self.loss_history = [np.finfo(0.0).max]
        # Minimum cycles
        for epoch in range(MNC):
            if print_variable: self.PrintVariables()
            self.optimizer.zero_grad()
            self.Trace()
            loss = self.calc_constraint_loss() + self.calc_field_loss()
            self.loss_history.append(loss.detach().cpu().numpy())
            loss.backward(retain_graph=True)
            print("Epoch {}: Loss = {}".format(epoch+1, self.loss_history[-1]))
            self.optimizer.step()

            if epoch < MNC - 1:
                self.update_surfaces()

        # Until maximal cycles
        counter = 0
        for epoch in range(MNC, MXC):
            if print_variable: self.PrintVariables()
            self.optimizer.zero_grad()
            self.Trace()
            loss = self.calc_constraint_loss() + self.calc_field_loss()
            self.loss_history.append(loss.detach().cpu().numpy())
            # loss_criteria check
            if self.loss_history[-1] < loss_criteria:
                break

            # IMP Check
            if np.abs((self.loss_history[-1] - self.loss_history[-2])/self.loss_history[-2]) < IMP:
                counter += 1
            else:
                counter = 0
            if counter >= 5:
                break

            loss.backward(retain_graph=True)
            print("Epoch {}: Loss = {}".format(epoch+1, self.loss_history[-1]))
            self.optimizer.step()

            if epoch < MXC - 1:
                self.update_surfaces()

        self.loss_history = np.array(self.loss_history[1:])
        return self.loss_history

    def calc_field_loss(self):
        loss = 0
        for f_i in range(len(self.fields)):       
            X = torch.stack(self.output_X[f_i])
            Y = torch.stack(self.output_X[f_i])
            loss = loss + torch.std(X, unbiased=False) + torch.std(Y, unbiased=False)
        
        return loss

    def calc_constraint_loss(self):
        if self.constraintManager == None:
            return torch.tensor(0).to(device)
        if len(self.constraintManager.losses)==0:
            return torch.tensor(0).to(device)
        
        loss = 0
        for i in range(len(self.constraintManager.losses)):
            loss = loss + self.constraintManager.losses[i](self.constraintManager.constraints[i][0])
        
        return loss

    def ShowSpotDiagram(self):
        self.spot_image_X = []
        self.spot_image_Y = []
        for i in range(len(self.output_X)):
            for j in range(len(self.output_X[0])):
                self.spot_image_X.append(self.output_X[i][j].detach().cpu().numpy())
                self.spot_image_Y.append(self.output_Y[i][j].detach().cpu().numpy())
        
        self.spot_image_X = np.hstack(self.spot_image_X)
        self.spot_image_Y = np.hstack(self.spot_image_Y)

        min_x = np.min(self.spot_image_X)
        max_x = np.max(self.spot_image_X)
        min_y = np.min(self.spot_image_Y)
        max_y = np.max(self.spot_image_Y)

        xlim = np.max([np.abs(min_x), np.abs(max_x)])
        ylim = np.max([np.abs(min_y), np.abs(max_y)])
        lim = np.max([xlim, ylim])

        plt.figure()
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.plot(self.spot_image_X, self.spot_image_Y, 'x')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Spot Diagram')
        #plt.axis('square')
        plt.show(block = False)

    def ShowModel2D(self):
        display2d(self, self.rayskeepers, 0)

    def ShowModel3D(self):
        display3d(self, self.rayskeepers[0][0], 0)

    def ShowOptimizeLoss(self):
        ymin = np.min(self.loss_history)
        ymax = np.quantile(self.loss_history, 0.95)
        ymin = ymin - (ymax-ymin)*0.05
        ymax = ymax + (ymax-ymin)*0.05
        plt.figure()
        plt.plot(self.loss_history)
        plt.ylim([ymin, ymax])
        plt.title("Loss Optimization")
        plt.show(block = False)

    # ______________________________________#
    def PrintVariables(self):
        print("===== Optimized Variables =====")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data, param.grad)
        print("===============================")

    def update_surfaces(self):
        self.SDT = self.surfaces
        self.update = False
        self.S_Matrix = []
        self.N_Matrix = []
        self.SystemMatrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).to(device)
        (self.a, self.b, self.c, self.d, self.EFFL, self.PPA, self.PPP) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.n = len(self.SDT)
        for ii in range(0, self.n):
            self.SDT[ii].SaveSetup()
        self.SuTo = SUT(self.SDT)
        self.Object_Num = torch.arange(0, self.n, 1)
        self.Targ_Surf = self.n
        self.SuTo.Surface_Flattener = 0
        self.Disable_Inner = 1
        self.ExtraDiameter = 0
        self.SuTo.ErrSurfCase = 0
        self.__SurFuncSuscrip()
        self.Pr3D = Prerequisites(self.SDT, self.SuTo)
        self.__PrerequisitesGlass()
        self.Pr3D.Prerequisites3SMath()
        self.Pr3D.Prerequisites3DSolids()
        self.PreWave = (- 1e-09)
        self.AAA = self.Pr3D.AAA
        self.BBB = self.Pr3D.BBB
        self.DDD = self.Pr3D.DDD
        self.EEE = self.Pr3D.EEE
        self.GlassOnSide = self.Pr3D.GlassOnSide
        self.side_number = self.Pr3D.side_number
        self.TypeTotal = self.Pr3D.TypeTotal
        self.TRANS_1A = self.Pr3D.TRANS_1A
        self.TRANS_2A = self.Pr3D.TRANS_2A
        self.HS = Hit_Solver(self.SDT)
        self.INORM = InterNormalCalc(self.SDT, self.TypeTotal, self.Pr3D, self.HS)
        self.INORM.Disable_Inner = 1
        self.Pr3D.Disable_Inner = 1
        (self.c_p, self.n_p, self.d_p) = (0, 0, 0)
        self.tt=1.
        
        self.rayskeepers = [[raykeeper(self) for _ in range(len(self.wavelengths))] for _ in range(len(self.fields))] 


    def Trace(self):
        if self.field_type == FieldType.ANGLE:
            #self.trace_parallel_light_with_Y_angle()
            self.output_X = []
            self.output_Y = []
            self.output_Z = []
            self.output_L = []
            self.output_M = []
            self.output_N = []
            for f_i in range(len(self.fields)):
                self.output_X.append([])
                self.output_Y.append([])
                self.output_Z.append([])
                self.output_L.append([])
                self.output_M.append([])
                self.output_N.append([])

                X = np.linspace(-self.aperture_value/2, self.aperture_value/2, self.field_sample_num)
                Y = np.linspace(-self.aperture_value/2, self.aperture_value/2, self.field_sample_num)
                for x in X:
                    for y in Y:
                        r = np.sqrt((x * x) + (y * y))
                        if r <= self.aperture_value/2:
                            pSource_0 = [x, y, 0.0]
                            dCos = xy_to_direction_cosine(self.fields[f_i][0], self.fields[f_i][1])
                            
                            for w_i in range(len(self.wavelengths)):
                                self.RayTrace(pSource_0, dCos, self.wavelengths[w_i])
                                self.rayskeepers[f_i][w_i].push()

                for w_i in range(len(self.wavelengths)):            
                    X, Y, Z, L, M, N = self.rayskeepers[f_i][w_i].pick(-1)
                    self.output_X[-1].append(X)
                    self.output_Y[-1].append(Y)
                    self.output_Z[-1].append(Z)
                    self.output_L[-1].append(L)
                    self.output_M[-1].append(M)
                    self.output_N[-1].append(N)
            

        
    def __init_system(self, SurfData, KN_Setup):
        """__init__.

        Parameters
        ----------
        SurfData :
            SurfData
        KN_Setup :
            KN_Setup
        """
        self.ExectTime=[]
        self.SDT = SurfData
        self.SETUP = KN_Setup
        self.update_surfaces()

    def __SurFuncSuscrip(self):
        """__SurFuncSuscrip.
        """
        for i in range(0, self.n):
            self.SDT[i].build_surface_function()

    def __PrerequisitesGlass(self):
        """__PrerequisitesGlass.
        """
        self.Glass = []
        self.GlobGlass = []
        for i in range(0, self.n):
            if (type(self.SDT[i].Glass) == type(1.0)):
                self.SDT[i].Glass = ('AIR_' + str(self.SDT[i].Glass))
                # print(self.SDT[i].Glass)
            self.Glass.append(self.SDT[i].Glass.replace(' ', ''))
            self.GlobGlass.append(self.SDT[i].Glass.replace(' ', ''))
            if ((self.GlobGlass[i] == 'NULL') or (self.GlobGlass[i] == 'ABSORB')):
                self.GlobGlass[i] = self.GlobGlass[(i - 1)]

    def __WavePrecalc(self):
        """__WavePrecalc.: Calculate wave-dependent parameteres like refraction index 
        """
        if (self.Wave != self.PreWave):
            self.N_Prec = []
            self.AlphaPrecal = []
            self.PreWave = self.Wave
            for i in range(0, self.n):
                (NP, AP) = n_wave_dispersion(self.SETUP, self.GlobGlass[i], self.Wave)
                self.N_Prec.append(NP)
                self.AlphaPrecal.append(AP)
    
    def __CollectDataInit(self):
        """__CollectDataInit.
        """
        self.val = 1
        self.SURFACE = []
        self.NAME = []
        self.GLASS = []
        self.S_XYZ = []
        self.T_XYZ = []
        self.XYZ = []
        self.XYZ.append([0, 0, 0])
        self.OST_XYZ = []
        self.OST_LMN = []
        self.S_LMN = []
        self.LMN = []
        self.R_LMN = []
        self.N0 = []
        self.N1 = []
        self.WAV = 1.0
        self.G_LMN = []
        self.ORDER = []
        self.GRATING = []
        self.DISTANCE = []
        #self.OP = []
        #self.TOP_S = []
        self.TOP = 0
        self.ALPHA = [0.0]
        self.BULK_TRANS = []
        self.RP = []
        self.RS = []
        self.TP = []
        self.TS = []
        self.TTBE = []
        self.TT = 1.0
        return None

    def __CollectData(self, ValToSav):
        """__CollectData.

        Parameters
        ----------
        ValToSav :
            ValToSav
        """
        [Glass, alpha, RayOrig, pTarget, HitObjSpace, LMNObjSpace, SurfNorm, ImpVec, ResVec, PrevN, CurrN, WaveLength, D, Ord, GrSpa, Name, j, RayTraceType] = ValToSav



        self.SURFACE.append(j)
        self.NAME.append(Name)
        self.GLASS.append(Glass)
        self.S_XYZ.append(RayOrig)
        self.T_XYZ.append(pTarget)
        self.XYZ[0] = self.S_XYZ[0]   
        self.XYZ.append(pTarget)
        self.OST_XYZ.append(HitObjSpace)
        self.OST_LMN.append(LMNObjSpace)

        p = RayOrig - pTarget
        dist = torch.linalg.norm(p)
        self.DISTANCE.append(dist)
        #self.OP.append((dist * PrevN) if PrevN != [] else [])

        self.TOP = (self.TOP + (dist * PrevN)) if PrevN != [] else []
        #self.TOP_S.append(self.TOP)
        self.ALPHA.append(alpha)
        self.S_LMN.append(SurfNorm)
        self.LMN.append(ImpVec)
        self.R_LMN.append(ResVec)
        self.N0.append(PrevN)
        self.N1.append(CurrN)
        self.WAV = WaveLength
        self.G_LMN.append(D)
        self.ORDER.append(Ord)
        self.GRATING.append(GrSpa)
        if (self.val == 1):
            (Rp, Rs, Tp, Ts) = FresnelEnergy(Glass, PrevN, CurrN, ImpVec, SurfNorm, ResVec, self.SETUP, self.Wave)
        else:
            (Rp, Rs, Tp, Ts) = (0, 0, 0, 0)
        self.RP.append(Rp)
        self.RS.append(Rs)
        self.TP.append(Tp)
        self.TS.append(Ts)
        if ((RayTraceType == 0) or (RayTraceType == 1)):
            if (Glass == 'MIRROR'):
                self.tt = ((1.0 * (Rp + Rs)) / 2.0)
                self.BULK_TRANS.append(self.tt)
            if (Glass != 'MIRROR'):
                IT = torch.exp(((- self.ALPHA[-2]) * dist))

                self.BULK_TRANS.append(IT)
                self.tt = (Tp + Ts) / 2.0
        else:
            self.tt = 1.0


        # si self.tt está vacio entonces es cero #
        if not self.tt:
            self.tt=0

        self.TTBE.append(self.tt*self.BULK_TRANS[-1])
        self.TT = (self.TT * self.tt*self.BULK_TRANS[-1])

        return None


    def RayTrace(self, pS, dC, WaveLength):
        """Trace.

        Parameters
        ----------
        pS :
            pS
        dC :
            dC
        WaveLength :
            WaveLength
        """
        self.__CollectDataInit()
        ResVec = torch.tensor(dC).to(device)
        RayOrig = torch.tensor(pS).to(device)
        
        self.Wave = WaveLength
        self.__WavePrecalc()
        j = 0
        Glass = self.GlobGlass[j]
        (PrevN, alpha) = (self.N_Prec[j], self.AlphaPrecal[j])
        j = 1
        SIGN = torch.ones_like(ResVec).to(device)
        
        RAY = []
        RAY.append(RayOrig)
        while True:
            if (j == self.Targ_Surf):
                break
            j_gg = j
            Glass = self.GlobGlass[j_gg]

            if (self.Glass[j] != 'NULL'):
                Proto_pTarget = (RayOrig + ((ResVec * 999999999.9) * SIGN))

                Output = self.INORM.InterNormal(RayOrig, Proto_pTarget, j, j)

                (SurfHit, SurfNorm, pTarget, GooveVect, HitObjSpace, LMNObjSpace, j) = Output
                
                if (SurfHit == 0):
                    break

                ImpVec = ResVec
                (CurrN, alpha) = (self.N_Prec[j_gg], self.AlphaPrecal[j_gg])
                S = ImpVec
                R = SurfNorm
                N = PrevN
                Np = CurrN
                D = GooveVect
                Ord = self.SDT[j].Diff_Ord
                GrSpa = self.SDT[j].Grating_D
                Secuent = 0

                (ResVec, CurrN, sign) = self.SDT[j].PHYSICS.calculate(S, R, N, Np, D, Ord, GrSpa, self.Wave, Secuent)

                SIGN = (SIGN * sign)
                Name = self.SDT[j].Name
                RayTraceType = 0
                ValToSav = [Glass, alpha, RayOrig, pTarget, HitObjSpace, LMNObjSpace, SurfNorm, ImpVec, ResVec, PrevN, CurrN, WaveLength, D, Ord, GrSpa, Name, j, RayTraceType]

                self.__CollectData(ValToSav)
                PrevN = CurrN
                RayOrig = pTarget
                RAY.append(RayOrig.clone())

            elif self.Glass[j] == 'NULL':

                ValToSav = [Glass, alpha, RayOrig, pTarget, HitObjSpace, LMNObjSpace, SurfNorm, ImpVec, ResVec, PrevN, CurrN, WaveLength, D, Ord, GrSpa, Name, j, RayTraceType]
                self.__CollectData(ValToSav)

            if self.Glass[j] == 'ABSORB':
                break

            j = (j + 1)
        
        if (len(self.GLASS) == 0):
            self.__CollectDataInit()
            self.val = 0
            self.__EmptyCollect(RayOrig, ResVec, WaveLength, j)

        self.ray_SurfHits = torch.vstack(RAY)
        AT = torch.transpose(self.ray_SurfHits, 0, 1)
        
        self.Hit_x = AT[0]
        self.Hit_y = AT[1]
        self.Hit_z = AT[2]


        self.ExectTime=[]




    def __EmptyCollect(self, pS, dC, WaveLength, j):
        """__EmptyCollect.

        Parameters
        ----------
        pS :
            pS
        dC :
            dC
        WaveLength :
            WaveLength
        j :
            j
        """
        Empty0 = []
        Empty1 = torch.tensor([float('nan')]).to(device)
        Empty2 = torch.tensor([float('nan'),float('nan')]).to(device)
        Empty3 = torch.tensor([float('nan'),float('nan'),float('nan')]).to(device)
        RayTraceType = 0
        ValToSav = [[], Empty1, pS, pS, Empty3, Empty3, dC, Empty3, Empty3, Empty1, WaveLength, Empty1, Empty3, Empty0, Empty0, Empty0, j, RayTraceType]
        self.__CollectData(ValToSav)