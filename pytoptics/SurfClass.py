
import numpy as np
import pyvista as pv
import os
import sys
from .MathShapesClass import *
from .PhysicsClass import *
import KrakenOS as Kos
import torch
from .Global import *
torch.set_default_tensor_type(torchPrecision)
device = torchdevice

class surf(torch.nn.Module, Kos.surf):
    
    def __init__(self):
        super(surf, self).__init__()
        self.Rc = torch.tensor(0.0).to(device)
        self.Thickness = torch.tensor(1e-11).to(device)
        self.Diameter = torch.tensor(1.0).to(device)
        self.InDiameter = torch.tensor(0.0).to(device)
        self.k = torch.tensor(0.0).to(device)
        self.ZNK = torch.from_numpy(np.zeros(36)).to(device)
        self.Glass = 'AIR'
        self.DespX = torch.tensor(0.0).to(device)
        self.DespY = torch.tensor(0.0).to(device)
        self.DespZ = torch.tensor(0.0).to(device)
        self.TiltX = torch.tensor(0.0).to(device)
        self.TiltY = torch.tensor(0.0).to(device)
        self.TiltZ = torch.tensor(0.0).to(device)
        self.Order = 0.0
        self.AxisMove = 1.0
        self.Diff_Ord = 0.0
        self.Grating_D = torch.tensor(0.0).to(device)
        self.Grating_Angle = torch.tensor(0.0).to(device)
        self.ShiftX = torch.tensor(0.0).to(device)
        self.ShiftY = torch.tensor(0.0).to(device)
        self.Mask_Type = 0.0

        self.Mask_Shape = "None"
        self.Solid_3d_stl = 'None'
        self.Cylinder_Rxy_Ratio = torch.tensor(1.0).to(device)
        self.Axicon = 0.0
        self.AspherData = torch.from_numpy(np.zeros(200)).to(device)
        self.ExtraData = torch.from_numpy(np.zeros(200)).to(device)
        self.Thin_Lens = 0.0
        self.Name = ''
        self.Nm_Poss = (0.0, 0.0)
        self.Note = 'None'
        self.Drawing = 1.0
        self.Color = [0, 0, 0]
        self.Error_map = []
        self.Res = 1
        self.Surface_type = 0.0
        self.SURF_FUNC = []
        self.SPECIAL_SURF_FUNC = []
        self.SURF_FUNC.append(conic__surf(0.0, 0.0, 1.0))
        self.General_Status = self.update()
        self.SubAperture = [1,0,0]

    def build_surface_function(self):
        """build_surface_function.
        """
        self.SURF_FUNC = []
        self.SPECIAL_SURF_FUNC = []
        self.Surface_type = 0
        if (self.Diff_Ord != 0):
            self.Surface_type = 1
            print('Surface type 1')
        if (self.Thin_Lens != 0):
            self.Surface_type = 2
            print('Surface type 2')
        if (self.Solid_3d_stl != 'None'):
            self.Surface_type = 3
            print('Surface type 3')
        if ((self.Diff_Ord != 0) and (self.Thin_Lens != 0)):
            self.warning()
            self.Surface_type = 0
        if ((self.Solid_3d_stl != 'None') and (self.Thin_Lens != 0)):
            self.warning()
            self.Surface_type = 0
        if ((self.Solid_3d_stl != 'None') and (self.Diff_Ord != 0)):
            self.warning()
            self.Surface_type = 0
        if (self.Surface_type == 0):
            self.PHYSICS = snell_refraction_vector_physics()
            if torch.any((self.ZNK != 0)):
                NC = len(self.ZNK)
                (self.Zern_pol, self.z_pow) = zernike_expand(NC)
                FUNC_0 = zernike__surf(self.ZNK, self.Zern_pol, self.z_pow, self.Diameter)
                self.SURF_FUNC.append(FUNC_0)
            if torch.any((self.AspherData != 0)):
                FUNC_1 = aspheric__surf(self.AspherData)
                self.SURF_FUNC.append(FUNC_1)
            if (self.Rc != 0):
                FUNC_2 = conic__surf(self.Rc, self.k, self.Cylinder_Rxy_Ratio)
                self.SURF_FUNC.append(FUNC_2)
            if (self.Axicon != 0):
                FUNC_3 = axicon__surf(self.Cylinder_Rxy_Ratio, self.Axicon)
                self.SURF_FUNC.append(FUNC_3)
            if torch.any((self.ExtraData != 0)):
                FUNC_4 = extra__surf(self.ExtraData)
                self.SURF_FUNC.append(FUNC_4)
            if (len(self.Error_map) != 0):
                [X, Y, Z, SPACE] = self.Error_map
                FUNC_5 = error_map__surf(X, Y, Z, SPACE)
                self.SURF_FUNC.append(FUNC_5)
        if (self.Surface_type == 1):
            self.PHYSICS = diffraction_grating_physics()
            FUNC = conic__surf(0, self.k, self.Cylinder_Rxy_Ratio)
            self.SURF_FUNC.append(FUNC)
        if (self.Surface_type == 2):
            self.PHYSICS = paraxial_exact_physics()
            FUNC = conic__surf(0, self.k, self.Cylinder_Rxy_Ratio)
            self.SURF_FUNC.append(FUNC)
            self.Glass = 'AIR'
        if (self.Surface_type == 3):
            self.PHYSICS = snell_refraction_vector_physics()
            FUNC = conic__surf(0, self.k, self.Cylinder_Rxy_Ratio)
            self.SURF_FUNC.append(FUNC)


    def sigma_z(self, x, y, case):
        """sigma_z.
        Parameters
        ----------
        x :
            x
        y :
            y
        case :
            case
        """
        x = (x + self.ShiftX)
        y = (y + self.ShiftY)
        # Z = 0.0 * np.copy(x)
        con = -1
        N_FUNC = len(self.SURF_FUNC)

        for i in range(0, (N_FUNC - case)):

            if con==-1:
                Z = self.SURF_FUNC[i].calculate(x, y)
                con = 0

            else:
                Z = (self.SURF_FUNC[i].calculate(x, y) + Z)
                con=1

        if con == -1:
            Z = 0.0 * x

        return Z


def convert_surface(surface_list):
    torch_surface_list = []
    for surface in surface_list:
        if type(surface) is Kos.MathShapesClass.conic__surf:
            torch_surface_list.append(conic__surf(surface.R_C, surface.KON, surface.C_RXY_RATIO))
        elif type(surface) is Kos.MathShapesClass.aspheric__surf:
            torch_surface_list.append(aspheric__surf(surface.E))
        elif type(surface) is Kos.MathShapesClass.axicon__surf:
            torch_surface_list.append(axicon__surf(surface.C_RXY_RATIO, surface.AXC))
        elif type(surface) is Kos.MathShapesClass.zernike__surf:
            torch_surface_list.append(zernike__surf(surface.COEF, surface.Z_POL, surface.Z_POW, surface.DMTR))
        
        else:
            torch_surface_list.append(surface)


        




''' Garbage
def __init__(self, KOS_surface):
        super(surf, self).__init__()
        for attribute, value in KOS_surface.__dict__.items():
            if (type(value) is float) and (attribute not in {'Surface_type'}):
                setattr(self, attribute, torch.tensor(value).to(device)))
            if type(value) is int and (attribute not in {'Surface_type'}):
                setattr(self, attribute, torch.tensor(float(value)).to(device)))
            elif type(value) is list:
                if len(value)==0:
                    setattr(self, attribute, value)
                elif (type(value[0]) is float) or (type(value[0]) is int):
                    value_ = [torch.tensor(float(v)).to(device)) for v in value]
                    setattr(self, attribute, value_)
                else:
                    value_ = convert_surface(value)
                    setattr(self, attribute, value_)
            elif type(value) is tuple:
                value_ = [torch.tensor(float(v)).to(device)) for v in value]
                setattr(self, attribute, value_)
            elif type(value) is np.ndarray:
                setattr(self, attribute, torch.from_numpy(value.astype(np.float))).to(device))
            else:
                setattr(self, attribute, value)

'''