"""Examp Perfect Lens"""
import sys
original_stdout = sys.stdout
import time
import matplotlib.pyplot as plt
import numpy as np
import KrakenOS as Kos
import torch
torch.autograd.set_detect_anomaly(True)

import pytoptics
from pytoptics.Model import OpticalSystem

torch.set_default_tensor_type(pytoptics.torchPrecision)
device = pytoptics.torchdevice
# ______________________________________#

start_time = time.time()
# ______________________________________#

opticalsystem = OpticalSystem(5)
# ______________________________________#

opticalsystem.surfaces[0].Rc = torch.tensor(0.0).to(device)
opticalsystem.surfaces[0].Thickness = torch.tensor(10.0).to(device) 
opticalsystem.surfaces[0].Glass = "AIR"
opticalsystem.surfaces[0].Diameter = torch.tensor(30.0).to(device)

# ______________________________________#

opticalsystem.surfaces[1].Rc = torch.nn.Parameter(torch.tensor(90.0).to(device))
opticalsystem.surfaces[1].Thickness = torch.tensor(6.0).to(device)
opticalsystem.surfaces[1].Glass = "BK7"
opticalsystem.surfaces[1].Diameter = torch.tensor(30.0).to(device)
opticalsystem.surfaces[1].Axicon = torch.tensor(0).to(device)

# ______________________________________#

opticalsystem.surfaces[2].Rc = torch.tensor(-3.071608670000159E+001).to(device)
opticalsystem.surfaces[2].Thickness = torch.tensor(3.0).to(device)
opticalsystem.surfaces[2].Glass = "F2"
opticalsystem.surfaces[2].Diameter = torch.tensor(30).to(device)

# ______________________________________#

opticalsystem.surfaces[3].Rc = torch.nn.Parameter(torch.tensor(-100.0).to(device))#torch.tensor(-7.819730726078505E+001).to(device)
opticalsystem.surfaces[3].Thickness = torch.nn.Parameter(torch.tensor(90.0).to(device))
opticalsystem.surfaces[3].Glass = "AIR"
opticalsystem.surfaces[3].Diameter = torch.tensor(30).to(device)

# ______________________________________#

opticalsystem.surfaces[4].Rc = torch.tensor(0.0).to(device)
opticalsystem.surfaces[4].Thickness = torch.tensor(0.0).to(device)
opticalsystem.surfaces[4].Glass = "AIR"
opticalsystem.surfaces[4].Diameter = torch.tensor(10.0).to(device)
opticalsystem.surfaces[4].Name = "Plano imagen"
#P_Ima = pytoptics.surf(P_Ima)

# ______________________________________#

opticalsystem.SetAperture(pytoptics.ApertureType.ENTRANCE_PUPIL_DIAMETER, 10.0)
opticalsystem.SetFields(pytoptics.FieldType.ANGLE, [[0, 0, 1.0]])
opticalsystem.SetWavelength([0.55])
opticalsystem.Initialize()
opticalsystem.ShowModel2D()
# ______________________________________#

opticalsystem.Optimize_GradientDescent(MXC=5)
    
# ______________________________________#
opticalsystem.ShowOptimizeLoss()
opticalsystem.ShowSpotDiagram()
opticalsystem.ShowModel2D()

# ______________________________________#

# ______________________________________#

print("--- %s seconds ---" % (time.time() - start_time))
input("Press Enter to continue...")

