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

torch.set_default_tensor_type(pytoptics.torchPrecision)
device = pytoptics.torchdevice
# ______________________________________#

start_time = time.time()
# ______________________________________#
opticalsystem = pytoptics.OpticalSystem(5)
# ______________________________________#
opticalsystem.surfaces[0].Rc = torch.tensor(0.0).to(device)
opticalsystem.surfaces[0].Thickness = torch.tensor(10.0).to(device) 
opticalsystem.surfaces[0].Glass = "AIR"
opticalsystem.surfaces[0].Diameter = torch.tensor(30.0).to(device)
# ______________________________________#
opticalsystem.surfaces[1].Rc = torch.nn.Parameter(torch.tensor(90.0).to(device))
opticalsystem.surfaces[1].Thickness = torch.nn.Parameter(torch.tensor(6.0).to(device))
opticalsystem.surfaces[1].Glass = "BK7"
opticalsystem.surfaces[1].Diameter = torch.tensor(30.0).to(device)
opticalsystem.surfaces[1].Axicon = torch.tensor(0).to(device)
# ______________________________________#
opticalsystem.surfaces[2].Rc = torch.nn.Parameter(torch.tensor(-3.071608670000159E+001).to(device))
opticalsystem.surfaces[2].Thickness = torch.nn.Parameter(torch.tensor(3.0).to(device))
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
opticalsystem.surfaces[4].Diameter = torch.tensor(30.0).to(device)
opticalsystem.surfaces[4].Name = "Plano imagen"
# ______________________________________#
constraints = [
    [opticalsystem.surfaces[1].Rc, ">", 85.0],
    [opticalsystem.surfaces[1].Rc, "<", 95.0],
    [opticalsystem.surfaces[1].Thickness, "=", 6.0, 0.5],  # half-width = 0.5
    [opticalsystem.surfaces[2].Rc, ">", -40.0],
    [opticalsystem.surfaces[2].Rc, "<", -20.0],
    [opticalsystem.surfaces[2].Thickness, ">", 2.8],
    [opticalsystem.surfaces[2].Thickness, "<", 4.0],
    [opticalsystem.surfaces[3].Rc, ">", -120.0],
    [opticalsystem.surfaces[3].Rc, "<", -70.0],
    [opticalsystem.surfaces[3].Thickness, ">", 80],
]

opticalsystem.AddConstraint(constraints)
# ______________________________________#
opticalsystem.SetAperture(pytoptics.ApertureType.ENTRANCE_PUPIL_DIAMETER, 10.0)
opticalsystem.SetFields(pytoptics.FieldType.ANGLE, [[0, 0, 1.0], [0, 5, 1.0], [0, -5, 1.0]])
opticalsystem.SetWavelength([0.55])
opticalsystem.Initialize()
opticalsystem.Trace()
opticalsystem.ShowModel2D()
opticalsystem.PrintVariables()
# ______________________________________#

opticalsystem.Optimize_GradientDescent(MXC=100, lr=10)
# ______________________________________#
opticalsystem.PrintVariables()
opticalsystem.ShowOptimizeLoss()
opticalsystem.ShowSpotDiagram()
opticalsystem.ShowModel2D()
# ______________________________________#

print("--- %s seconds ---" % (time.time() - start_time))
input("Press Enter to continue...")

