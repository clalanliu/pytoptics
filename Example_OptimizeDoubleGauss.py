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
opticalsystem = pytoptics.OpticalSystem(codev_seq="CodeV_File/DoubleGauss.seq")

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
#opticalsystem.SetAperture(pytoptics.ApertureType.ENTRANCE_PUPIL_DIAMETER, 10)
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

