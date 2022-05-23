
# PyTorch Implementation of Optics Simulator and Optimizer (Pytoptics)         
![GitHub Logo]()

Pytoptics (PyTorch Implementation of Optics Simulator and Optimizer) is a python library extended from [KrakenOS](https://github.com/Garchupiter/Kraken-Optical-Simulator/tree/KrakenOS) and based in Pytorch, Numpy, Matplotlib, PyVTK and PyVista libraries. In addition to the original utilities in KrakenOS including performing sequential exact ray tracing, generating off-axis systems and calculation of wavefront aberrations in terms of Zernike polynomials, Pytoptics enables users to optimize their optical systems with Pytorch optimizer


## Install Pytoptics
```python
pip install Pytoptics
```

## Prerequisites
To install prerequisites

```python
pip install pyvista
pip install PyVTK
pip install vtk
pip install numpy
pip install scipy
pip install matplotlib
pip install csv342
pip install KrakenOS
pip install torch torchvision torchaudio
```

## Usage
To construct a optical system and assign surface information with trainable pytorch parameters:

```python
opticalsystem = pytoptics.OpticalSystem(5)
...
opticalsystem.surfaces[1].Rc = torch.nn.Parameter(torch.tensor(90.0).to(device))
opticalsystem.surfaces[1].Thickness = torch.nn.Parameter(torch.tensor(6.0).to(device))
opticalsystem.surfaces[1].Glass = "BK7"
opticalsystem.surfaces[1].Diameter = torch.tensor(30.0).to(device)
opticalsystem.surfaces[1].Axicon = torch.tensor(0).to(device)  
```

To add optimization constraints:

```python
constraints = [
    [opticalsystem.surfaces[1].Rc, ">", 85.0],
    [opticalsystem.surfaces[1].Rc, "<", 95.0],
    [opticalsystem.surfaces[1].Thickness, "=", 6.0, 0.5],  # half-width = 0.5
]
opticalsystem.AddConstraint(constraints)
```

To configure the system:
```python
opticalsystem.SetAperture(pytoptics.ApertureType.ENTRANCE_PUPIL_DIAMETER, 10.0)
opticalsystem.SetFields(pytoptics.FieldType.ANGLE, [[0, 0, 1.0], [0, 5, 1.0], [0, -5, 1.0]])
opticalsystem.SetWavelength([0.55])
```

To initialize, optimize, and show the system:
```python
opticalsystem.Initialize()
opticalsystem.Optimize_GradientDescent(MXC=100, lr=10)
opticalsystem.ShowOptimizeLoss()
opticalsystem.ShowSpotDiagram()
opticalsystem.ShowModel2D()
```

See Example_*.py to get more information.

## Reference
If you find the codes useful, please cite this paper
```
@article{pytoptics,
  title={Pytoptics: Optimzie Optical System in PyTorch},
  author={Chang-Le, Liu}
}
```