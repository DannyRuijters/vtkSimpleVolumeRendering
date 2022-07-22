#!/usr/bin/env python3

import os
import pydicom
import numpy as np
import vtk
from vtk.util import numpy_support
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkRenderingCore import (
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)


def array_as_vtk_image_data(image_array, dimensions, spacing):
    vtk_type_by_numpy_type = {
        np.uint8: vtk.VTK_UNSIGNED_CHAR,
        np.uint16: vtk.VTK_UNSIGNED_SHORT,
        np.uint32: vtk.VTK_UNSIGNED_INT,
        np.uint64: vtk.VTK_UNSIGNED_LONG if vtk.VTK_SIZEOF_LONG == 64 else vtk.VTK_UNSIGNED_LONG_LONG,
        np.int8: vtk.VTK_CHAR,
        np.int16: vtk.VTK_SHORT,
        np.int32: vtk.VTK_INT,
        np.int64: vtk.VTK_LONG if vtk.VTK_SIZEOF_LONG == 64 else vtk.VTK_LONG_LONG,
        np.float32: vtk.VTK_FLOAT,
        np.float64: vtk.VTK_DOUBLE
    }
    vtk_datatype = vtk_type_by_numpy_type[image_array.dtype.type]
    depth_array = numpy_support.numpy_to_vtk(image_array.ravel(), deep=False, array_type = vtk_datatype)
    depth_array.SetNumberOfComponents(1)

    vtkImage = vtk.vtkImageData()
    vtkImage.SetDimensions(dimensions)
    vtkImage.SetSpacing(spacing)
    vtkImage.SetOrigin([0,0,0])
    vtkImage.GetPointData().SetScalars(depth_array);
    return vtkImage



dirSlices = '../../data/MANIX_CTA/'  #3D voxel volume, stored per slice
dicomSlice = []

for (dirpath, dirnames, filenames) in os.walk(dirSlices):
    for filename in filenames:
        dicomSlice.append(pydicom.read_file(dirSlices+filename))


volume = lambda:0  #empty struct
volume.nrOfVoxels = [dicomSlice[0].Columns, dicomSlice[0].Rows, len(dicomSlice)]
volume.voxelSize = [dicomSlice[0].PixelSpacing[1], dicomSlice[0].PixelSpacing[0], abs(dicomSlice[0].SliceLocation - dicomSlice[1].SliceLocation)]
volume.pixel_array = np.empty(volume.nrOfVoxels[::-1], dicomSlice[0].pixel_array.dtype)  #shape is in z,y,x

for slice in range(0, len(dicomSlice)):
    np.copyto(volume.pixel_array[slice], dicomSlice[slice].pixel_array)


stretch = 0.05
alphaChannelFunc = vtk.vtkPiecewiseFunction()
alphaChannelFunc.AddPoint(0, 0.0)
alphaChannelFunc.AddPoint(stretch * 22102, 0.0)
alphaChannelFunc.AddPoint(stretch * 22103, 0.3)
alphaChannelFunc.AddPoint(stretch * 42532, 0.7)
alphaChannelFunc.AddPoint(stretch * 65535, 1.0)

colorFunc = vtk.vtkColorTransferFunction()
colorFunc.AddRGBPoint(0, 0.0, 0.0, 0.0)
colorFunc.AddRGBPoint(stretch * 22102, 0.0, 0.0, 0.0)
colorFunc.AddRGBPoint(stretch * 22103, 0.8, 0.0, 0.0)
colorFunc.AddRGBPoint(stretch * 32532, 1.0, 0.5, 0.5)
colorFunc.AddRGBPoint(stretch * 42532, 1.0, 0.9, 0.9)
colorFunc.AddRGBPoint(stretch * 65535, 1.0, 1.0, 1.0)

volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetColor(colorFunc)
volumeProperty.SetScalarOpacity(alphaChannelFunc)
volumeProperty.SetInterpolationTypeToLinear()
volumeProperty.ShadeOn()

volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()
vtkImage = array_as_vtk_image_data(volume.pixel_array, volume.nrOfVoxels, volume.voxelSize)
volumeMapper.SetInputData(vtkImage)

vtk_volume = vtk.vtkVolume()
vtk_volume.SetMapper(volumeMapper)
vtk_volume.SetProperty(volumeProperty)

renderer = vtkRenderer()
render_window = vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetWindowName('3D Visualization')
interactor = vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

renderer.AddVolume(vtk_volume)

render_window.Render()
interactor.Start()


