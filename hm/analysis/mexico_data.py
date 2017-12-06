import gdal
import numpy as np


data = gdal.Open("hrsl_mex.tif")

for n in range(data.RasterCount):
	band = data.GetRasterBand(n+1)
	band.ReadAsArray()
