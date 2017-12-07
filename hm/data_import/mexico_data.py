import gdal
import numpy as np


data = gdal.Open("/Users/Ilaria/Documents/Imperial/MSci_Project/Datasets/hrsl_mex_v1/hrsl_mex.tif")

for n in range(data.RasterCount):
	band = data.GetRasterBand(n+1)
	band.ReadAsArray()
