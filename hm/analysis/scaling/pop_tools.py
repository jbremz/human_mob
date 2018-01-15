from hm.pop_models.pop_explicit import explicit as pop_explicit
import numpy as np 

def make_pop(df):
	'''
	Takes a pandas dataframe object from the CDRC dataset and returns a population object

	'''
	x = np.array(df['Easting'])
	y = np.array(df['Northing'])
	m = np.array(df['TotPop2011'])
	xy = np.array([x, y]).T

	return pop_explicit(xy, m)