import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


def eps_heatmap(M, model='gravity'):
	'''
	Takes epsilon matrix and plots a heatmap for the locations
	'''

	ax = sns.heatmap(M, square=True)
	plt.rc('text', usetex=True)
	ax.set_xlabel('Location Origin')
	ax.set_xlabel('Location Destination')

	ax.set_title(r'$\epsilon$' + ' in the ' + model)

	plt.show()

	return 