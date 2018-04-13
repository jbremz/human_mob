# Contains details for graph formatting conventions
import matplotlib.ticker as plticker

# Global settings
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('seaborn-deep')

# Resolution
fig = plt.figure(figsize=(840/110.27, 800/110.27), dpi=300)

# Gravity
ax.scatter(yEps[:,0], yEps[:,1], s=60, label='Simulation', color='C5', marker='x')
ax.errorbar(yEps[:,0], yEps[:,1], yerr=sigmaEps, elinewidth=1, fmt='o', ms=2, color='C5')

# Radiation
ax.scatter(yEps[:,0], yEps[:,1], s=60, label='Simulation', color='C2', marker='x')
ax.errorbar(yEps[:,0], yEps[:,1], yerr=sigmaEps, elinewidth=1, fmt='o', ms=2, color='C2')

# Analytical
ax.plot(anlytYEps[:,0], anlytYEps[:,1], label='Analytical', color='grey')

# Legend
ax.legend(frameon=False, fontsize=25)

# Axes/tick labels
loc = plticker.MultipleLocator(base=0.02) # you can change the base as desired
ax.yaxis.set_major_locator(loc)
ax.set_xlabel(r'$r_{jk} \sqrt{N}$', fontsize=30)
ax.set_ylabel(r'$\epsilon$', fontsize=40)
plt.tick_params(axis='both', labelsize=20)
ax.ticklabel_format(style='sci')

# This can be useful
plt.ylim(-0.001,0.006)

# stops lables being cut off
plt.tight_layout()

# saves figure
plt.savefig(time_label())





# -------------------------  OLD PLOT INFO  ------------------------------

# Resolution
fig = plt.figure(figsize=(800/110.27, 800/110.27), dpi=300)

# Gravity
ax.scatter(yEps[:,0], yEps[:,1], s=20, label='Simulation', color='C5', marker='x')

# Radiation
ax.scatter(yEps[:,0], yEps[:,1], s=20, label='Simulation', color='C4', marker='x')

# Analytical
ax.plot(anlytYEps[:,0], anlytYEps[:,1], label='Analytical', color='grey')

# Legend
ax.legend(frameon=False, fontsize=20)

# Axes/tick labels
ax.set_xlabel(r'$r_{jk} \sqrt{N}$', fontsize=20)
ax.set_ylabel(r'$\epsilon$', fontsize=20)
plt.tick_params(axis='both', labelsize=15)
ax.ticklabel_format(style='sci')

# stops lables being cut off
plt.tight_layout()

# saves figure
plt.savefig(time_label())