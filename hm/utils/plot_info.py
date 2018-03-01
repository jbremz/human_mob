# Contains details for graph formatting conventions

# Global settings
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('seaborn-deep')

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