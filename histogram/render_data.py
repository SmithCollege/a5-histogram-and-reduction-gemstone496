# Import seaborn
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Apply default theme
sns.set_theme()

# Load data
PATH = "data/"
cpu = pd.read_csv(PATH + "cpu.csv", header=0, index_col=0)
diverge = pd.read_csv(PATH + "unstrided.csv", header=0, index_col=0)
converge = pd.read_csv(PATH + "strided.csv", header=0, index_col=0)

# Transpose
cpu = cpu.transpose()
diverge = diverge.transpose()
converge = converge.transpose()

# Join
times = cpu.join(diverge, lsuffix='cpu', rsuffix='diverge').join(converge, rsuffix='converge')
times['run'] = times.index # turn this back into a proper row

print(times)

# Pull out size through a wide-long pivot
times = pd.wide_to_long(times, ['cpu', 'unstrided', 'strided'], i='run', j='size')
times = times.reset_index() # turn back to a proper row

print(times)

# Pull out method through a melting pivot
times  = pd.melt(times, id_vars=['run', 'size'], var_name='method', value_name='time').reset_index()

print(times)

# Create graph
plot = sns.lmplot(data=times, x='size', y='time', hue='method')
plt.xscale('log')

# Export graph
plt.savefig("histogram.png")
