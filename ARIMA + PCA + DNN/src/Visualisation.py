# Authors: Damian Burczyk, Grzegorz Pielot
# Based on Based on "Stock prediction using deep learning" by Ritika Singh and Shashi Srivastava
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import sys

register_matplotlib_converters()

# input
result_path = sys.stdin.readline().strip()

save_path = result_path
data_begin = 0

# input data
data = pd.read_csv(result_path + '.csv', header=0, parse_dates=True, index_col=0, squeeze=True, date_parser=pd.to_datetime)
data = data.truncate(before=data.index[data_begin], after=data.index[-1])

# data plotting
ticks = data.index[range(0, data.shape[0], data.shape[0]//10 + 1)]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data['Close'], label='Close')
ax.plot(data['Forecast'], label='Forecast')
ax.legend()
ax.set_xticks(ticks)
ax.set_xticklabels(ticks.date, rotation=45)
ax.set_xlim(xmin=data.index[0], xmax=data.index[-1])
ax.grid(linestyle='--')
fig.tight_layout()

fig_diff, ax_diff = plt.subplots(figsize=(10, 5))
ax_diff.plot(data['Forecast'] - data['Close'], label='Difference', color='red')
ax_diff.legend()
ax_diff.set_xticks(ticks)
ax_diff.set_xticklabels(ticks.date, rotation=45)
ax_diff.set_xlim(xmin=data.index[0], xmax=data.index[-1])
ax_diff.grid(linestyle='--')
fig_diff.tight_layout()

# output
fig.savefig(save_path)
fig_diff.savefig(save_path + 'Diff')

