# Authors: Damian Burczyk, Grzegorz Pielot
# Based on Based on "Stock prediction using deep learning" by Ritika Singh and Shashi Srivastava
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
import sys

# constants
data_path = 'F.csv'

# input
result_path = sys.stdin.readline().strip()
p = int(sys.stdin.readline().strip())
q = int(sys.stdin.readline().strip())
r = int(sys.stdin.readline().strip())
test_ratio = float(sys.stdin.readline().strip())
predict_by_rate = sys.stdin.readline().strip() == 'True'
date_from = sys.stdin.readline().strip()
date_to = sys.stdin.readline().strip()

parameters = (p, q, r)

# input data
raw = pd.read_csv(data_path, header=0, parse_dates=True, index_col=0, squeeze=True, date_parser=pd.to_datetime)
close = raw.truncate(before=date_from, after=date_to)['Close']

size = close.shape[0]
if predict_by_rate:
    data_tmp = ((np.array(close[1:size]) - np.array(close[0:size - 1])) / np.array(close[0:size - 1]))
    data = np.concatenate(([0], data_tmp))
else:
    data = close

# data split
split = int(size * (1.0 - test_ratio))
learning_set, test_set = data.values[:split], data.values[split:]
forecast = []

# result generation
for i in range(0, test_set.size):
    prediction = ARIMA(learning_set, order=parameters).fit(disp=-1).forecast()[0][0]

    if predict_by_rate:
        forecast.append(prediction * close.values[split + i - 1] + close.values[split + i - 1])
    else:
        forecast.append(prediction)

    learning_set = np.append(learning_set, test_set[i])

    if i % 10 == 0:
        print('ARIMA progress {}/{}'.format(i, test_set.size))

mse = mean_squared_error(close.values[split:], forecast)

# output
result = pd.DataFrame({'Close': close.values[split:], 'Forecast': forecast},
                      index=raw.truncate(before=date_from, after=date_to).index[split:])
result.to_csv(path_or_buf=result_path + '.csv')

report = open(result_path + '.txt', 'w+')

report.write('Model = ARIMA\n')

report.write('\nINPUT:\n')
report.write('date from = %s\n' % date_from)
report.write('date to = %s\n' % date_to)
report.write('test ratio = %f\n' % test_ratio)
report.write('predict by rate of close instead = %s\n' % str(predict_by_rate))
report.write('ARIMA parameters (p, q, r) = %s\n' % str(parameters))

report.write('\nOUTPUT:\n')
report.write('mean square error = %f\n' % mse)

report.close()
