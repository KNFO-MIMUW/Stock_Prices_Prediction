import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima_model import ARIMA

TEST_RATIO = 0.80

ford_data = pd.read_csv('f_us_d.csv', header = 0, parse_dates = [0], index_col = 0, squeeze = True)
ford_data = ford_data.truncate(before = '2012-01-01', after = '2015-12-31')
ford_data = ford_data[['Open']]

split = int(ford_data.size * TEST_RATIO)
dataset, tests = ford_data.values[:split], ford_data.values[split:]
dataset = [x[0] for x in dataset]
tests = [x[0] for x in tests]
print(dataset)
print(tests)
predictions = list()

for i in range(len(tests)):
    model = ARIMA(dataset, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = tests[i]
    dataset.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

mae = mean_absolute_error(tests, predictions)

print('Loss (mean absolute error): ', mae)

plt.plot(tests, color='orange', label='True data')
plt.plot(predictions, label= 'Predictions')
plt.legend(loc='upper left')
plt.xlabel('Days')
plt.ylabel('Price')

plt.show()