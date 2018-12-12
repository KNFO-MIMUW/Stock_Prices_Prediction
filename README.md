# Stock_Prices_Prediction
The aim of the project is to implement and test different methods of the stock prices time series prediction. We assume that each participant of the project will implement different method. Built models should predict stock price of the specific company one day ahead - models will be tested and compared with each other on the real data. Plan of the project is as follows.
1. We select specific company from S&P 500 and set of time series, which may have influence on the prices of selected company (for instance - if you decide to predict the prices of Exxon Mobil Corp, then we probably should look at the prices of oil). Based on the selected set of time series we conduct basic analysis of the time series.
2. We conduct basic analysis of the time series - we check trend, seasonality, heteroscedasticity, stationarity, etc. What is more, one can compute statistics and plot the results.
3. We build predictive models by means of different classic time series prediction and machine learning techniques. Methods to consider:
• SVM - support vector machine;
• LSTM - Long Short-Term Memory;
• ARIMA, ARMA, SARIMA, etc;
• GARCH;
• Exponential smoothing, Holt-Winters;
• Kalman filtering;
• MLP - Multilayer perceptron;
• CNN - Convolutional Neural Network;
• other, which can be found in the publications, books, etc.
