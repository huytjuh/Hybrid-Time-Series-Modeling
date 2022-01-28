### ARIMA MODEL ###

# INITIALIZE/SELECT MARKET
df = data.copy()
list_markets = ['Europe', 'Rest of the world']

# PERFORM FORECASTING ANALYSIS
list_preds_ARIMA = {}
list_resid_ARIMA = {}
for market in list_markets:
  random_seed = 1234
  df_market = df[df['Mega Market'] == market]

  # SPLIT INTO TRAINING & TEST SET
  X_train = df_market[df_market['Date_RAW'] < '2020-07-01']['Quantity'].reset_index(drop=True)
  X_test = df_market[df_market['Date_RAW'] >= '2020-07-01']['Quantity'].reset_index(drop=True)

  # ARIMA MODEL MULTI-STEP AHEAD
  #ARIMA = auto_arima(X_train, max_p=3, max_q=3, d=1, seasonal=False, stepwise=False, random_state=random_seed).fit(X_train)
  ARIMA = sm.tsa.SARIMAX(X_train, order=list_best_model[market]['Order'], trend='c', random_seed=random_seed).fit()
  #ARIMA_preds = ARIMA.predict(n_periods=len(X_test))
  ARIMA_preds = ARIMA.forecast(steps=len(X_test))
  ARIMA_resid = pd.Series(ARIMA.resid)

  # STORE RESULTS ARIMA
  df_results = pd.DataFrame({'ARIMA_preds': ARIMA_preds})
  list_preds_ARIMA[market] = df_results

  # STORE RESIDUALS OF ARIMA
  df_resid = pd.DataFrame({'ARIMA': ARIMA_resid})
  list_resid_ARIMA[market] = df_resid

df_ARIMA = create_df_plot('ARIMA', list_preds_ARIMA)
df_ARIMA_resid = create_df_plot_resid('ARIMA_Resid', list_resid_ARIMA)
