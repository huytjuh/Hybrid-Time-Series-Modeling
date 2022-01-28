### SARIMA MODEL ###

def rmse(resid):
  return np.sqrt(np.square(resid)).mean()

def mape(resid, true):
  return np.mean(np.abs(resid)/true)

# INITIALIZE/SELECT MARKET
df = data.copy()
list_markets = ['Europe', 'Rest of the world']

list_params = {'p': list(range(0,4)), 'q': list(range(0,4)), 'P': list(range(0,3)), 'Q': list(range(0,3))}
list_SARIMA = {}
list_order = {'Europe': (0,1,3), 'Rest of the world': (1,1,1)}
list_best_model = {}
for market in list_markets:
  random_seed = 1234
  df_market = df[df['Mega Market'] == market]

  # SPLIT INTO TRAINING & TEST SET
  X_train = df_market[df_market['Date_RAW'] < '2020-07-01']['Quantity'].reset_index(drop=True)
  X_test = df_market[df_market['Date_RAW'] >= '2020-07-01']['Quantity'].reset_index(drop=True)

  df_SARIMA = pd.DataFrame()
  for order in [(p,1,q) for p, q in itertools.product(list_params['p'], list_params['q'])]:
    for seasonal_order in [(P,1,Q,12) for P,Q in itertools.product(list_params['P'], list_params['Q'])]:
      SARIMA = sm.tsa.SARIMAX(X_train, order=order, seasonal_order=seasonal_order, trend='c', random_seed=random_seed).fit()
      
      temp = pd.DataFrame({'Order': [order], 'Seasonal Order': [seasonal_order], 'AIC': [SARIMA.aic], 'In-sample RMSE': [rmse(SARIMA.resid)], 'In-sample MAPE': [mape(SARIMA.resid, X_train)]})
      df_SARIMA = df_SARIMA.append(temp, ignore_index=True)

  list_SARIMA[market] = df_SARIMA
  print('{} - SARIMA models: '.format(market))
  print(df_SARIMA)
  best_model = df_SARIMA.sort_values('AIC').iloc[0, :]
  list_best_model[market] = best_model
  print('\n Best Model: {} {}], AIC={} \n'.format(best_model['Order'], best_model['Seasonal Order'], best_model['AIC']))
  
list_SARIMA['Europe']
