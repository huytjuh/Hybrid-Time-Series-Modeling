import scipy.optimize as optimize

def fun_HW(theta):
  alpha, beta, gamma = theta
  HW = ExponentialSmoothing(X_train, trend='mul', seasonal='mul', seasonal_periods=12).fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)
  HW_resid = HW.resid
  return rmse(HW_resid)

def callbackF(theta):
  global df_grid
  alpha, beta, gamma = theta
  HW = ExponentialSmoothing(X_train, trend='mul', seasonal='mul', seasonal_periods=12).fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)
  _rmse = rmse(HW.resid)
  _mape = mape(HW.resid, X_train)

  temp = pd.DataFrame({'alpha': [alpha], 'beta': [beta], 'gamma': [gamma], 'RMSE': [_rmse], 'MAPE': [_mape]})
  df_grid = df_grid.append(temp, ignore_index=True)
  return

# INITIALIZE/SELECT MARKET
df = data.copy()
list_markets = ['Europe', 'Rest of the world']

# PERFORM FORECASTING ANALYSIS
list_grid_HW = {}
list_theta = {'Europe': (0.05,0.1,0.5), 'Rest of the world': (0.5, 0.5, 0.5)}
for market in list_markets:
  random_seed = 1234
  df_market = df[df['Mega Market'] == market]

  # SPLIT INTO TRAINING & TEST SET
  X_train = df_market[df_market['Date_RAW'] < '2020-07-01']['Quantity'].reset_index(drop=True)
  X_test = df_market[df_market['Date_RAW'] >= '2020-07-01']['Quantity'].reset_index(drop=True)

  df_grid = pd.DataFrame()
  theta0 = list_theta[market]
  fmin = optimize.minimize(fun_HW, x0=theta0, method='L-BFGS-B', callback=callbackF, bounds=((0,1),(0,1),(0,1)), options={'gtol': 1e-4, 'disp': True})

  list_grid_HW[market] = df_grid

list_grid_HW['Europe'].to_csv('HW_Gridsearch_Europe.csv', index=False)
list_grid_HW['Europe']
