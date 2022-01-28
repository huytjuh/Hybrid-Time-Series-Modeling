### SVR MODEL ###

# INITIALIZE/SELECT MARKET
df = data.copy()
list_markets = ['Europe', 'Rest of the world']

# FUNCTION TO CREATE LAGS USED FOR PERFORMING MODELS
def create_lag(data, lags=[]):
  output = pd.DataFrame({'value': data.values.ravel()})
  for i in lags:
    output['lag_{}'.format(i)] = data.shift(i)
  output = output.dropna()
  return output

# PERFORM FORECASTING ANALYSIS
list_preds_SVR = {}
for market in list_markets:
  random_seed = 1234
  df_market = df[df['Mega Market'] == market]

  Y_scaler = MinMaxScaler().fit(df_market['Quantity'].to_frame())
  df_market['Quantity'] = Y_scaler.transform(df_market['Quantity'].to_frame()).ravel()
  
  #############################
  ### FEATURE/LAG SELECTION ###
  #############################

  # ADD LAGGED VARIABLES AS PREDICTORS: PAST 24-MONTHS
  X = df_market[df_market['Date_RAW'] < '2020-07-01']['Quantity'].reset_index(drop=True)
  X = pd.DataFrame(X)
  lag_range = [i+1 for i in range(24)]
  X = create_lag(X.iloc[:, :1], lag_range)

  # SPLIT INTO TRAINING & TEST SET
  X_train = X.iloc[:, 1:]
  y_train = X.iloc[:, :1]

  # FEATURE SELECTION ON LAGGED REGRESSORS USING RF BASED ON PERMUTATION PERFORMANCES
  RF = LGBMRegressor(boosting_type='rf', bagging_freq=1, bagging_fraction=0.8, feature_fraction=0.8, random_state=random_seed).fit(X_train, y_train)
  result = permutation_importance(RF, X_train, y_train, n_repeats=10, random_state=random_seed)

  # USE SELECTED LAGS OBTAINED FROM FEATURE SELECTION
  lags_selected = 1 + np.sort(result.importances_mean.argsort()[::-1][:5])

  ###############################
  ### PERFORM SVR FORECASTING ###
  ###############################

  # SPLIT INTO TRAINING & TEST SET
  X_train = df_market[df_market['Date_RAW'] < '2020-07-01']['Quantity'].reset_index(drop=True)
  X_test = df_market[df_market['Date_RAW'] >= '2020-07-01']['Quantity'].reset_index(drop=True)

  X_train_lagged = create_lag(X_train, lags=lags_selected)
  X_test_lagged = X_train.sort_index(ascending=False).reset_index(drop=True)[[i-1 for i in lags_selected]]
  X_test_lagged = pd.DataFrame(X_test_lagged).T
  X_test_lagged.columns = X_train_lagged.columns[1:]

  # INITIALIZE REGION FOR GRIDSEARCH CV USING TIMESERIES SPLIT
  tcv = TimeSeriesSplit(n_splits=4)
  n_jobs = -1
  C = [0.001, 0.01, 0.1, 1, 10, 100] 
  gamma = [0.0001, 0.001, 0.01, 0.1, 1, 10]
  
  # SVM USING RBF KERNEL
  np.random.RandomState(random_seed)
  SVR_rbf = SVR(kernel='rbf', cache_size=1500)
  SVR_rbf_optimal = GridSearchCV(estimator=SVR_rbf, param_grid=dict(C=C, gamma=gamma), n_jobs=n_jobs, cv=tcv).fit(X_train_lagged.iloc[:, 1:], X_train_lagged.iloc[:, :1].values.ravel()).best_estimator_
  
  # APPLY RECURSIVE MULTI-STEP AHEAD FORWARD PREDICTION
  SVR_preds = []
  for i in range(len(X_test)):
    X_train_lagged = create_lag(X_train, lags=lags_selected)
    X_test_lagged = X_train.sort_index(ascending=False).reset_index(drop=True).loc[[i-1 for i in lags_selected]]
    X_test_lagged = pd.DataFrame(X_test_lagged).T
    X_test_lagged.columns = X_train_lagged.columns[1:]

    SVR_rbf_fit = SVR_rbf_optimal
    SVR_preds.append(SVR_rbf_fit.predict(X_test_lagged)[0])

    X_train = X_train.append(pd.Series(SVR_preds[i]), ignore_index=True)

  # STORE RESULTS SVR
  df_results = pd.DataFrame({'SVR_preds': Y_scaler.inverse_transform(pd.DataFrame(SVR_preds)).ravel()})
  list_preds_SVR[market] = df_results

df_SVR = create_df_plot('SVR', list_preds_SVR)
create_graph(df_SVR)
