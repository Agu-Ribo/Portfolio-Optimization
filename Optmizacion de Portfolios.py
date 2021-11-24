import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from pandas_datareader import data as pdr

plt.style.use('seaborn-colorblind')
# Activos
tickers = ['GGAL','BMA','YPF','TS','MELI']
prices = pdr.get_data_yahoo(tickers, start = '2015-01-01', end = dt.date.today())['Adj Close']
returns = prices.pct_change()
# Graficamos la evolución de precios
plt.figure(figsize=(14, 7))
for i in prices.columns.values:
 plt.plot(prices.index, np.log(prices[i]), lw=2, alpha=0.8,label=i)
plt.legend(loc='upper left', fontsize=12)
plt.ylabel('log price')

from pypfopt import expected_returns
from pypfopt import risk_models

## no tenemos de donde sale monthly prices =( 
#monthly_prices = prices

## plan b:  la siguiente linea
monthly_prices = prices.resample('M').first()

mu = expected_returns.mean_historical_return(monthly_prices,frequency=12)
covmat = risk_models.sample_cov(monthly_prices, frequency = 12)
sd = np.sqrt(np.diag(covmat))
fig = plt.figure()
plt.plot(sd, mu.to_numpy(), 'x', markersize = 5)
plt.ylabel('Retorno esperado')
plt.xlabel('Volatilidad')
plt.title('Mean-Variance Analysis')
plt.show()

# Calculamos volatilidad y retorno de un portafolio
def portfolio_metrics(weights, mean_returns, cov_matrix):
 ret = np.sum(mean_returns * weights)
 std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
 return ret, std

# armamos los portafolios random
def random_portfolios(num_port, mean_returns, cov_matrix):
  metrics = np.zeros((2,num_port))
  weights_matrix = []
  for i in range(num_port):
    weights = np.random.random(len(mean_returns))
    weights /= np.sum(weights)
    weights_matrix.append(weights)
    port_mu, port_std = portfolio_metrics(weights, mean_returns, cov_matrix)
    metrics[0,i] = port_mu
    metrics[1,i] = port_std
  return metrics, weights_matrix

n_port = 100000
metrics, weights_matrix = random_portfolios(n_port, mu, covmat)
plt.figure(figsize=(16,9))
plt.plot(metrics[1,:], metrics[0,:], 'o')
plt.scatter(sd, mu, marker = 'x', color = 'r')
plt.title('Random Portfolios')
plt.xlabel('Volatilidad anualizada')
plt.ylabel('Retorno esperado anualizado')
plt.show()

import cvxopt as opt
# minimize (1/2)*x'*P*x + q'*x
# subject to G*x <= h
# A*x = b.
P = opt.matrix(covmat.values)
q = opt.matrix(np.zeros((len(mu),1)))
G = opt.matrix(-np.identity(len(mu)))
h = opt.matrix(np.zeros((len(mu),1)))
A= opt.matrix(1.0,(1,len(mu)))
b = opt.matrix(1.0)
solution = opt.solvers.qp(P,q,G,h,A,b)
mvg_pond = np.array(solution['x'])
mvg_ret = np.array(mu) * np.array(mvg_pond)
mvg_std = np.sqrt(np.array(mvg_pond).T * np.array(covmat) * np.array(mvg_pond))
print(mvg_ret)
# 0.07779782
print(mvg_std)

plt.bar( ['GGAL','BMA','YPF','TS','MELI'], mvg_pond.flatten())
plt.xticks(rotation='vertical')
plt.title("Distribucion de estos 5 activos")

# Portafolio con retorno target
# actualizamos G, h
ret_tgt = 0.10
G = opt.matrix(np.concatenate((-np.transpose(np.expand_dims(np.array(mu),axis = 1)), -np.identity(len(mu))),0))
h = opt.matrix(np.concatenate((-np.ones((1,1))*ret_tgt,  np.zeros((len(mu),1))),  0))
solution = opt.solvers.qp(P,q,G,h,A,b)
tgt_pond = np.array(solution['x'])
tgt_ret = np.array(mu) * np.array(tgt_pond)
tgt_std = np.sqrt(np.array(tgt_pond).T * np.array(covmat) * np.array(tgt_pond))
print(tgt_ret)
# 0.10000001
print(tgt_std)
#0.29513768

def optimizacion(mean_returns, cov_matrix, ret_tgt):
  opt.solvers.options['show_progress'] = False
  P = opt.matrix(cov_matrix.values)
  q = opt.matrix(np.zeros((len(mean_returns),1)))

  G = opt.matrix(np.concatenate((-np.transpose(np.expand_dims(np.array(mean_returns),axis = 1)), -np.identity(len(mean_returns))),0))
  h = opt.matrix(np.concatenate((-np.ones((1,1))*ret_tgt,  np.zeros((len(mean_returns),1))),  0))
  A= opt.matrix(1.0,(1,len(mean_returns)))
  b = opt.matrix(1.0)

  sol = opt.solvers.qp(P,q,G,h,A,b)['x']
  return sol

def portfolio_metrics2(weights, mean_returns, cov_matrix):
  mean_returns = np.array(mean_returns)
  cov_matrix = np.array(cov_matrix)
  ret = np.sum(mean_returns * weights)
  std = np.asscalar(np.sqrt(np.dot(weights, np.dot(cov_matrix, weights.T))))
  return ret, std

# vector de retornos target
max_ret = np.max(mu)
min_ret = mvg_ret[0,0]
n_port = 100
r_tgts = np.linspace(min_ret, max_ret,n_port)
ponderaciones_list = []
metricas_list = np.zeros((2,n_port))
for i in range(len(r_tgts)):
  ponderacion = optimizacion(mu, covmat, r_tgts[i])
  ponderacion = np.array(ponderacion).T
  ponderaciones_list.append(ponderacion[0].tolist())
  port_mu, port_std = portfolio_metrics2(ponderacion, mu, covmat)
  metricas_list[0,i] = port_mu
  metricas_list[1,i] = port_std
  
  plt.figure(figsize=(16,9))
plt.plot(metrics[1,:], metrics[0,:], 'o')
plt.plot(metricas_list[1,:], metricas_list[0,:], 'y-o')
plt.scatter(sd, mu, marker = 'x', color = 'r')
plt.title('Random Portfolios')
plt.xlabel('Volatilidad anualizada')
plt.ylabel('Retorno esperado anualizado')
plt.show()

