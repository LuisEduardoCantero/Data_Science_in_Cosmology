#!/usr/bin/env python
from __future__ import absolute_import, unicode_literals, print_function
import numpy as np
from numpy import pi, cos
from pymultinest.solve import solve
import os
if not os.path.exists("chains"): os.mkdir("chains")

# probability function, taken from the eggbox problem.

data = np.load('fit_ejercise.npy')

def line(x, m, b):
    return(m * x + b)

def lnlh(theta, data):
    modelo=line(data[0], theta[0],theta[1]) # Se evalua el modelo en todos los datos independientes o 'x' del experimento
    diferencia2 = (data[1] - modelo)**2 # Se compara el modelo con los datos y se obtiene el cuadrado de la diferencia
    sigma2 = data[2] ** 2 # Se obtiene sigma^2 (vector)
    u = -0.5 * np.sum(diferencia2/sigma2) #Se multiplica por -1/2 y se suman los valores
    return u


def myprior(cube):
    cubo[0]=cube * -2
    cubo[1]=cube  * -10
    cubo[2]=6*cube[2]-5
    return cube * 10 * pi

def myloglike(cube):
    loglike=lnlh(cube, data)
    return loglike



# number of dimensions our problem has
parameters = ["x", "y"]
n_params = len(parameters)
# name of the output files
prefix = "chains/3-"

# run MultiNest
result = solve(LogLikelihood=myloglike, Prior=myprior, 
	n_dims=n_params, outputfiles_basename=prefix, verbose=True)

print()
#print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
print()
print('parameter values:')
for name, col in zip(parameters, result['samples'].transpose()):
	print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
    
# lets analyse the results
a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename=prefix)
s = a.get_stats()
    

# make marginal plots by running:
# $ python multinest_marginals.py chains/3-
# For that, we need to store the parameter names:
import json
with open('%sparams.json' % prefix, 'w') as f:
	json.dump(parameters, f, indent=2)

with open('%sstats.json' %  a.outputfiles_basename, mode='w') as f:
        json.dump(s, f, indent=2)
print()
print("-" * 30, 'ANALYSIS', "-" * 30)
print("Global Evidence:\n\t%.15e +- %.15e" % ( s['nested sampling global log-evidence'], s['nested sampling global log-evidence error'] ))
