# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

"""
Oscilador Armonico Simple
    
El oscilador armonico es un sistema fisico que consta de movimientos
sinusoidales o sinusoidales amortiguados en torno a un punto de equilibrio.
 
Segun la segunda ley de Newton 

F = ma

en el caso donde el oscilador no tiene perdidas

m d^2y/dt^2 = -ky

entonces la aceleracion toma la forma

d^2y/dt^2 = -(k/m)y

La solucion a esta ecuacion diferencial ordinaria

y = A cos(wt + \phi)
 
donde 
w = Sqrt(k/m)
 
y 

T = 2\piSqrt(m/k)  
"""
#Parametros

m = 1
k = 1
omega = np.sqrt(k/m)

#Funcion para la ecuacion diferencial
def f(y,t):
    x,v = y # y es un vector que cuenta con x como posicion y v como velocidad
    dxdt = v #la derivada de x con respecto a t es la velocidad
    dvdt = -(k/m) * x #La aceleracion tine la forma 
    return [dxdt,dvdt]

#Valores iniciales
y0 = [1.0, 0.0]
t = np.linspace(0, 10, 500)

#Resolucion de EDO
solucion = odeint(f,y0,t)

#Graficacion
plt.plot(t, solucion[:,0], label = 'x(t)') #Grafica de posicion
plt.plot(t, solucion[:,1], label = 'v(t)') #Grafica de velocidad
plt.xlabel('Tiempo(segundos)')
plt.legend()
plt.title('Oscilador armonico')
plt.show()







 


 
 


