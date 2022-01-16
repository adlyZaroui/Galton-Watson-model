# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from math import comb

def simuDiscret(mu,n):
    '''Prend une loi de probabilite discrete mu et un entier naturel n,
       retourne n réalisations independantes de loi mu (P(X = k) = mu[k])'''
       
    processus = []
    for i in range(n):
        u = np.random.uniform()
        k=0
        while u > np.cumsum(mu)[k]:
            k = k + 1
        processus.append(k)
        
    return processus



def simuSomme(mu,n):
    '''prend une loi de probabilité discrète mu et un entier naturel n,
       retourne une réalisation de la variable aleatoire somme de n realisations i.i.d. de loi mu'''

    return sum(simuDiscret(mu,n))


def Za(mu,n):
    '''prends loi de probabilité discrète mu, entier naturel n,
       simule un processus jusqu'à l'instant n-1 et retourne la liste de ses états'''
    
    processus = [1]
    for i in range(n-1):
        X = processus[-1]
        processus.append(simuSomme(mu,X))
    return processus

def pmfb(k,n,p):
    return comb(n,k)*(p**k)*((1-p)**(n-k))

#mu = [1/4 for i in range(4)] ## fonction de masse de la loi uniforme
#[m,n,p] = [[[1,2,3,4,5,6,7,8,9,10],10,0.11]] # nombre de générations, B(n,p)
def reproduction(n,p):
    mu = [pmfb(k,n,p) for k in range(n+1)]
    return mu


#Za(mu,m) ## Vecteur contenant les m générations de Zn dont les Xn sont de loi mu (loi de reproduction)

print(Za(reproduction(10,0.15),25))
#mu est la fonction de masse de la loi binomiale de paramètres (10,0.15)
#On regarde 25 générations d'un processus Zn = sum_(1_Zn)(Xn,i)  
#Les Xn suivent une loi de reproduction B(10,0.15)


Z = np.array([Za(reproduction(10,0.15),25) for i in range(10)])

print(Z) #On regarde 10 trajectoires du processus (Zn) où Xn ~ B(10,0.15)

X = np.arange(1,26)
Y = Za(reproduction(10,0.15),25)
plt.plot(X,Y)
plt.title('Processus (Zn) sur 25 générations; (Zn) ~ B(10,0.15) ')
plt.ylabel('Nb enfants')
plt.xlabel('Z_i')
plt.show()


def Generation(t) :
    G = sum([pmfb(k,10,0.11)*(t**k) for k in range(11)])
    return G

def pointfixe(f,eps):
    u = 0.001
    while abs(u-f(u))>eps : 
        u = f(u)
    return u

### Méthode de Monte-Carlo

### m : espérance de B(n,p)

# def mmcGW(mu,n = 100000):
#     tirages = [x for x in Za(mu,n)]
#     approximation = np.mean(tirages)
#     sigma = np.std(tirages)
#     erreur = 1.96*sigma/np.sqrt(n)
#     b_inf = approximation - erreur
#     b_sup = approximation + erreur
#     return (approximation, sigma, erreur, b_inf, b_sup)

def composite_function(f, n, x): 
    if n == 1: return f(x)

    return composite_function(f,n-1,f(x))


def derivee(f,x):
    h = 0.0001
    return (f(x+h)-f(x))/h


def G(mu,t):
    return sum([mu[i] * (t**i) for i in range(len(mu))])


def simuM(mu,n):
    ev = sum([i*mu[i] for i in range(len(mu))])
    return Za(mu,n)[-1]/(ev**(n-1))

def MC(mu,n,N):
    ''' renvoi une approximation de IE[M_n]'''
    
    m = sum([i*mu[i] for i in range(len(mu))])
    
    G = lambda t: composite_function(lambda x: sum([mu[i] * (x**i) for i in range(len(mu))]), n, t)
    ev = derivee(G,1)/(m**n) #esperance de Mn
    
    tirages = []
    for i in range(N):
        tirages.append(simuM(mu,n))
        
    approximation = np.mean(tirages)
    sigma = np.std(tirages)
    erreur = 1.96*sigma/np.sqrt(N)
    b_inf = approximation - erreur
    b_sup = approximation + erreur
    return (ev, approximation, sigma, erreur, b_inf, b_sup)
        


A = np.zeros((6,20))
for i in range(3,22):
    A[:,i-3] = MC(reproduction(10,0.15),i,10000)


### Simulation des trajectoires de Zn


Ts=100            # Nombre de trajectoires simulées
T=60              # Durée de la simulation
[n,p] =[10,0.09]  # Paramètres de la binomiale
#[n,p]=[10,0.11]

m=n*p             # Espérance de la binomiale
S=0               # S : nombre de processus survivant

for i in range(Ts):
  processus=[1] # On commence avec un individu 
  for t in range(T):
    Z=processus[-1]
    processus.append(np.random.binomial(n*Z,p)) # On simule Zn+1 et on l'ajoute à la liste des processus
  S=S+(processus[-1]>0)   # Si le processus a survécu au moment T, il y a S+1 survivants
  plt.plot(processus, 'purple')                                           


E = Ts - S # Nombre de processus éteints 
Q = E / Ts # Proportion de processus éteints

X=np.arange(1,T+1)
Y=[(m**x) for x in X]
plt.plot(X,Y,'gold',label='Espérance : $(np)^t = '+str(np.round(m,2))+'^t$')
plt.title(''+str(S)+' trajectoires de $(Z_n)$ non-éteintes, Binom('+str(n)+','+str(p)+')')
plt.xlabel(''+str(100 - Q*100)+' % des '+str(Ts)+' processus ont réussi à survivre ('+str(S)+' survivants , '+str(E)+' éteints) ')
plt.legend()
plt.show()






### Simulation des trajectoires de la martingale Z_n/m


Ts=100          # Nombre de trajectoires simulées
T=60            # Durée de la simulation
[n,p]=[10,0.11] # Paramètres de la binomiale
#[n,p]=[10,0.11]

m=n*p          # Espérance de la binomiale
S=0            # S : nombre de processus survivant


Y=np.asarray([1/(m**x) for x in np.arange(0,T+1)])  # Crée le vecteur [1/m 1/m^2 1/m^3 ...]

for i in range(Ts):
  processus=[1]
  for t in range(T):
    Z=processus[-1]
    processus.append(np.random.binomial(n*Z,p)) # On simule Zn+1 et on l'ajoute à la liste des processus
  S=S+(processus[-1]>0)   # On ajoute 1 si le processus a survécu à l'instant T
  plt.plot(processus*Y)   # On plot le vecteur [Z_n/m Z_n/m^2 Z_n/m^3 ...]                                      

E = Ts - S # Nombre de processus éteints 
Q = E / Ts # Proportion de processus éteints

plt.title(''+str(S)+' trajectoires de $Z_n/m^n$, Binom('+str(n)+','+str(p)+')')
plt.xlabel(''+str(100 - Q*100)+' % des '+str(Ts)+' processus ont réussi à survivre ('+str(S)+' survivants , '+str(E)+' éteints) ')
plt.show()
print(''+str(100 - Q*100)+' % des '+str(Ts)+' processus ont réussi à survivre ('+str(S)+' survivants , '+str(E)+' éteints) ')


