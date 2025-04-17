import numpy as np
from molmod.units import angstrom
# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2022-05-05
# Update: 2024-09-24

def translationFT(kx,ky,kz,a):
    return np.exp(1.0j*(kx*a[0]+ky*a[1]+kz*a[2]))

" The weight functions of the FMT functional"

def w3FT(k,sigma=1.0):
    output = 1.0-(k*0.5*sigma)**2/10
    mask = k*sigma>1e-6
    output[mask] = 3*(np.sin(k[mask]*0.5*sigma)-0.5*k[mask]*sigma*np.cos(k[mask]*0.5*sigma))/(k[mask]*0.5*sigma)**3
    return (np.pi*sigma**3/6)*output

def w2FT(k,sigma=1.0):
    return np.pi*sigma**2*np.sinc(0.5*sigma*k/np.pi)

" The Auxiliary functions of the FMT eos/functional"

## phi1(n3)*n0
def phi1func(eta):
   return -np.log(1-eta)

def dphi1dnfunc(eta):
    return 1/(1-eta)

## phi2(n3)*(n1*n2-n1vec*n2vec)
def phi2func(eta,model='WBI'):
    if model == 'RF' or model == 'WBI' or model == 'aRF' or model == 'aWBI':
        return 1/(1-eta)
    elif model == 'WBII' or model == 'aWBII':
        return np.where(eta<=1e-8,(1+ eta**2/9)/(1-eta), ((5 - eta)*eta + 2*(1-eta)*np.log(1-eta))/(3*eta*(1-eta)))
    
def dphi2dnfunc(eta,model='WBI'):
    if model == 'RF' or model == 'WBI' or model == 'aRF' or model == 'aWBI':
        return 1/(1-eta)**2
    elif model == 'WBII' or model == 'aWBII':
        return np.where(eta<=1e-8,(1+ 2*eta/9 + eta**2/18)/(1-eta)**2,-2*(eta - 3*eta**2 + (1-eta)**2*np.log(1-eta))/(3*eta**2*(1-eta)**2))

## phi3(n3)*(n2**3-3*n2*n2vec*n2vec)  
def phi3func(eta,model='WBI'):
    if model == 'RF' or model == 'aRF':
        return 1/(24*np.pi*(1-eta)**2)
    elif model == 'WBI' or model == 'aWBI':
        return np.where(eta<=1e-8,(1.0-2*eta/9-eta**2/18)/(24*np.pi*(1-eta)**2),(eta+(1-eta)**2*np.log(1-eta))/(36*np.pi*eta**2*(1-eta)**2))
    elif model == 'WBII' or model == 'aWBII':
        return np.where(eta<=1e-8,(1-4*eta/9+eta**2/18)/(24*np.pi*(1-eta)**2),-2*(eta + (eta-3)*eta**2+np.log(1-eta)*(1-eta)**2)/((3*eta**2)*24*np.pi*(1-eta)**2))
    
def dphi3dnfunc(eta,model='WBI'):
    if model == 'RF' or model == 'aRF':
        return 1/(12*np.pi*(1-eta)**3)
    elif model == 'WBI' or model == 'aWBI':
        return np.where(eta<=1e-8,(8/3-0.5*eta-0.1*eta**2)/(36*np.pi*(1-eta)**3),-(eta*(2-5*eta+eta**2)+2*(1-eta)**3*np.log(1-eta))/(36*np.pi*eta**3*(1-eta)**3))
    elif model == 'WBII' or model == 'aWBII':
        return np.where(eta<=1e-8,(7/3-eta/2+eta**2/10)/(36*np.pi*(1-eta)**3),(2*eta-5*eta**2+6*eta**3-eta**4 + 2*(1-eta)**3*np.log(1-eta))/(36*np.pi*eta**3*(1-eta)**3))


" The Two Yukawa representation of LJ potential"

def YKFT(k,l,sigma=1.0):
    output = np.full_like(k, (1+l)/l**2)
    mask = k>1e-6
    output[mask] = (k[mask]*sigma*np.cos(k[mask]*sigma)+l*np.sin(k[mask]*sigma))/(k[mask]*sigma*(l**2+(k[mask]*sigma)**2))
    return 4*np.pi*sigma**3*output



def YKcutoffFT(k,l,rc=5.0,sigma=1.0):
    output = np.full_like(k, (1+l*rc/sigma)/l**2)
    mask = k>1e-6
    output[mask] = (k[mask]*sigma*np.cos(k[mask]*rc)+l*np.sin(k[mask]*rc))/((k[mask]*sigma)**3+k[mask]*sigma*l**2)
    return YKFT(k,l,sigma=sigma) - 4*np.pi*sigma**3*np.exp(l-l*rc/sigma)*output

def lj3dFT(k,sigma,epsilon,cutoff=None,model='BH'):
    if model == 'BH':
        r0 = sigma
        l = [2.544944560171331,15.464088962136259]
        # eps = [-1.8577081618771705*epsilon,1.8577081618771705*epsilon]
        eps = [-1.8577081618771705*epsilon,1.8577081618771705*epsilon]
    elif model == 'WCA':
        r0 = 2**(1/6)*sigma
        l = [2.771306422399706,29.424387814010043]
        eps = [-1.141496075706534*epsilon,0.141496075706534*epsilon]
        # eps = [-1.141496075706534*epsilon,0.141496075706534*epsilon]

    if cutoff == None:
        return eps[0]*YKFT(k,l[0],sigma=r0)+eps[1]*YKFT(k,l[1],sigma=r0) + (eps[0]+eps[1])*w3FT(k,sigma=2*r0)
    else:
        return eps[0]*YKcutoffFT(k,l[0],rc=cutoff,sigma=r0) + eps[1]*YKcutoffFT(k,l[1],rc=cutoff,sigma=r0) + (eps[0]+eps[1])*w3FT(k,sigma=2*r0)

    