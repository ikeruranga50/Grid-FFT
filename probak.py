#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from scipy.fft import fft, fftfreq
from scipy.signal import argrelextrema


# In[18]:


def DFT(x,N):
    xt=np.zeros(N)
    xf=np.zeros(N)
    for k in range(N):
        for i in range(N):
            xt[i]=x[i]*np.exp(-2*np.pi*1j*k*i/N)
        xf[k]=sum(xt)
    return xf
def RadialMeshGenerator(pointDis,MeshSize,ratio):
     # number of bins in r and theta dimensions
    N_bins_theta = round(MeshSize/(np.sqrt(2**3)*pointDis))
    N_bins_r = round(MeshSize/(np.sqrt(2**3)*pointDis))

    # limits in r dimension
    rmin = 0
    rmax = MeshSize/2

    # setting up 1D arrays in r and theta
    pausur=(rmax)/N_bins_r
    up=pointDis/np.sqrt(2)*ratio
    r=np.zeros(N_bins_r)
    for i in range(N_bins_r):
        r[i]=pausur*i+up*i**2
     
    theta = np.linspace(0, 2*np.pi, N_bins_theta*3)  # N.B. radians not degrees

    # 'gridding' the 1D arrays into 2D arrays so they can be used by pcolor
    theta, r = np.meshgrid(theta, r)
    return theta,r
    

def get_frecuency(xf,ite):
    erroak=[]
    dif=0
    # for local minima
    minima=argrelextrema(xf,np.less)[0]
    for inde in minima:
            erroak.append(ite[inde])
            for i in range(len(erroak)-1):
                dif+=erroak[i+1]-erroak[i]
    return dif/len(erroak)


def MeshGenerator(pointDis,MeshSize):
    
    pausux=pointDis
    pausuy=pointDis
    numx=round(MeshSize/pointDis)
    numy=round(MeshSize/pointDis)
    if(np.mod(numx,2)==0):
        numx+=1
    if(np.mod(numy,2)==0):
        numy+=1
    x=np.zeros(numx*numy)
    y=np.zeros(numy*numy)
    for i in range(numx):
        for j in range(numy):
            x[j+numx*i]=pausux*i-MeshSize/2
            y[j+numy*i]=pausuy*j-MeshSize/2
    fig,plot=plt.subplots()
    plot.set_title("Mesh")
    plot.scatter(x,y)
#Setting grid physical parameters
pointDis=1.25
MeshSize=35


#Generate diverging radial mesh
theta05,r05=RadialMeshGenerator(pointDis,MeshSize,ratio=0.1)
#Generate equidistant radial mesh
theta0,r0=RadialMeshGenerator(pointDis,MeshSize,ratio=0)


#FFT of diverging mesh
r=np.zeros(len(r05))
for i in range(len(r05)):
    r[i]=r05[i][0]
zeroak=100
rexp05=np.zeros(len(r)+zeroak)
a=slice(int(zeroak/2),-int(zeroak/2))
rexp05[a]=r
rf05=fft(rexp05)
r05frek=np.arange(0,len(rf05))
fig2,ax2=plt.subplots()
ax2.set_xlabel("angle(rad)")
ax2.set_ylabel("FFT(r05)(2π/mm)")
ax2.set_title("Fast Fourier Transform of radially diverging points")
ax2.scatter(r05frek,np.abs(rf05))
savefig=("simul_radial_div.png")


#FFT of equidistant mesh
r=np.zeros(len(r0))
for i in range(len(r0)):
    r[i]=r0[i][0]
rexp0=np.zeros(len(r)+zeroak)
rexp0[a]=r
rf0=fft(rexp0)
r0frek=np.arange(0,len(rf0))
fig3,ax3=plt.subplots()
ax3.set_xlabel("angle(rad)")
ax3.set_ylabel("FFT(r0)(2π/mm)")
ax3.set_title("Fast Fourier Transform of radially equidistant points")
ax3.scatter(r0frek,np.abs(rf0))
savefig=("simul_radial.png")

# DFT of diverging mesh
r=np.zeros(len(r05))
rdis=np.zeros(len(r05)-1)
for i in range(len(r05)):
    r[i]=r05[i][0]
for i in range(len(r)-1):
    rdis[i]=r[i+1]-r[i]
zero=100
rdisx=np.zeros(len(rdis)+zero)
if(zero!=0):
    a=slice(int(zero/2),-int(zero/2))
    rdisx[a]=rdis
else:
    rdisx=rdis
rf05=DFT(rdisx,len(rdisx))
rfrek05=np.arange(len(rf05))
fig3,plot3=plt.subplots()
plot3.set_xlabel("angle(rad)")
plot3.set_ylabel("DFT(r05)(2π/mm)")
plot3.set_title("DFT of diverging mesh")
plot3.scatter(rfrek05,np.abs(rf05))
print(get_frecuency(rf05,rfrek05))


#DFT of equidistant mesh
r=np.zeros(len(r0))
rdis=np.zeros(len(r0)-1)
for i in range(len(r05)):
    r[i]=r0[i][0]
for i in range(len(r)-1):
    rdis[i]=r[i+1]-r[i]
rdisx=np.zeros(len(rdis)+zero)
if(zero!=0):
    a=slice(int(zero/2),-int(zero/2))
    rdisx[a]=rdis
else:
    rdisx=rdis
rf0=DFT(rdisx,len(rdisx))
rfrek0=np.arange(len(rf0))
fig3,plot3=plt.subplots()
plot3.set_xlabel("angle(rad)")
plot3.set_ylabel("DFT(r0)(2π/mm)")
plot3.set_title("DFT of equidistant mesh")
plot3.scatter(rfrek0,np.abs(rf0))
print(get_frecuency(rf0,rfrek0))


# In[ ]:





# In[ ]:




