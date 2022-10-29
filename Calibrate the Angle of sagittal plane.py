#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#法向量作图
from vedo import *
embedWindow(False) #to pop an external VTK rendering window
from random import gauss, uniform as u
import numpy as np
import math
import pyquaternion
vp = Plotter()
Santina=[LP,LA,LH,RP,RA,RH]=[[0.69611, 0.6682, 0.26257],[0.58954, -0.7875, 0.17971],[0.31645, 0.04108, -0.94772],[0.69248, -0.66561, 0.27826],[0.58905, 0.78928, 0.17334],[0.32891, -0.03566, -0.94369]]



AP=91.59/180*np.pi
PL=89.57/180*np.pi
AL=85.56/180*np.pi

LA_RP=15.5/180*np.pi
LP_RA=15.5/180*np.pi

#设定RL和LL位于水平面，关于矢状面对称，
RL_LL=11.7/180*np.pi
RL=[-np.sin(RL_LL/2),np.cos(RL_LL/2),0]
#RL=RL/np.linalg.norm(RL)
LL=[np.sin(RL_LL/2),np.cos(RL_LL/2),0]


## 架构三棱锥，OL为X轴，OP位于XY平面
OL=[1,0,0]
OP=[np.cos(PL),np.sin(PL),0]
## 设OA=[x,y,z],则
x=np.cos(AL)
y=(np.cos(AP)-x*np.cos(PL))/np.sin(PL)
z=(1-x**2-y**2)**0.5
OA=[x,y,z]

## 然后旋转OL到LL,再绕LL旋转使得LA_RP差值小于0.001°

a=(90-11.7/2)/180*np.pi

# Create another quaternion representing no rotation at all
quaternion = pyquaternion.Quaternion(axis=[0, 0, 1], angle=a)


## 新的矢量
OA=quaternion.rotate(OA) 
OP=quaternion.rotate(OP)
OL=quaternion.rotate(OL)
## 绕后绕OL转动角度，达到目标





for i in range(36000):
    X=i/18000*np.pi
    quaternion = pyquaternion.Quaternion(axis=OL, angle=X)
    loa=quaternion.rotate(OA) 
    lop=quaternion.rotate(OP)
    lol=quaternion.rotate(OL)
    roa=[-loa[0],loa[1],loa[2]]
    roa_lop=np.arccos(np.dot(roa,lop))*180
    #print(i,roa_lop,np.dot(roa,lop))
    if abs(roa_lop-15.5)<0.01:
        print(i,roa_lop)
        print(roa_lop-15.5)
        break
rop=[-lop[0],lop[1],lop[2]]
rol=[-lol[0],lol[1],lol[2]]


quaternion = pyquaternion.Quaternion(axis=[0, 0, 1], angle=-90/180*np.pi)

loa=quaternion.rotate(loa) 
lop=quaternion.rotate(lop) 
lol=quaternion.rotate(lol) 
roa=quaternion.rotate(roa) 
rop=quaternion.rotate(rop) 
rol=quaternion.rotate(rol) 


 






vp +=Arrow([0, 0, 0], loa).color('green')

#vp +=Arrow([0, 0, 0], roa).color('yellow')


vp +=Arrow([0, 0, 0], lop).color('blue')
                       
#vp +=Arrow([0, 0, 0], rop).color('5')

vp +=Arrow([0, 0, 0], lol).color('red')


#vp +=Arrow([0, 0, 0], rol).color('red')





vp.show(axes=1)  # render the internal list of objects in vp.actors
vp.close()


# In[2]:


get_ipython().system('pip install vedo')


# In[2]:


get_ipython().system('pip install pyquaternion')


# In[ ]:


#法向量作图
from vedo import *
embedWindow(False) #to pop an external VTK rendering window
from random import gauss, uniform as u
import numpy as np
import math
import pyquaternion
vp = Plotter()
Santina=[LP,LA,LH,RP,RA,RH]=[[0.69611, 0.6682, 0.26257],[0.58954, -0.7875, 0.17971],[0.31645, 0.04108, -0.94772],[0.69248, -0.66561, 0.27826],[0.58905, 0.78928, 0.17334],[0.32891, -0.03566, -0.94369]]



AP=91.59/180*np.pi
PL=89.57/180*np.pi
AL=85.56/180*np.pi

LA_RP=15.5/180*np.pi
LP_RA=15.5/180*np.pi

#设定RL和LL位于水平面，关于矢状面对称，
RL_LL=11.7/180*np.pi
RL=[-np.sin(RL_LL/2),np.cos(RL_LL/2),0]
#RL=RL/np.linalg.norm(RL)
LL=[np.sin(RL_LL/2),np.cos(RL_LL/2),0]


## 架构三棱锥，OL为X轴，OP位于XY平面
OL=[1,0,0]
OP=[np.cos(PL),np.sin(PL),0]
## 设OA=[x,y,z],则
x=np.cos(AL)
y=(np.cos(AP)-x*np.cos(PL))/np.sin(PL)
z=(1-x**2-y**2)**0.5
OA=[x,y,z]

## 然后旋转OL到LL,再绕LL旋转使得LA_RP差值小于0.001°

a=(90-11.7/2)/180*np.pi

# Create another quaternion representing no rotation at all
quaternion = pyquaternion.Quaternion(axis=[0, 0, 1], angle=a)


## 新的矢量
OA=quaternion.rotate(OA) 
OP=quaternion.rotate(OP)
OL=quaternion.rotate(OL)
## 绕后绕OL转动角度，达到目标





for i in range(36000):
    X=i/18000*np.pi
    quaternion = pyquaternion.Quaternion(axis=OL, angle=X)
    loa=quaternion.rotate(OA) 
    lop=quaternion.rotate(OP)
    lol=quaternion.rotate(OL)
    roa=[-loa[0],loa[1],loa[2]]
    roa_lop=np.arccos(np.dot(roa,lop))*180
    #print(i,roa_lop,np.dot(roa,lop))
    if abs(roa_lop-15.5)<0.01:
        print(i,roa_lop)
        print(roa_lop-15.5)
        break
rop=[-lop[0],lop[1],lop[2]]
rol=[-lol[0],lol[1],lol[2]]


quaternion = pyquaternion.Quaternion(axis=[0, 0, 1], angle=-90/180*np.pi)

loa=quaternion.rotate(loa) 
lop=quaternion.rotate(lop) 
lol=quaternion.rotate(lol) 
roa=quaternion.rotate(roa) 
rop=quaternion.rotate(rop) 
rol=quaternion.rotate(rol) 


 






vp +=Arrow([0, 0, 0], loa).color('green')

#vp +=Arrow([0, 0, 0], roa).color('yellow')


vp +=Arrow([0, 0, 0], lop).color('blue')
                       
#vp +=Arrow([0, 0, 0], rop).color('5')

vp +=Arrow([0, 0, 0], lol).color('red')


#vp +=Arrow([0, 0, 0], rol).color('red')





vp.show(axes=1)  # render the internal list of objects in vp.actors
vp.close()


# In[15]:


#法向量作图
from vedo import *
embedWindow(False) #to pop an external VTK rendering window
from random import gauss, uniform as u
import numpy as np
import math
import pyquaternion
vp = Plotter()


AP=91.59/180*np.pi
PL=89.57/180*np.pi
AL=85.56/180*np.pi

LA_RP=15.5/180*np.pi
LP_RA=15.5/180*np.pi

#设定RL和LL位于水平面，关于矢状面对称，
RL_LL=11.7/180*np.pi
RL=[-np.sin(RL_LL/2),np.cos(RL_LL/2),0]
#RL=RL/np.linalg.norm(RL)
LL=[np.sin(RL_LL/2),np.cos(RL_LL/2),0]


## 架构三棱锥，OL为X轴，OP位于XY平面
OL=[1,0,0]
OP=[np.cos(PL),np.sin(PL),0]
## 设OA=[x,y,z],则
x=np.cos(AL)
y=(np.cos(AP)-x*np.cos(PL))/np.sin(PL)
z=(1-x**2-y**2)**0.5
OA=[x,y,z]

## 然后旋转OL到LL,再绕LL旋转使得LA_RP差值小于0.001°

a=(90-11.7/2)/180*np.pi

# Create another quaternion representing no rotation at all
quaternion = pyquaternion.Quaternion(axis=[0, 0, 1], angle=a)


## 新的矢量
OA=quaternion.rotate(OA) 
OP=quaternion.rotate(OP)
OL=quaternion.rotate(OL)
## 绕后绕OL转动角度，达到目标





for i in range(36000):
    X=i/18000*np.pi
    quaternion = pyquaternion.Quaternion(axis=OL, angle=X)
    loa=quaternion.rotate(OA) 
    lop=quaternion.rotate(OP)
    lol=quaternion.rotate(OL)
    roa=[-loa[0],loa[1],loa[2]]
    roa_lop=np.arccos(np.dot(roa,lop))*180
    #print(i,roa_lop,np.dot(roa,lop))
    if abs(roa_lop-15.5)<0.01:
        print(i,roa_lop)
        print(roa_lop-15.5)
        break
rop=[-lop[0],lop[1],lop[2]]
rol=[-lol[0],lol[1],lol[2]]


quaternion = pyquaternion.Quaternion(axis=[0, 0, 1], angle=-90/180*np.pi)

loa=quaternion.rotate(loa) 
lop=quaternion.rotate(lop) 
lol=quaternion.rotate(lol) 
roa=quaternion.rotate(roa) 
rop=quaternion.rotate(rop) 
rol=quaternion.rotate(rol) 


## 然后绕Y轴旋转，使得对应点的距离之和最小。
k=1000
roll=0
for i in range(3600):
    X=i/1800*np.pi
    quaternion = pyquaternion.Quaternion(axis=[0,1,0], angle=X)
    la=quaternion.rotate(loa) 
    lp=quaternion.rotate(lop)
    lh=quaternion.rotate(lol)
    ra=quaternion.rotate(roa) 
    rp=quaternion.rotate(rop)
    rh=quaternion.rotate(rol)
    this=np.array([lp,la,lh,rp,ra,rh])
    m=[np.linalg.norm(Santina[i]-this[i]) for i in range(6)]
    diff=sum(m)
    print(i,sum(m))
      
    if k> diff:
        k=diff
        roll=i
        
        
quaternion = pyquaternion.Quaternion(axis=[0, 1, 0], angle=roll/1800*np.pi)

loa=quaternion.rotate(loa) 
lop=quaternion.rotate(lop) 
lol=quaternion.rotate(lol) 
roa=quaternion.rotate(roa) 
rop=quaternion.rotate(rop) 
rol=quaternion.rotate(rol) 
        


vp +=Arrow([0, 0, 0], loa).color('green')

vp +=Arrow([0, 0, 0], roa).color('yellow')


vp +=Arrow([0, 0, 0], lop).color('blue')
                       
vp +=Arrow([0, 0, 0], rop).color('5')

vp +=Arrow([0, 0, 0], lol).color('red')


vp +=Arrow([0, 0, 0], rol).color('red')






vp +=Line([0, 0, 0], LA).color('green')

vp +=Line([0, 0, 0], LP).color('blue')

vp +=Line([0, 0, 0], LH).color('red')
    
    
vp +=Line([0, 0, 0], RA).color('green')

vp +=Line([0, 0, 0], RP).color('blue')

vp +=Line([0, 0, 0], RH).color('red')



vp.show(axes=1)  # render the internal list of objects in vp.actors
vp.close()


# In[17]:


print(lop,loa,lol,rop,roa,rol)


# In[13]:


print(Santina)


# In[16]:


print(lop,LP)


# In[ ]:


#法向量作图
from vedo import *
embedWindow(False) #to pop an external VTK rendering window
from random import gauss, uniform as u
import numpy as np
import math
import pyquaternion
vp = Plotter()

## 左侧
AP=97.24/180*np.pi
PL=90.03/180*np.pi
AL=89.51/180*np.pi

## 架构三棱锥，OL为X轴，OP位于XY平面
OL=[1,0,0]
OP=[np.cos(PL),np.sin(PL),0]
## 设OA=[x,y,z],则
x=np.cos(AL)
y=(np.cos(AP)-x*np.cos(PL))/np.sin(PL)
z=(1-x**2-y**2)**0.5
OA=[x,y,z]




LA_RP=19/180*np.pi
LP_RA=17.2/180*np.pi

#设定RL和LL位于水平面，关于矢状面对称，
RL_LL=9.8/180*np.pi
 
## 然后旋转OL到LL,再绕LL旋转使得LA_RP差值小于0.001°

a=(90-9.8/2)/180*np.pi

# Create another quaternion representing no rotation at all
quaternion = pyquaternion.Quaternion(axis=[0, 0, 1], angle=a)


## 新的矢量
OA=quaternion.rotate(OA) 
OP=quaternion.rotate(OP)
OL=quaternion.rotate(OL)
## 绕后绕OL转动角度，达到目标





for i in range(36000):
    X=i/18000*np.pi
    quaternion = pyquaternion.Quaternion(axis=OL, angle=X)
    loa=quaternion.rotate(OA) 
    lop=quaternion.rotate(OP)
    lol=quaternion.rotate(OL)
    roa=[-loa[0],loa[1],loa[2]]
    roa_lop=np.arccos(np.dot(roa,lop))*180
    #print(i,roa_lop,np.dot(roa,lop))
    if abs(roa_lop-15.5)<0.01:
        print(i,roa_lop)
        print(roa_lop-15.5)
        break
rop=[-lop[0],lop[1],lop[2]]
rol=[-lol[0],lol[1],lol[2]]


quaternion = pyquaternion.Quaternion(axis=[0, 0, 1], angle=-90/180*np.pi)

loa=quaternion.rotate(loa) 
lop=quaternion.rotate(lop) 
lol=quaternion.rotate(lol) 
roa=quaternion.rotate(roa) 
rop=quaternion.rotate(rop) 
rol=quaternion.rotate(rol) 


## 然后绕Y轴旋转，使得对应点的距离之和最小。
k=1000
roll=0
for i in range(3600):
    X=i/1800*np.pi
    quaternion = pyquaternion.Quaternion(axis=[0,1,0], angle=X)
    la=quaternion.rotate(loa) 
    lp=quaternion.rotate(lop)
    lh=quaternion.rotate(lol)
    ra=quaternion.rotate(roa) 
    rp=quaternion.rotate(rop)
    rh=quaternion.rotate(rol)
    this=np.array([lp,la,lh,rp,ra,rh])
    m=[np.linalg.norm(Santina[i]-this[i]) for i in range(6)]
    diff=sum(m)
    print(i,sum(m))
      
    if k> diff:
        k=diff
        roll=i
        
        
quaternion = pyquaternion.Quaternion(axis=[0, 1, 0], angle=roll/1800*np.pi)

loa=quaternion.rotate(loa) 
lop=quaternion.rotate(lop) 
lol=quaternion.rotate(lol) 
roa=quaternion.rotate(roa) 
rop=quaternion.rotate(rop) 
rol=quaternion.rotate(rol) 
        


vp +=Arrow([0, 0, 0], loa).color('green')

vp +=Arrow([0, 0, 0], roa).color('yellow')


vp +=Arrow([0, 0, 0], lop).color('blue')
                       
vp +=Arrow([0, 0, 0], rop).color('5')

vp +=Arrow([0, 0, 0], lol).color('red')


vp +=Arrow([0, 0, 0], rol).color('red')






vp +=Line([0, 0, 0], LA).color('green')

vp +=Line([0, 0, 0], LP).color('blue')

vp +=Line([0, 0, 0], LH).color('red')
    
    
vp +=Line([0, 0, 0], RA).color('green')

vp +=Line([0, 0, 0], RP).color('blue')

vp +=Line([0, 0, 0], RH).color('red')



vp.show(axes=1)  # render the internal list of objects in vp.actors
vp.close()

