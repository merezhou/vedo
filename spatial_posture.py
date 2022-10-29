#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
total=[[[0.763, 0.566, 0.307],
  [0.757, 0.561, 0.32],
  [0.66, 0.702, 0.266],
  [0.709, 0.652, 0.268],
  [0.69611, 0.6682, 0.26257],
  [0.74, 0.64, 0.14],
  [0.684648424926134, 0.6749678399737521, 0.2750908018224727],
  [0.5991025624663624, 0.7877614874026744, 0.1432060006122813],
  [0.6975921035395724, 0.6808986338940329, 0.22302983531463555],
  [0.6748580001329091, 0.6961126351276784, 0.24493647925984874],
  [0.6949667103076324, 0.6803091426958685, 0.23281052795911075],
  [0.6918295621492893, 0.6866973237931421, 0.2232009015027919],
  [0.7405810890875133, 0.6497079613417431, 0.1715203062471295],
  [0.6036112216790016, 0.7826156180540872, 0.1522047549881467]],
 [[0.462, -0.829, 0.306],
  [0.652, -0.753, -0.017],
  [-0.739, 0.588, 0.329],
  [0.618, -0.773, 0.268],
  [0.58954, -0.7875, 0.17971],
  [0.54, -0.83, 0.04],
  [0.6109380889638016, -0.7617983132108493, 0.2154483266176876],
  [0.5065069225200065, -0.858612680339132, 0.07896203264964491],
  [0.6026758241093872, -0.7657775790923713, 0.22442493264674654],
  [0.5688488812220542, -0.7755248741629223, 0.27381037213188486],
  [0.5937619088044377, -0.7634937680737625, 0.25401586912127894],
  [0.5883968250110697, -0.7693398838105712, 0.24880779629109126],
  [0.6363754427166535, -0.7399472997766202, 0.21795478765210077],
  [0.48054623874081426, -0.8532963675744316, 0.20238730571934613]],
 [[0.412, -0.115, -0.896],
  [0.365, -0.158, -0.905],
  [-0.025, 0.279, -0.96],
  [0.354, 0.011, -0.935],
  [0.31645, 0.04108, -0.94772],
  [0.15, 0.04, -0.98],
  [0.3372602857004155, -0.12163731157385778, -0.9335201466076444],
  [0.4438784916935308, -0.13671037607201297, -0.8855970628260489],
  [0.3684921352784543, -0.07846800555547459, -0.9263132938385755],
  [0.3108801254411885, -0.05522397893634291, -0.9488434326885069],
  [0.34265730628177826, -0.06208292833553452, -0.9374068916223095],
  [0.34947933992779523, -0.059657332562775355, -0.9350428833134484],
  [0.42683458597005036, -0.04722141438118318, -0.9030959939251292],
  [0.36668762256186105, -0.04862134202713547, -0.9290727380347622]],
 [[0.763, -0.566, 0.307],
  [0.757, -0.561, 0.32],
  [-0.651, 0.702, 0.287],
  [-0.709, 0.652, -0.268],
  [0.69248, -0.66561, 0.27826],
  [0.77, -0.61, 0.14],
  [0.6850313001308215, -0.6653434061281155, 0.29673265705499047],
  [0.6026815588237138, -0.7806655785889308, 0.16533660532449618],
  [0.6937018946397963, -0.673509473185787, 0.25526980021569606],
  [0.6758774297173016, -0.6870317285531075, 0.26679037456411914],
  [0.7036908708098086, -0.662638639197429, 0.2563770508089845],
  [0.6932135825476716, -0.6777414146115746, 0.24519686762217127],
  [0.7427135628711982, -0.640961200623954, 0.19376610338722736],
  [0.6069732819341371, -0.7753600427894537, 0.17435664330227907]],
 [[0.462, 0.829, 0.306],
  [0.652, 0.753, -0.017],
  [0.749, 0.577, 0.324],
  [-0.618, -0.773, -0.142],
  [0.58905, 0.78928, 0.17334],
  [0.57, 0.82, 0.04],
  [0.6018829483154949, 0.7696958036006248, 0.21285038512212145],
  [0.49813120823450674, 0.863799639040158, 0.07560081333506219],
  [0.5904284226397649, 0.775275295042969, 0.224371332026081],
  [0.5577125731332506, 0.7839510975293902, 0.2727221341430743],
  [0.5887880645083642, 0.763985279211711, 0.263922542122302],
  [0.5777151018201632, 0.7776107461955467, 0.24812655748654436],
  [0.626374817441819, 0.7482877956999187, 0.21844899377494115],
  [0.4699881115295397, 0.8600351892192188, 0.19862187272694448]],
 [[0.412, 0.115, -0.896],
  [0.365, 0.158, -0.905],
  [0.017, 0.299, -0.954],
  [-0.354, 0.011, 0.935],
  [0.32891, -0.03566, -0.94369],
  [0.13, 0.02, -0.98],
  [0.3543379142756686, 0.11316150123814031, -0.9282451815895948],
  [0.45870843085179797, 0.12765039445199503, -0.8793702020547041],
  [0.3752296321195927, 0.07518282217170083, -0.9238778417245941],
  [0.3273040493830935, 0.04431829277651186, -0.9438792021137054],
  [0.35594104878093125, 0.044492708562851584, -0.9334486427637418],
  [0.3655653774364688, 0.04927843793873805, -0.9294802797122974],
  [0.44230337043905893, 0.037993547997990235, -0.8960603879252627],
  [0.3830513643923475, 0.03831373929725498, -0.9229321262250122]]]

totaldo=copy.deepcopy(total)


# In[2]:


len(total)


# In[7]:


LP = [[0.763, 0.566, 0.307], [0.757, 0.561, 0.32], [0.66, 0.702, 0.266], [0.709, 0.652, 0.268], [0.69611, 0.6682, 0.26257], [0.74, 0.64, 0.14], [0.684648424926134, 0.6749678399737521, 0.2750908018224727], [0.5991025624663624, 0.7877614874026744, 0.1432060006122813], [0.6975921035395724, 0.6808986338940329, 0.22302983531463555], [0.6748580001329091, 0.6961126351276784, 0.24493647925984874], [0.6949667103076324, 0.6803091426958685, 0.23281052795911075], [0.6918295621492893, 0.6866973237931421, 0.2232009015027919], [0.7405810890875133, 0.6497079613417431, 0.1715203062471295], [0.6036112216790016, 0.7826156180540872, 0.1522047549881467]]
len(LP)


# In[6]:


title=['LP','LA','LH','RP','RA','RH']

for i in range(len(total)):
    print(title[i],'=',total[i])


# In[8]:


LP = [[0.763, 0.566, 0.307], [0.757, 0.561, 0.32], [0.66, 0.702, 0.266], [0.709, 0.652, 0.268], [0.69611, 0.6682, 0.26257], [0.74, 0.64, 0.14], [0.684648424926134, 0.6749678399737521, 0.2750908018224727], [0.5991025624663624, 0.7877614874026744, 0.1432060006122813], [0.6975921035395724, 0.6808986338940329, 0.22302983531463555], [0.6748580001329091, 0.6961126351276784, 0.24493647925984874], [0.6949667103076324, 0.6803091426958685, 0.23281052795911075], [0.6918295621492893, 0.6866973237931421, 0.2232009015027919], [0.7405810890875133, 0.6497079613417431, 0.1715203062471295], [0.6036112216790016, 0.7826156180540872, 0.1522047549881467]]
LA = [[0.462, -0.829, 0.306], [0.652, -0.753, -0.017], [-0.739, 0.588, 0.329], [0.618, -0.773, 0.268], [0.58954, -0.7875, 0.17971], [0.54, -0.83, 0.04], [0.6109380889638016, -0.7617983132108493, 0.2154483266176876], [0.5065069225200065, -0.858612680339132, 0.07896203264964491], [0.6026758241093872, -0.7657775790923713, 0.22442493264674654], [0.5688488812220542, -0.7755248741629223, 0.27381037213188486], [0.5937619088044377, -0.7634937680737625, 0.25401586912127894], [0.5883968250110697, -0.7693398838105712, 0.24880779629109126], [0.6363754427166535, -0.7399472997766202, 0.21795478765210077], [0.48054623874081426, -0.8532963675744316, 0.20238730571934613]]
LH = [[0.412, -0.115, -0.896], [0.365, -0.158, -0.905], [-0.025, 0.279, -0.96], [0.354, 0.011, -0.935], [0.31645, 0.04108, -0.94772], [0.15, 0.04, -0.98], [0.3372602857004155, -0.12163731157385778, -0.9335201466076444], [0.4438784916935308, -0.13671037607201297, -0.8855970628260489], [0.3684921352784543, -0.07846800555547459, -0.9263132938385755], [0.3108801254411885, -0.05522397893634291, -0.9488434326885069], [0.34265730628177826, -0.06208292833553452, -0.9374068916223095], [0.34947933992779523, -0.059657332562775355, -0.9350428833134484], [0.42683458597005036, -0.04722141438118318, -0.9030959939251292], [0.36668762256186105, -0.04862134202713547, -0.9290727380347622]]
RP = [[0.763, -0.566, 0.307], [0.757, -0.561, 0.32], [-0.651, 0.702, 0.287], [-0.709, 0.652, -0.268], [0.69248, -0.66561, 0.27826], [0.77, -0.61, 0.14], [0.6850313001308215, -0.6653434061281155, 0.29673265705499047], [0.6026815588237138, -0.7806655785889308, 0.16533660532449618], [0.6937018946397963, -0.673509473185787, 0.25526980021569606], [0.6758774297173016, -0.6870317285531075, 0.26679037456411914], [0.7036908708098086, -0.662638639197429, 0.2563770508089845], [0.6932135825476716, -0.6777414146115746, 0.24519686762217127], [0.7427135628711982, -0.640961200623954, 0.19376610338722736], [0.6069732819341371, -0.7753600427894537, 0.17435664330227907]]
RA = [[0.462, 0.829, 0.306], [0.652, 0.753, -0.017], [0.749, 0.577, 0.324], [-0.618, -0.773, -0.142], [0.58905, 0.78928, 0.17334], [0.57, 0.82, 0.04], [0.6018829483154949, 0.7696958036006248, 0.21285038512212145], [0.49813120823450674, 0.863799639040158, 0.07560081333506219], [0.5904284226397649, 0.775275295042969, 0.224371332026081], [0.5577125731332506, 0.7839510975293902, 0.2727221341430743], [0.5887880645083642, 0.763985279211711, 0.263922542122302], [0.5777151018201632, 0.7776107461955467, 0.24812655748654436], [0.626374817441819, 0.7482877956999187, 0.21844899377494115], [0.4699881115295397, 0.8600351892192188, 0.19862187272694448]]
RH = [[0.412, 0.115, -0.896], [0.365, 0.158, -0.905], [0.017, 0.299, -0.954], [-0.354, 0.011, 0.935], [0.32891, -0.03566, -0.94369], [0.13, 0.02, -0.98], [0.3543379142756686, 0.11316150123814031, -0.9282451815895948], [0.45870843085179797, 0.12765039445199503, -0.8793702020547041], [0.3752296321195927, 0.07518282217170083, -0.9238778417245941], [0.3273040493830935, 0.04431829277651186, -0.9438792021137054], [0.35594104878093125, 0.044492708562851584, -0.9334486427637418], [0.3655653774364688, 0.04927843793873805, -0.9294802797122974], [0.44230337043905893, 0.037993547997990235, -0.8960603879252627], [0.3830513643923475, 0.03831373929725498, -0.9229321262250122]]


# In[10]:


len(LP)


# In[ ]:


#法向量作图
from vedo import *
embedWindow(False) #to pop an external VTK rendering window
from random import gauss, uniform as u #随机数据库
import numpy as np
import math
vp = Plotter()

AP=110.3/180*np.pi
PL=87.6/180*np.pi
AL=88.3/180*np.pi


OL=[1,0,0]
OP=[np.cos(PL),np.sin(PL),0]#
## 设OL=[x,y,z],则
x=np.cos(AL)
y=(np.cos(AP)-x*np.cos(PL))/np.sin(PL)
z=(1-x**2-y**2)**0.5
OA=[x,y,z]#三个半规管有夹角即可

vp +=Arrow([0, 0, 0], OP).color('green')#arrow是箭头，op是后半规管法向量

vp +=Arrow([0, 0, 0], OA).color('blue')

vp +=Arrow([0, 0, 0], OL).color('red')

vp +=Line(OP, OA).color('black')

vp +=Line(OA, OL).color('black')

vp +=Line(OL, OP).color('black')


gon = Goniometer(OA, [0,0,0], OP)#夹角表示符号

vp +=gon

gon = Goniometer(OA, [0,0,0], OL)

vp +=gon

gon = Goniometer(OL, [0,0,0], OP)

vp +=gon




vp.show(axes=1)  # render the internal list of objects in vp.actors
vp.close()


# In[20]:


#以第三份数据为基准，代入半规管角度计算法向量
import numpy as np

## Lee2013
## 上、后和外半规管单位法向量0A,OP,OL构成三棱锥，O为圆心，以OL为X轴正方向，以OL*OP作为Z轴，建立右手空间坐标系
## 三半规管法向量的夹角分别为AP,AL,PL
## 为保持空间坐标系一致，X Y 需要进行互换


OL=[1,0,0]
OP=[np.cos(PL),np.sin(PL),0]
## 设OL=[x,y,z],则
x=np.cos(AL)
y=(np.cos(AP)-x*np.cos(PL))/np.sin(PL)
z=(1-x**2-y**2)**0.5
OA=[x,y,z]

A=np.array([[0,0,0],OP,OA,OL])#三棱锥的四个点，校准
B=np.array([[0,0,0],LP[2],LA[2],LH[2]])

m = A.shape[1]
src = np.ones((m+1,A.shape[0]))
dst = np.ones((m+1,B.shape[0]))
src[:m,:] = np.copy(A.T)
dst[:m,:] = np.copy(B.T)
H = np.dot(A.T, B)
U, S, Vt = np.linalg.svd(H)
R = np.dot(Vt.T, U.T)
t=np.array([0,0,0])
T = np.identity(m+1)
T[:m, :m] = R
T[:m, m] = t
btsrc=np.dot(T,src)
A=btsrc[:m,:].T#校准以后的算法

print(A)#放于附件中


# In[24]:


#法向量作图
from vedo import *
embedWindow(False) #to pop an external VTK rendering window
from random import gauss, uniform as u
import numpy as np
import math
vp = Plotter()

OP=B[1]
OA=B[2]
OL=B[3]
 

vp +=Line([0, 0, 0], OP).color('green')

vp +=Line([0, 0, 0], OA).color('blue')

vp +=Line([0, 0, 0], OL).color('red')

vp +=Line(OP, OA).color('black')

vp +=Line(OA, OL).color('black')

vp +=Line(OL, OP).color('black')


gon = Goniometer(OA, [0,0,0], OP)

vp +=gon

gon = Goniometer(OA, [0,0,0], OL)

vp +=gon

gon = Goniometer(OL, [0,0,0], OP)

vp +=gon



OP=A[1]
OA=A[2]
OL=A[3]
 

vp +=Arrow([0, 0, 0], OP).color('green')

vp +=Arrow([0, 0, 0], OA).color('blue')

vp +=Arrow([0, 0, 0], OL).color('red')

vp +=Line(OP, OA).color('black')

vp +=Line(OA, OL).color('black')

vp +=Line(OL, OP).color('black')


gon = Goniometer(OA, [0,0,0], OP)

vp +=gon

gon = Goniometer(OA, [0,0,0], OL)

vp +=gon

gon = Goniometer(OL, [0,0,0], OP)

vp +=gon








vp.show(axes=1)  # render the internal list of objects in vp.actors
vp.close()


# In[25]:


A


# In[39]:


## 重心坐标
A_center=[np.mean([i[0] for i in A]),np.mean([i[1] for i in A]),np.mean([i[2] for i in A])]


# In[40]:


A_center#计算圆锥的中心


# In[41]:


B_center=[np.mean([i[0] for i in B]),np.mean([i[1] for i in B]),np.mean([i[2] for i in B])]


# In[42]:


B_center


# In[43]:


A_center=A_center/np.linalg.norm(A_center)
B_center=B_center/np.linalg.norm(B_center)


# In[44]:


A_center


# In[45]:


B_center


# In[57]:


#法向量作图
from vedo import *
embedWindow(False) #to pop an external VTK rendering window
from random import gauss, uniform as u
import numpy as np
import math
vp = Plotter()

OP=B[1]
OA=B[2]
OL=B[3]
 

vp +=Line([0, 0, 0], OP).color('green')

vp +=Line([0, 0, 0], OA).color('blue')

vp +=Line([0, 0, 0], OL).color('red')

vp +=Line(OP, OA).color('black')

vp +=Line(OA, OL).color('black')

vp +=Line(OL, OP).color('black')


gon = Goniometer(OA, [0,0,0], OP)

vp +=gon

gon = Goniometer(OA, [0,0,0], OL)

vp +=gon

gon = Goniometer(OL, [0,0,0], OP)

vp +=gon



OP=A[1]
OA=A[2]
OL=A[3]
 

vp +=Arrow([0, 0, 0], OP).color('green')

vp +=Arrow([0, 0, 0], OA).color('blue')

vp +=Arrow([0, 0, 0], OL).color('red')

vp +=Line(OP, OA).color('black')

vp +=Line(OA, OL).color('black')

vp +=Line(OL, OP).color('black')


gon = Goniometer(OA, [0,0,0], OP)

vp +=gon

gon = Goniometer(OA, [0,0,0], OL)

vp +=gon

gon = Goniometer(OL, [0,0,0], OP)

vp +=gon



vp +=Line([0, 0, 0], A_center).color('RED')

vp +=Line([0, 0, 0], B_center).color('RED')


vp +=mesh


vp.show(axes=1)  # render the internal list of objects in vp.actors
vp.close()


# In[51]:


import meshio


# In[52]:


A


# In[53]:


cells = [
    ("triangle", [[0, 1, 2], [0, 1, 3],[0, 1, 3],[1, 2, 3]])
]


# In[55]:


mesh = meshio.Mesh(
    A,
    cells   
)


# In[56]:


mesh


# In[84]:


#法向量作图
from vedo import *
embedWindow(False) #to pop an external VTK rendering window
from random import gauss, uniform as u
import numpy as np#mesh作图
import math
vp = Plotter()

 


verts = [(i[0],i[1],i[2]) for i in A]
faces = [(0, 1, 2), (0, 1, 3), (0, 1, 3),(1, 2, 3)]
# (the first triangle face is formed by vertex 0, 1 and 2)

# Build the polygonal Mesh object:
Amesh = Mesh([verts, faces])

#vp +=Amesh


verts = [(i[0],i[1],i[2]) for i in B]
faces = [(0, 1, 2), (0, 1, 3), (0, 1, 3),(1, 2, 3)]
# (the first triangle face is formed by vertex 0, 1 and 2)

# Build the polygonal Mesh object:
Bmesh = Mesh([verts, faces])
Bmesh.backColor('violet').lineColor('tomato')

#vp +=Bmesh

b1 = Amesh.boolean("intersect", Bmesh) 
 
vp +=b1



vp.show(axes=1)  # render the internal list of objects in vp.actors
vp.close()


# In[86]:


b1.polydata()


# In[88]:


import vtk
mass = vtk.vtkMassProperties()
mass.SetGlobalWarningDisplay(0)
mass.SetInputData(b1.polydata())
mass.Update()
mass.GetVolume()


# In[68]:


list(A)


# In[75]:


[(i[0],i[1],i[2]) for i in A]


# In[89]:


print(A,B)


# In[90]:


import vtk

verts = [(i[0],i[1],i[2]) for i in A]
faces = [(0, 1, 2), (0, 1, 3), (0, 1, 3),(1, 2, 3)]
# (the first triangle face is formed by vertex 0, 1 and 2)

# Build the polygonal Mesh object:
Amesh = Mesh([verts, faces])


verts = [(i[0],i[1],i[2]) for i in B]
faces = [(0, 1, 2), (0, 1, 3), (0, 1, 3),(1, 2, 3)]
# (the first triangle face is formed by vertex 0, 1 and 2)

# Build the polygonal Mesh object:
Bmesh = Mesh([verts, faces])
b1 = Amesh.boolean("intersect", Bmesh) 

mass = vtk.vtkMassProperties()
mass.SetGlobalWarningDisplay(0)
mass.SetInputData(b1.polydata())
mass.Update()
mass.GetVolume()


# In[174]:


angle=[['Hashimoto2003[^Hashimoto2003]',91.31,91.49,89.33,91.31,91.49,89.33,23.11,23.11,13.3],
['Blanks1975[^Blanks1975]',86.2,95.95,67.78,86.2,95.95,67.78,23.28,23.28,18.39],
['YangXK2021[^YangXK2021]',89.28,94.36,97.66,89.4,94.3,97.12,9.41,8.61,2.69],
['Bradshaw2010[^Bradshaw2010]',89.66,89.57,92.31,91.59,89.57,85.56,11.42,0,1.95],
['Santina2005[^Santina2005]',93.94,90.06,90.92,93.97,90.64,89.88,10.59,10.76,4.46],
['Aoki2012[^Aoki2012]',97.24,90.03,89.51,93.19,92.83,87.06,17.79,21.92,11.28],
['Hideaki2005[^Hideaki2005]',91.70,94.52,90.50,91.70,94.52,90.50,0,0,0,],
['Lee2013[^Lee2013]',92.1,96.2,84.4,92.1,96.2,84.1,8.05,8.39,13.52],
['Khoury2014[^Khoury2014]',111.2,88.2,74.2,111.2,88.2,74.2,8.22,8.66,15.22],
['Tang2020[^Tang2020]',92.92,90.17,85.74,93.17,91.5,85.84,8.19,7.64,8.82],
['Suzuki2010[^Suzuki2010]',95.1,93.5,92.3,95.1,93.5,92.3,8.55,7.97,5.79],
['谭惠斌2012[^谭惠斌2012]',92.73,91.28,89.27,91.39,91.05,90.16,7.95,8.56,6.16],
['王道才2006[^王道才2006]',93.77,90.45,88.92,93.77,90.45,88.92,8.49,7.98,6.32],
['Kim2015[^Kim2015]',88.4,82.5,83.7,88.4,82.5,83.7,9.06,8.45,4.98],
['Lyu2016[^Lyu2016]',110.3,87.6,88.3,110.3,87.6,88.3,9.25,8.67,5.08]]


# In[175]:


ear=[['[^Hashimoto2003]',5,6.85,8.99,9.94,6.11,8.11,9.23],
['[^Blanks1975]',20,7.21,20.32,9.65,6.35,19.97,9.39],
['[^YangXK2021]',110,1.29,0,5.09,1.62,10.85,3.85],
['[^Bradshaw2010]',34,1.29,0,5.09,1.62,10.85,3.85],     
['[^Santina2005]',44,1.77,8.98,4.38,2.15,9.17,3.61],
['[^Aoki2012]',22,11.61,19.27,11.0,11.95,16.95,13.04],
['[^Hideaki2005]',7,1.41,6.77,6.64,1.39,6.64,6.53],
['[^Lee2013]',40,1.41,6.77,6.64,1.39,6.64,6.53],
['[^Khoury2014]',274,11.76,16.61,12.2,11.69,16.38,11.95],
['[^Tang2020]',30,2.75,6.24,6.28,2.28,5.96,5.76],
['[^Suzuki2010]',39,2.86,3.95,2.6,2.8,3.72,2.33],
['[^谭惠斌2012]',120,2.27,4.53,4.42,1.88,3.61,3.81],
['[^王道才2006]',210,2.95,4.91,4.72,2.89,4.65,4.47],
['[^Kim2015]',40,5.88,6.94,9.17,5.86,6.68,8.95],
['[^Lyu2016]',152,11.14,11.57,5.5,11.07,11.41,5.28]]


# In[166]:


lab=['LP_LA','LP_LH','LA_LH','RP_RA','RP_RH','RA_RH','LP_RA','LA_RP','LH_RH']


# In[167]:


import numpy as np
LP=[[0.763,0.566,0.307], [0.757,0.561,0.320], [0.660, 0.702, 0.266],[0.709,0.652,0.268],[0.69611, 0.6682, 0.26257],[0.74,0.64,0.14],[0.763,0.566,0.307]]
LA=[[0.462,-0.829,0.306], [0.652,-0.753,-0.017], [-0.739, 0.588, 0.329],[0.618,-0.773,0.268],[0.58954, -0.7875, 0.17971],[0.54,-0.83,0.04],[0.462,-0.829,0.306]]
LH=[[0.412,-0.115,-0.896], [0.365,-0.158,-0.905], [0.025, -0.279, 0.960],[0.354,0.011,-0.935],[0.31645, 0.04108, -0.94772],[0.15, 0.04, -0.98],[0.412,-0.115, -0.896]]
RP=[[0.763,-0.566,0.307], [0.757,-0.561,0.320], [-0.651, 0.702, 0.287],[-0.709,0.652,-0.268],[0.69248, -0.66561, 0.27826],[0.77, -0.61, 0.14],[0.763, -0.566, 0.307]]
RA=[[0.462,0.829,0.306], [0.652,0.753,-0.017], [0.749, 0.577, 0.324],[-0.618,-0.773,-0.142],[0.58905, 0.78928, 0.17334],[0.57, 0.82, 0.04],[0.462, 0.829,0.306]]
RH=[[0.412,0.115,-0.896], [0.365,0.158,-0.905], [-0.017, -0.299, 0.954],[-0.354,0.011,0.935],[0.32891, -0.03566, -0.94369],[0.13, 0.02, -0.98],[0.412,0.115, -0.896]]

LH[2]=[LH[2][0]*-1,LH[2][1]*-1,LH[2][2]*-1]
RH[2]=[RH[2][0]*-1,RH[2][1]*-1,RH[2][2]*-1]#同一方向的校准

total=[LP,LA,LH,RP,RA,RH]#规定半规管的夹角要统一


    
for ca in total:#参照的标准模型
    x,y,z=ca[2]
    print(ca[2])
    ca[2][1]=x
    ca[2][0]=y
    ca[2][2]=z
    print(ca[2])    
    
for i in range(3):
    x,y,z=total[i+3][3]
    total[i+3][3][0]=-x
    total[i+3][3][1]=-y
    total[i+3][3][2]=-z
  
import copy
totaldo=copy.deepcopy(total)

for i in range(6):
    for j in range(3):
        lac=np.array(totaldo[i][j])
        lac=lac/np.linalg.norm(lac)
        #print(lac)
        #print(total[i][j])
        totaldo[i][j]=[lac[0],lac[1],lac[2]]
        
        
        
title=['LP','LA','LH','RP','RA','RH']#形成一个标准的数据集


# In[ ]:


#以第三份数据为基准，代入半规管角度计算法向量
import numpy as np

## Lee2013
## 上、后和外半规管单位法向量0A,OP,OL构成三棱锥，O为圆心，以OL为X轴正方向，以OL*OP作为Z轴，建立右手空间坐标系
## 三半规管法向量的夹角分别为AP,AL,PL
## 为保持空间坐标系一致，X Y 需要进行互换

i=0

print('|研究组|耳数|LP|LA|LH|RP|RA|RH|')
print('|---|---|---|---|---|---|---|---|')

for i in range(len(angle)):

    AP=angle[i][1]/180*np.pi
    PL=angle[i][2]/180*np.pi
    AL=angle[i][3]/180*np.pi

    OL=[1,0,0]
    OP=[np.cos(PL),np.sin(PL),0]
    ## 设OL=[x,y,z],则
    x=np.cos(AL)
    y=(np.cos(AP)-x*np.cos(PL))/np.sin(PL)
    z=(1-x**2-y**2)**0.5
    OA=[x,y,z]

    A=np.array([[0,0,0],OP,OA,OL])
    B=np.array([[0,0,0],LP[2],LA[2],LH[2]])
    #B=np.array([[0,0,0],totaldo[2],LA[2],LH[2]])

    m = A.shape[1]
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)
    H = np.dot(A.T, B)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    t=np.array([0,0,0])
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    btsrc=np.dot(T,src)
    LA=btsrc[:m,:].T

    #print(LA)



    AP=angle[i][4]/180*np.pi
    PL=angle[i][5]/180*np.pi
    AL=angle[i][6]/180*np.pi

    OL=[1,0,0]
    OP=[np.cos(PL),np.sin(PL),0]
    ## 设OL=[x,y,z],则
    x=np.cos(AL)
    y=(np.cos(AP)-x*np.cos(PL))/np.sin(PL)
    z=(1-x**2-y**2)**0.5
    OA=[x,y,z]

    A=np.array([[0,0,0],OP,OA,OL])
    B=np.array([[0,0,0],RP[2],RA[2],RH[2]])

    m = A.shape[1]
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)
    H = np.dot(A.T, B)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    t=np.array([0,0,0])
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    btsrc=np.dot(T,src)
    RA=btsrc[:m,:].T

    #print(RA)


    #print('|'+angle[i][0]+'|',LA[1],'|',LA[2],'|',LA[3],'|',RA[1],'|',RA[2],'|',RA[3],'|')
    #print('|'+angle[i][0]+'|',[round(k*100)/100 for k in LA[1]],'|',[round(k*100)/100 for k in LA[2]],'|',[round(k*100)/100 for k in LA[3]],'|',[round(k*100)/100 for k in RA[1]],'|',[round(k*100)/100 for k in RA[2]],'|',[round(k*100)/100 for k in RA[3]],'|')
  
    print('|'+angle[i][0]+'|',ear[i][1],'|',[round(k*1000)/1000 for k in LA[1]],'|',[round(k*1000)/1000 for k in LA[2]],'|',[round(k*1000)/1000 for k in LA[3]],'|',[round(k*1000)/1000 for k in RA[1]],'|',[round(k*1000)/1000 for k in RA[2]],'|',[round(k*1000)/1000 for k in RA[3]],'|')


# In[1]:


#以第三份数据为基准，代入半规管角度计算法向量
import numpy as np

## Lee2013
## 上、后和外半规管单位法向量0A,OP,OL构成三棱锥，O为圆心，以OL为X轴正方向，以OL*OP作为Z轴，建立右手空间坐标系
## 三半规管法向量的夹角分别为AP,AL,PL
## 为保持空间坐标系一致，X Y 需要进行互换

i=0

print('|研究组|耳数|LP|LA|LH|RP|RA|RH|')
print('|---|---|---|---|---|---|---|---|')

snum=4

for i in range(len(angle)):

    AP=angle[i][1]/180*np.pi
    PL=angle[i][2]/180*np.pi
    AL=angle[i][3]/180*np.pi

    OL=[1,0,0]
    OP=[np.cos(PL),np.sin(PL),0]
    ## 设OL=[x,y,z],则
    x=np.cos(AL)
    y=(np.cos(AP)-x*np.cos(PL))/np.sin(PL)
    z=(1-x**2-y**2)**0.5
    OA=[x,y,z]

    A=np.array([[0,0,0],OP,OA,OL])
    #B=np.array([[0,0,0],LP[2],LA[2],LH[2]])
    B=np.array([[0,0,0],totaldo[0][snum],totaldo[1][snum],total[2][snum]])

    m = A.shape[1]
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)
    H = np.dot(A.T, B)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    t=np.array([0,0,0])
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    btsrc=np.dot(T,src)
    TL=btsrc[:m,:].T

    #print(LA)



    AP=angle[i][4]/180*np.pi
    PL=angle[i][5]/180*np.pi
    AL=angle[i][6]/180*np.pi

    OL=[1,0,0]
    OP=[np.cos(PL),np.sin(PL),0]
    ## 设OL=[x,y,z],则
    x=np.cos(AL)
    y=(np.cos(AP)-x*np.cos(PL))/np.sin(PL)
    z=(1-x**2-y**2)**0.5
    OA=[x,y,z]

    A=np.array([[0,0,0],OP,OA,OL])
    #B=np.array([[0,0,0],RP[2],RA[2],RH[2]])
    B=np.array([[0,0,0],totaldo[3][snum],totaldo[4][snum],total[5][snum]])

    m = A.shape[1]
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)
    H = np.dot(A.T, B)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    t=np.array([0,0,0])
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    btsrc=np.dot(T,src)
    TR=btsrc[:m,:].T

    #print(RA)


    #print('|'+angle[i][0]+'|',LA[1],'|',LA[2],'|',LA[3],'|',RA[1],'|',RA[2],'|',RA[3],'|')
    #print('|'+angle[i][0]+'|',[round(k*100)/100 for k in LA[1]],'|',[round(k*100)/100 for k in LA[2]],'|',[round(k*100)/100 for k in LA[3]],'|',[round(k*100)/100 for k in RA[1]],'|',[round(k*100)/100 for k in RA[2]],'|',[round(k*100)/100 for k in RA[3]],'|')
  
    #print('|'+angle[i][0]+'|',ear[i][1],'|',[round(k*1000)/1000 for k in AL[1]],'|',[round(k*1000)/1000 for k in AL[2]],'|',[round(k*1000)/1000 for k in AL[3]],'|',[round(k*1000)/1000 for k in AR[1]],'|',[round(k*1000)/1000 for k in AR[2]],'|',[round(k*1000)/1000 for k in AR[3]],'|')
   
    TRL=[TL[1],TL[2],TL[3],TR[1],TR[2],TR[3]]
    print('\n|'+angle[i][0]+'|',ear[i][1],'|',end="")
    for j in range(6):
        myangle=np.arccos(np.dot(totaldo[j][snum],list(TRL[j])))*180/np.pi
        if myangle >0:
            print(round(myangle*100)/100,end="|")
        else:
            print(myangle,end="|")


# |研究组|耳数|LP|LA|LH|RP|RA|RH|
# |---|---|---|---|---|---|---|---|
# |Hashimoto2003[^Hashimoto2003]| 5 | [0.702, 0.671, 0.238] | [0.602, -0.755, 0.258] | [0.341, -0.063, -0.938] | [0.703, -0.662, 0.26] | [0.592, 0.764, 0.257] | [0.358, 0.052, -0.932] |
# |Blanks1975[^Blanks1975]| 20 | [0.724, 0.623, 0.295] | [0.691, -0.721, 0.051] | [0.398, -0.206, -0.894] | [0.723, -0.614, 0.316] | [0.684, 0.727, 0.055] | [0.414, 0.199, -0.889] |
# |YangXK2021[^YangXK2021]| 110 | [0.695, 0.669, 0.263] | [0.651, -0.734, 0.193] | [0.181, 0.084, -0.98] | [0.695, -0.66, 0.284] | [0.642, 0.743, 0.193] | [0.205, -0.091, -0.975] |
# |Bradshaw2010[^Bradshaw2010]| 34 | [0.71, 0.668, 0.224] | [0.636, -0.74, 0.217] | [0.291, 0.024, -0.956] | [0.701, -0.671, 0.243] | [0.632, 0.759, 0.159] | [0.346, 0.01, -0.938] |
# |Santina2005[^Santina2005]| 44 | [0.687, 0.692, 0.222] | [0.606, -0.768, 0.209] | [0.304, 0.002, -0.953] | [0.692, -0.677, 0.251] | [0.601, 0.78, 0.173] | [0.307, -0.023, -0.951] |
# |Aoki2012[^Aoki2012]| 22 | [0.682, 0.697, 0.219] | [0.572, -0.799, 0.186] | [0.312, -0.008, -0.95] | [0.701, -0.659, 0.273] | [0.597, 0.786, 0.159] | [0.315, 0.017, -0.949] |
# |Hideaki2005[^Hideaki2005]| 7 | [0.716, 0.643, 0.272] | [0.588, -0.785, 0.196] | [0.275, -0.023, -0.961] | [0.706, -0.645, 0.291] | [0.596, 0.781, 0.185] | [0.283, 0.0, -0.959] |
# |Lee2013[^Lee2013]| 40 | [0.704, 0.652, 0.282] | [0.609, -0.779, 0.15] | [0.298, -0.075, -0.952] | [0.698, -0.649, 0.302] | [0.61, 0.779, 0.143] | [0.313, 0.061, -0.948] |
# |Khoury2014[^Khoury2014]| 274 | [0.609, 0.774, 0.17] | [0.506, -0.863, -0.012] | [0.397, -0.07, -0.915] | [0.608, -0.77, 0.192] | [0.504, 0.864, -0.016] | [0.412, 0.058, -0.909] |
# |Tang2020[^Tang2020]| 30 | [0.763, 0.598, 0.244] | [0.562, -0.825, 0.056] | [0.276, 0.034, -0.96] | [0.757, -0.592, 0.276] | [0.555, 0.83, 0.057] | [0.285, -0.038, -0.958] |
# |Suzuki2010[^Suzuki2010]| 39 | [0.716, 0.645, 0.266] | [0.559, -0.817, 0.14] | [0.236, 0.044, -0.971] | [0.716, -0.636, 0.288] | [0.55, 0.823, 0.143] | [0.256, -0.053, -0.965] |
# |谭惠斌2012[^谭惠斌2012]| 120 | [0.73, 0.638, 0.247] | [0.579, -0.799, 0.161] | [0.291, 0.002, -0.957] | [0.737, -0.62, 0.269] | [0.575, 0.799, 0.174] | [0.308, -0.017, -0.951] |
# |王道才2006[^王道才2006]| 210 | [0.717, 0.656, 0.235] | [0.583, -0.796, 0.165] | [0.302, 0.0, -0.953] | [0.714, -0.652, 0.256] | [0.576, 0.799, 0.174] | [0.324, -0.004, -0.946] |
# |Kim2015[^Kim2015]| 40 | [0.763, 0.622, 0.176] | [0.63, -0.765, 0.134] | [0.379, 0.007, -0.925] | [0.763, -0.616, 0.198] | [0.622, 0.77, 0.141] | [0.399, -0.012, -0.917] |
# |Lyu2016[^Lyu2016]| 152 | [0.605, 0.775, 0.18] | [0.498, -0.861, 0.105] | [0.312, 0.03, -0.949] | [0.607, -0.769, 0.2] | [0.489, 0.865, 0.108] | [0.333, -0.037, -0.942] |

# In[171]:


#以第三份数据为基准，代入半规管角度计算法向量
import numpy as np

## Lee2013
## 上、后和外半规管单位法向量0A,OP,OL构成三棱锥，O为圆心，以OL为X轴正方向，以OL*OP作为Z轴，建立右手空间坐标系
## 三半规管法向量的夹角分别为AP,AL,PL
## 为保持空间坐标系一致，X Y 需要进行互换

i=0

print('|研究组|耳数|LP|LA|LH|RP|RA|RH|')
print('|---|---|---|---|---|---|---|---|')

snum=4

for i in range(len(angle)):

    AP=angle[i][1]/180*np.pi
    PL=angle[i][2]/180*np.pi
    AL=angle[i][3]/180*np.pi

    OL=[1,0,0]
    OP=[np.cos(PL),np.sin(PL),0]
    ## 设OL=[x,y,z],则
    x=np.cos(AL)
    y=(np.cos(AP)-x*np.cos(PL))/np.sin(PL)
    z=(1-x**2-y**2)**0.5
    OA=[x,y,z]

    A=np.array([[0,0,0],OP,OA,OL])
    #B=np.array([[0,0,0],LP[2],LA[2],LH[2]])
    B=np.array([[0,0,0],totaldo[0][snum],totaldo[1][snum],total[2][snum]])

    m = A.shape[1]
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)
    H = np.dot(A.T, B)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    t=np.array([0,0,0])
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    btsrc=np.dot(T,src)
    TL=btsrc[:m,:].T

    #print(LA)



    AP=angle[i][4]/180*np.pi
    PL=angle[i][5]/180*np.pi
    AL=angle[i][6]/180*np.pi

    OL=[1,0,0]
    OP=[np.cos(PL),np.sin(PL),0]
    ## 设OL=[x,y,z],则
    x=np.cos(AL)
    y=(np.cos(AP)-x*np.cos(PL))/np.sin(PL)
    z=(1-x**2-y**2)**0.5
    OA=[x,y,z]

    A=np.array([[0,0,0],OP,OA,OL])
    #B=np.array([[0,0,0],RP[2],RA[2],RH[2]])
    B=np.array([[0,0,0],totaldo[3][snum],totaldo[4][snum],total[5][snum]])

    m = A.shape[1]
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)
    H = np.dot(A.T, B)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    t=np.array([0,0,0])
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    btsrc=np.dot(T,src)
    TR=btsrc[:m,:].T

    #print(RA)


    #print('|'+angle[i][0]+'|',LA[1],'|',LA[2],'|',LA[3],'|',RA[1],'|',RA[2],'|',RA[3],'|')
    #print('|'+angle[i][0]+'|',[round(k*100)/100 for k in LA[1]],'|',[round(k*100)/100 for k in LA[2]],'|',[round(k*100)/100 for k in LA[3]],'|',[round(k*100)/100 for k in RA[1]],'|',[round(k*100)/100 for k in RA[2]],'|',[round(k*100)/100 for k in RA[3]],'|')
  
    print('|'+angle[i][0]+'|',ear[i][1],'|',[round(k*1000)/1000 for k in TL[1]],'|',[round(k*1000)/1000 for k in TL[2]],'|',[round(k*1000)/1000 for k in TL[3]],'|',[round(k*1000)/1000 for k in TR[1]],'|',[round(k*1000)/1000 for k in TR[2]],'|',[round(k*1000)/1000 for k in TR[3]],'|')


# In[158]:


#以第三份数据为基准，代入半规管角度计算法向量
import numpy as np

## Lee2013
## 上、后和外半规管单位法向量0A,OP,OL构成三棱锥，O为圆心，以OL为X轴正方向，以OL*OP作为Z轴，建立右手空间坐标系
## 三半规管法向量的夹角分别为AP,AL,PL
## 为保持空间坐标系一致，X Y 需要进行互换

i=0

print('|研究组|耳数|LP|LA|LH|RP|RA|RH|')
print('|---|---|---|---|---|---|---|---|')

snum=2

for i in range(len(angle)):

    AP=angle[i][1]/180*np.pi
    PL=angle[i][2]/180*np.pi
    AL=angle[i][3]/180*np.pi

    OL=[1,0,0]
    OP=[np.cos(PL),np.sin(PL),0]
    ## 设OL=[x,y,z],则
    x=np.cos(AL)
    y=(np.cos(AP)-x*np.cos(PL))/np.sin(PL)
    z=(1-x**2-y**2)**0.5
    OA=[x,y,z]

    A=np.array([[0,0,0],OP,OA,OL])
    #B=np.array([[0,0,0],LP[2],LA[2],LH[2]])
    B=np.array([[0,0,0],totaldo[0][snum],totaldo[1][snum],total[2][snum]])

    m = A.shape[1]
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)
    H = np.dot(A.T, B)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    t=np.array([0,0,0])
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    btsrc=np.dot(T,src)
    TL=btsrc[:m,:].T

    #print(LA)



    AP=angle[i][4]/180*np.pi
    PL=angle[i][5]/180*np.pi
    AL=angle[i][6]/180*np.pi

    OL=[1,0,0]
    OP=[np.cos(PL),np.sin(PL),0]
    ## 设OL=[x,y,z],则
    x=np.cos(AL)
    y=(np.cos(AP)-x*np.cos(PL))/np.sin(PL)
    z=(1-x**2-y**2)**0.5
    OA=[x,y,z]

    A=np.array([[0,0,0],OP,OA,OL])
    #B=np.array([[0,0,0],RP[2],RA[2],RH[2]])
    B=np.array([[0,0,0],totaldo[3][snum],totaldo[4][snum],total[5][snum]])

    m = A.shape[1]
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)
    H = np.dot(A.T, B)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    t=np.array([0,0,0])
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    btsrc=np.dot(T,src)
    TR=btsrc[:m,:].T
    TRL=[TL[1],TL[2],TL[3],TR[1],TR[2],TR[3]]
    #print("\n"+'|'+str(j)+'|',end="")
    print('\n|'+angle[i][0]+'|',ear[i][1],'|',end="")
    myangle=np.arccos(np.dot(TRL[0],TRL[1]))*180/np.pi
    if myangle >0:
        print(round(myangle*100)/100,end="|")
    else:
        print(myangle,end="|")
    myangle=np.arccos(np.dot(TRL[0],TRL[2]))*180/np.pi
    if myangle >0:
        print(round(myangle*100)/100,end="|")
    else:
        print(myangle,end="|")
    myangle=np.arccos(np.dot(TRL[1],TRL[2]))*180/np.pi
    if myangle >0:
        print(round(myangle*100)/100,end="|")
    else:
        print(myangle,end="|")
    myangle=np.arccos(np.dot(TRL[3],TRL[4]))*180/np.pi
    if myangle >0:
        print(round(myangle*100)/100,end="|")
    else:
        print(myangle,end="|")
    myangle=np.arccos(np.dot(TRL[3],TRL[5]))*180/np.pi
    if myangle >0:
        print(round(myangle*100)/100,end="|")
    else:
        print(myangle,end="|")
    myangle=np.arccos(np.dot(TRL[4],TRL[5]))*180/np.pi
    if myangle >0:
        print(round(myangle*100)/100,end="|")
    else:
        print(myangle,end="|")
    myangle=np.arccos(np.dot(TRL[0],TRL[4]))*180/np.pi
    if myangle >0:
        print(round(myangle*100)/100,end="|")
    else:
        print(myangle,end="|")
    myangle=np.arccos(np.dot(TRL[3],TRL[1]))*180/np.pi
    if myangle >0:
        print(round(myangle*100)/100,end="|")
    else:
        print(myangle,end="|")
    myangle=np.arccos(np.dot(TRL[2],TRL[5]))*180/np.pi
    if myangle >0:
        print(round(myangle*100)/100,end="|")
    else:
        print(myangle,end="|")

     


# In[141]:


print(TL)


# In[144]:


sum(TR[1]**2)


# In[146]:


sum(np.array(totaldo[0][snum])**2)


# In[147]:


lac=totaldo[0][snum]
lac=lac/np.linalg.norm(lac)


# In[148]:


lac


# In[149]:


totaldo[0][snum]


# In[177]:


print('|研究组|耳数|LP|LA|LH|RP|RA|RH|')
print('|---|---|---|---|---|---|---|---|')

import numpy as np
for j in range(6):
    #print("\n"+'|'+str(j)+'|',end="")
    print('\n|'+angle[j][0]+'|',ear[j][1],'|',end="")
    myangle=np.arccos(np.dot(totaldo[0][j],totaldo[1][j]))*180/np.pi
    if myangle >0:
        print(round(myangle*100)/100,end="|")
    else:
        print(myangle,end="|")
    myangle=np.arccos(np.dot(totaldo[0][j],totaldo[2][j]))*180/np.pi
    if myangle >0:
        print(round(myangle*100)/100,end="|")
    else:
        print(myangle,end="|")
    myangle=np.arccos(np.dot(totaldo[1][j],totaldo[2][j]))*180/np.pi
    if myangle >0:
        print(round(myangle*100)/100,end="|")
    else:
        print(myangle,end="|")
    myangle=np.arccos(np.dot(totaldo[3][j],totaldo[4][j]))*180/np.pi
    if myangle >0:
        print(round(myangle*100)/100,end="|")
    else:
        print(myangle,end="|")
    myangle=np.arccos(np.dot(totaldo[3][j],totaldo[5][j]))*180/np.pi
    if myangle >0:
        print(round(myangle*100)/100,end="|")
    else:
        print(myangle,end="|")
    myangle=np.arccos(np.dot(totaldo[4][j],totaldo[5][j]))*180/np.pi
    if myangle >0:
        print(round(myangle*100)/100,end="|")
    else:
        print(myangle,end="|")
    myangle=np.arccos(np.dot(totaldo[0][j],totaldo[4][j]))*180/np.pi
    if myangle >0:
        print(round(myangle*100)/100,end="|")
    else:
        print(myangle,end="|")
    myangle=np.arccos(np.dot(totaldo[3][j],totaldo[1][j]))*180/np.pi
    if myangle >0:
        print(round(myangle*100)/100,end="|")
    else:
        print(myangle,end="|")
    myangle=np.arccos(np.dot(totaldo[2][j],totaldo[5][j]))*180/np.pi
    if myangle >0:
        print(round(myangle*100)/100,end="|")
    else:
        print(myangle,end="|")





# In[ ]:




