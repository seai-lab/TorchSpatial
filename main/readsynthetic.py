
import pickle as pk
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle as pk

###change name constant or vary
with open("../data/constantbandwidth.pkl", "rb") as f:
    train_loc,train_class,val_loc,val_class,test_loc,test_class,dic=pk.load(f)
print('done')
'''
dict:
    key: cluster label
    value: (center_coord, bandwidth)
        center_coord: np.array(), (x, y ,z)
'''

def xyz2lonlan(x):
    lan=np.arcsin(x[:,2])
    lon=np.zeros(len(x))
    temp=np.sqrt(1-np.power(x[:,2],2))
    for i,coord in enumerate(x):
        if abs(coord[2])==1:
            continue
        if coord[1]>=0:
            lon[i]=np.arccos(coord[0]/temp[i])
        else:
            lon[i]=2*np.pi-np.arccos(coord[0]/temp[i])
    return np.array([lon,lan]).T
def lonlan2xyz(lonlan):
    x=np.array([np.cos(lonlan[:,1])])*np.array([np.cos(lonlan[:,0])])
    y=np.array([np.cos(lonlan[:,1])])*np.array([np.sin(lonlan[:,0])])
    z=np.array([np.sin(lonlan[:,1])])
    return np.concatenate((x,y,z),0).T

train_locp=xyz2lonlan(train_loc)
val_locp=xyz2lonlan(val_loc)
test_locp=xyz2lonlan(test_loc)
print(np.max(np.abs(train_loc-lonlan2xyz(train_locp))))
print(np.max(np.abs(val_loc-lonlan2xyz(val_locp))))
print(np.max(np.abs(test_loc-lonlan2xyz(test_locp))))


######visualize
ax = plt.subplot(111, projection='3d')
cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0, 1, len(dic))]
print(dic)
for i in range(len(dic)):
    x=[train_loc[j,0] for j in range(len(train_loc)) if train_class[j]==i]
    y=[train_loc[j,1] for j in range(len(train_loc)) if train_class[j]==i]
    z=[train_loc[j,2] for j in range(len(train_loc)) if train_class[j]==i]
    ax.scatter(x,y,z,color=colors[i])
ax.set_zlabel('Z')  
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()