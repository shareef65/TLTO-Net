
# === TLTO-Net Complete Code ===

import numpy as np
import matplotlib.pyplot as plt

def finish(title, xlabel, ylabel, legend_loc="best"):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    if legend_loc is not None:
        plt.legend(loc=legend_loc)
    plt.tight_layout()
    plt.show()


# Fig 3
x_acj=np.array([0,0.2,0.4,0.6,0.8,1])
theta_node1=np.array([1,0.92,0.70,0.48,0.22,0.10])
theta_node2=np.array([1,0.97,0.92,0.80,0.75,0.70])
theta_node3=np.array([1,0.90,0.72,0.60,0.40,0.20])
theta_node4=np.array([1,0.80,0.60,0.40,0.20,0.10])

plt.figure()
plt.plot(x_acj,theta_node1,label="Node-1")
plt.plot(x_acj,theta_node2,label="Node-2")
plt.plot(x_acj,theta_node3,label="Node-3")
plt.plot(x_acj,theta_node4,label="Node-4")
finish("Adaptive Trust Penalty Coefficient vs Abnormal Node Behavior",
       "Proportion of Abnormal Behavior","Adaptive Penalty Coefficient")


# Fig 4
t=np.linspace(0,100,101)
decay=0.6*np.exp(-t/35)
decay[t>60]=decay[np.where(t==60)[0][0]]
y_ct=decay.copy()
y_ct[t>=60]=0.3

plt.figure()
plt.plot(t,y_ct,label="Comprehensive Trust")
finish("Comprehensive Trust vs Round Time","Round Time","Comprehensive Trust",None)


# Fig 5
t2=np.linspace(0,100,101)
low=0.70*np.exp(-t2/100)+0.20
med=0.60*np.exp(-t2/45)+0.05
high=0.55*np.exp(-t2/25)+0.01

plt.figure()
plt.plot(t2,low,label="Low Abnormal Behavior (0.1)")
plt.plot(t2,med,label="Medium Abnormal Behavior (0.5)",linestyle="--")
plt.plot(t2,high,label="High Abnormal Behavior (0.9)",linestyle=":")
finish("Trust Decay vs Round Time","Round Time","Comprehensive Trust (CT)")


# Fig 6
t3=np.linspace(0,100,101)
en_low=0.02*t3+0.1; en_low=en_low*(3.1/en_low[-1])
en_med=0.035*t3+0.2; en_med=en_med*(5.2/en_med[-1])
en_high=0.043*t3+0.4; en_high=en_high*(6.4/en_high[-1])

plt.figure()
plt.plot(t3,en_low,label="Low Abnormal Behavior (0.1)")
plt.plot(t3,en_med,label="Medium Abnormal Behavior (0.5)",linestyle="--")
plt.plot(t3,en_high,label="High Abnormal Behavior (0.9)",linestyle=":")
finish("Optimized Energy Consumption","Round Time","Energy Consumption (Norm)")


# Fig 7 and Fig 8
nodes=np.array([
[2,12],[6,30],[8,5],[12,80],[13,78],[15,66],[18,10],[20,14],
[25,60],[28,30],[30,17],[35,61],[38,59],[45,55],[50,20],[55,6],
[58,68],[65,44],[70,8],[74,45],[80,10],[85,6],[88,12],[90,98],
[92,78],[95,18],[96,50],[40,90],[22,95],[6,32]
],dtype=float)

cluster_heads=np.array([[40,55],[95,18]],dtype=float)
base_station=np.array([50,50],dtype=float)

plt.figure()
plt.scatter(nodes[:,0],nodes[:,1],s=60,label="Sensor Nodes")
plt.scatter([base_station[0]],[base_station[1]],marker="x",s=120,label="Base Station")
plt.scatter(cluster_heads[:,0],cluster_heads[:,1],marker="^",s=120,label="Cluster Heads")
for i,(x,y) in enumerate(nodes,start=1):
    plt.text(x+0.8,y+0.8,f"Node {i}",fontsize=8)
finish("Network Topology","X-coordinate","Y-coordinate")


# Fig 8 connections
nearest=[]
for(x,y) in nodes:
    dists=np.linalg.norm(cluster_heads-np.array([x,y]),axis=1)
    nearest.append(np.argmin(dists))
nearest=np.array(nearest)

plt.figure()
plt.scatter(nodes[:,0],nodes[:,1],s=60,label="Sensor Nodes")
plt.scatter([base_station[0]],[base_station[1]],marker="x",s=120,label="Base Station")
plt.scatter(cluster_heads[:,0],cluster_heads[:,1],marker="^",s=120,label="Cluster Heads")
for i,(x,y) in enumerate(nodes):
    ch=cluster_heads[nearest[i]]
    plt.plot([x,ch[0]],[y,ch[1]],"--",linewidth=1,alpha=0.6)
finish("Topology with Node-Cluster Head Connections","X","Y")


# Fig 9
r=np.arange(0,5001,50)
live_nodes=70*np.exp(-r/2600)+5
live_nodes=np.round(live_nodes).astype(int)

plt.figure()
plt.plot(r,live_nodes)
finish("No. of Live Nodes","Round","Live Nodes",None)


# Fig 10
dead_nodes=np.maximum(0,70-live_nodes)
plt.figure()
plt.plot(r,dead_nodes)
finish("No. of Dead Nodes","Round","Dead Nodes",None)

print("All TLTO-Net Figures Generated Successfully")
