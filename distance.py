import numpy as np
import random
import math
import matplotlib.pyplot as plt
def initQtable():
    qtable=np.zeros((15,4))
    qtable[0][0]=0.5
    airplane=np.array([[200,280],#1
                       [240,325],#2
                       [340,425],#3
                       [380,450],#4
                       [420,470],#5
                       [520,590],#6
                       [535,600],#7
                       [580,640],#8
                       [685,765],#9
                       [690,790],#10
                       [705,775],#11
                       [795,900],#12
                       [860,960],#13
                       [880,930]])#14
    gate1=np.zeros((1,1))
    gate2=np.zeros((1,1))
    gate3=np.zeros((1,1))
    gate4 = np.zeros((1, 1))
    gate1[0]=200
    gate1=np.append(gate1,280)
    distance=np.array([300,500,200,400])
    return qtable,airplane,gate1,gate2,gate3,gate4,distance

def epsilon_greedy_choose_action(qtable,gate1,gate2,gate3,gate4,pre_state,airplane,epsilon):
    if random.uniform(0,1)<epsilon:   #随机选择
        return random.randint(0,3)
    else:
        m =np.argwhere(qtable[pre_state]==max(qtable[pre_state]))
        if m.shape[0]>1:
            return random.choice(m)
        else:
            return int(m)

# def initReward(gate1,gate2,gate3,gate4,action,pre_state,airplane):
#     if action==0:
#         if gate1.shape[0]==2:
#             return 0.5
#         else:
#             pre_time=gate1[gate1.shape[0]-3]
#             now_time=airplane[pre_state][0]
#             if now_time-pre_time>=30:
#                 return 1-(now_time-pre_time)/60
#             else:
#                 return -2
#     elif action==1:
#         if gate2.shape[0]==2:
#             return 0.5
#         else:
#             pre_time=gate2[gate2.shape[0]-3]
#             now_time=airplane[pre_state][0]
#             if now_time-pre_time>=30:
#                 return 1-(now_time-pre_time)/60
#             else:
#                 return -2
#     elif action==2:
#         if gate3.shape[0]==2:
#             return 0.5
#         else:
#             pre_time=gate3[gate3.shape[0]-3]
#             now_time=airplane[pre_state][0]
#             if now_time-pre_time>=30:
#                 return 1-(now_time-pre_time)/60
#             else:
#                 return -2
#     elif action==3:
#         if gate4.shape[0]==2:
#             return 0.5
#         else:
#             pre_time=gate4[gate4.shape[0]-3]
#             now_time=airplane[pre_state][0]
#             if now_time-pre_time>=30:
#                 return 1-(now_time-pre_time)/60
#             else:
#                 return -2

def ch(t):#用于时间
    j=2
    while(1):
       if t.shape[0]<=2:
           break
       else:
            if j<t.shape[0]:
                t[j]=t[j]-40
                j=j+2
            else:
                break
    return t

def fitness(t):
    j = 2
    temp=0
    while (1):
        if t.shape[0] <= 2:
            temp=0
            break
        else:
            # if (t[t.shape[0]-2]-t[t.shape[0]-3])<30:
            #     temp=-1
            #     break
            if j < t.shape[0]:
                if (t[j]-t[j-1]>=40):
                    temp=temp+1/(1+math.exp((t[j]-t[j-1]-30)/10))
                else:
                    temp=temp-1.5
                j = j + 2
            else:
                break
    return temp

def fitness2(distance,t):
    return 2/(1+math.exp(distance[t]/350))

first_state=1
alpha = 0.85#学习系数
gamma = 0.90#gamma
epsilon = 0.3#epsilon
Qtable,Airplane,Gate1,Gate2,Gate3,Gate4,Distance=initQtable()
pre_state=first_state
Gbest=np.array([0])
T1=[]
T2=[]
T3=[]
T4=[]
T11=[]
T22=[]
T33=[]
T44=[]
hhh=0
N=10000
p=np.zeros((N,1))
totalfitness=0
while (hhh<N):
    if pre_state==first_state:
        Gate1 = np.zeros((1, 1))
        Gate2 = np.zeros((1, 1))
        Gate3 = np.zeros((1, 1))
        Gate4 = np.zeros((1, 1))
        Gate1[0] = 200
        Gate1 = np.append(Gate1, 280)
        action = epsilon_greedy_choose_action(Qtable, Gate1, Gate2, Gate3, Gate4,pre_state, Airplane, epsilon)
        gbest = [0]
        reward2=(fitness(Gate1) + fitness(Gate2) + fitness(Gate3) + fitness(Gate4))/(pre_state)
    gbest.append(int(action))
    if action==0:
        if Gate1.shape[0]==1:
            Gate1[0]=Airplane[pre_state][0]
            Gate1=np.append(Gate1,Airplane[pre_state][1])
        else:
            Gate1 = np.append(Gate1, Airplane[pre_state][0])
            Gate1 = np.append(Gate1, Airplane[pre_state][1])
    if action==1:
        if Gate2.shape[0]==1:
            Gate2[0]=Airplane[pre_state][0]
            Gate2=np.append(Gate2,Airplane[pre_state][1])
        else:
            Gate2 = np.append(Gate2, Airplane[pre_state][0])
            Gate2 = np.append(Gate2, Airplane[pre_state][1])
    if action==2:
        if Gate3.shape[0]==1:
            Gate3[0]=Airplane[pre_state][0]
            Gate3=np.append(Gate3,Airplane[pre_state][1])
        else:
            Gate3 = np.append(Gate3, Airplane[pre_state][0])
            Gate3 = np.append(Gate3, Airplane[pre_state][1])
    if action==3:
        if Gate4.shape[0]==1:
            Gate4[0]=Airplane[pre_state][0]
            Gate4=np.append(Gate4,Airplane[pre_state][1])
        else:
            Gate4 = np.append(Gate4, Airplane[pre_state][0])
            Gate4 = np.append(Gate4, Airplane[pre_state][1])
    if pre_state<13:
        new_action = epsilon_greedy_choose_action(Qtable, Gate1, Gate2, Gate3,Gate4 ,pre_state + 1, Airplane, epsilon)
    else:
        new_action=0
    reward1 = (fitness(Gate1) + fitness(Gate2) + fitness(Gate3) + fitness(Gate4))/(pre_state+1)
    reward= 0.9*((pre_state+1)*reward1-reward2*pre_state)+0.3*reward1+fitness2(Distance,action)
    Qtable[pre_state][action] = Qtable[pre_state][action] + alpha * (
                reward + gamma * (Qtable[pre_state+1][new_action]) - Qtable[pre_state][action])
    action=new_action
    reward2=reward1
    if pre_state==13:
        t11 = np.copy(Gate1)
        t22 = np.copy(Gate2)
        t33 = np.copy(Gate3)
        t44 = np.copy(Gate4)
        t1=np.copy(Gate1)
        t2 = np.copy(Gate2)
        t3 = np.copy(Gate3)
        t4 = np.copy(Gate4)
        t1=ch(t1)
        t2=ch(t2)
        t3=ch(t3)
        t4 = ch(t4)
        t11=ch(t11)
        t22=ch(t22)
        t33=ch(t33)
        t44 = ch(t44)
        t44.sort()
        t33.sort()
        t22.sort()
        t11.sort()
        hhh=hhh+1
        epsilon=0.3/(1+2*math.exp((hhh-1)*1/N))
        for g in range(14):
            totalfitness=totalfitness+fitness2(Distance,gbest[g])
        if ((gbest[0:13]==Gbest[0:13]).all()):
            p[hhh]=1
            print(Gbest)
        if  (t11==t1).all() and (t22==t2).all() and (t33==t3).all() and (t44==t4).all():
            if Gbest.shape[0]<4:
                Gbest = np.array(gbest)
                Gbest = np.append(Gbest,totalfitness)
                T1=t1
                T2=t2
                T3=t3
                T4=t4
                T11 = t11
                T22 = t22
                T33 = t33
                T44=t44
            else:
                if totalfitness>Gbest[14]:
                    Gbest = np.array(gbest)
                    Gbest = np.append(Gbest, totalfitness)
                    T1 = t1
                    T2 = t2
                    T3 = t3
                    T4=t4
                    T11 = t11
                    T22 = t22
                    T33 = t33
                    T44=t44
                else:
                    Gbest=Gbest
        pre_state=first_state
        totalfitness=0
    else:
        pre_state=pre_state+1
print(Gbest)
plt.figure()
for i in range(14):
    plt.hlines(Gbest[i],Airplane[i][0],Airplane[i][1])
plt.figure()
plt.plot(np.linspace(1,N,N),p)
plt.show()
