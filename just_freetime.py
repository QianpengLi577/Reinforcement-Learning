import numpy as np
import random
import math
import matplotlib.pyplot as plt
def initQtable():
    qtable=np.zeros((7,3))
    qtable[0][0]=0.5
    qtable[1][1]=0.5
    airplane=np.array([[685,765],
                       [690,790],
                       [705,775],
                       [795,900],
                       [860,960],
                       [880,930]])
    gate1=np.zeros((1,1))
    gate2=np.zeros((1,1))
    gate3=np.zeros((1,1))
    gate1[0]=685
    gate1=np.append(gate1,765)
    return qtable,airplane,gate1,gate2,gate3

def epsilon_greedy_choose_action(qtable,gate1,gate2,gate3,pre_state,airplane,epsilon):
    if random.uniform(0,1)<epsilon:   #随机选择
        return random.randint(0,2)
    else:
        m =np.argwhere(qtable[pre_state]==max(qtable[pre_state]))
        if m.shape[0]>1:
            return random.choice(m)
        else:
            return int(m)

def initReward(gate1,gate2,gate3,action,pre_state,airplane):
    if action==0:
        if gate1.shape[0]==2:
            return 0.5
        else:
            pre_time=gate1[gate1.shape[0]-3]
            now_time=airplane[pre_state][0]
            if now_time-pre_time>=30:
                return 1-(now_time-pre_time)/60
            else:
                return -2
    elif action==1:
        if gate2.shape[0]==2:
            return 0.5
        else:
            pre_time=gate2[gate2.shape[0]-3]
            now_time=airplane[pre_state][0]
            if now_time-pre_time>=30:
                return 1-(now_time-pre_time)/60
            else:
                return -2
    elif action==2:
        if gate3.shape[0]==2:
            return 0.5
        else:
            pre_time=gate3[gate3.shape[0]-3]
            now_time=airplane[pre_state][0]
            if now_time-pre_time>=30:
                return 1-(now_time-pre_time)/60
            else:
                return -2

def ch(t):
    j=2
    while(1):
       if t.shape[0]<=2:
           break
       else:
            if j<t.shape[0]:
                t[j]=t[j]-30
                j=j+2
            else:
                break
    return t
def fitness(t):
    j = 2
    temp=0
    while (1):
        if t.shape[0] <= 2:
            break
        else:
            if j < t.shape[0]:
                temp=temp+1-(t[j]-t[j-1])/60
                j = j + 2
            else:
                break
    return temp
first_state=1
alpha = 0.8#学习系数
gamma = 0.90#gamma
epsilon = 0.3#epsilon
Qtable,Airplane,Gate1,Gate2,Gate3=initQtable()
pre_state=first_state
Gbest=np.array([0])
T1=[]
T2=[]
T3=[]
T11=[]
T22=[]
T33=[]
for i in range(10000):
    if pre_state==first_state:
        Gate1 = np.zeros((1, 1))
        Gate2 = np.zeros((1, 1))
        Gate3 = np.zeros((1, 1))
        Gate1[0] = 685
        Gate1 = np.append(Gate1, 765)
        action = epsilon_greedy_choose_action(Qtable, Gate1, Gate2, Gate3, pre_state, Airplane, epsilon)
        gbest = [0]
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
    if pre_state<5:
        new_action = epsilon_greedy_choose_action(Qtable, Gate1, Gate2, Gate3, pre_state + 1, Airplane, epsilon)
    else:
        new_action=0
    reward = initReward(Gate1, Gate2, Gate3, action, pre_state, Airplane)
    Qtable[pre_state][action] = Qtable[pre_state][action] + alpha * (
                reward + gamma * Qtable[pre_state + 1][new_action] - Qtable[pre_state][action])
    action=new_action
    if pre_state==5:
        t11 = np.copy(Gate1)
        t22 = np.copy(Gate2)
        t33 = np.copy(Gate3)
        t1=np.copy(Gate1)
        t2 = np.copy(Gate2)
        t3 = np.copy(Gate3)
        t1=ch(t1)
        t2=ch(t2)
        t3=ch(t3)
        t11=ch(t11)
        t22=ch(t22)
        t33=ch(t33)
        t33.sort()
        t22.sort()
        t11.sort()
        if  (t11==t1).all() and (t22==t2).all() and (t33==t3).all():
            if Gbest.shape[0]<4:
                Gbest = np.array(gbest)
                Gbest = np.append(Gbest,fitness(t11)+fitness(t22)+fitness(t33))
                T1=t1
                T2=t2
                T3=t3
                T11 = t11
                T22 = t22
                T33 = t33
            else:
                if (fitness(t11)+fitness(t22)+fitness(t33))>Gbest[6]:
                    Gbest = np.array(gbest)
                    Gbest = np.append(Gbest, fitness(t11) + fitness(t22) + fitness(t33))
                    T1 = t1
                    T2 = t2
                    T3 = t3
                    T11 = t11
                    T22 = t22
                    T33 = t33
                else:
                    Gbest=Gbest
        pre_state=first_state
    else:
        pre_state=pre_state+1
print(Gbest)
for i in range(6):
    plt.hlines(Gbest[i],Airplane[i][0],Airplane[i][1])
plt.show()
