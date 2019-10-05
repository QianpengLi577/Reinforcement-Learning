import numpy as np
import random

def initmaze_r():#初始化迷宫和奖励函数
    maze=np.array([[1,1,1,0],
                   [1,0,1,0],
                   [1,0,1,1],
                   [1,1,0,2]])
    reward=np.zeros(16)
    for i in range(4):
        for j in range (4):
            if maze[i][j]==1:
                reward[4*i+j]=0
            if maze[i][j]==2:
                reward[4*i+j]=100
            if maze[i][j]==0:
                reward[4*i+j]=-100  #不可走地方设值不可很小
    return maze,reward
def initqtable():#初始化Qtable
    qtable=np.zeros((16,4))
    for i in range(16):
            x=i//4
            y=i%4
            if x==0and(y in range(4)):
                qtable[i][0]=-2
            if y==0:
                qtable[i][2]=-2
            if y==3:
                qtable[i][3]=-2
            if x==3and(y in range (4)):
                qtable[i][1]=-2
    return qtable
def epsilon_greedy_choose_action(qtable,prev_state,epsilon):
    x = prev_state // 4
    y = prev_state %4
    if random.uniform(0, 1) < epsilon:
        if x == 0 and y!=0and y!=3:   #第一行首尾
            return random.choice([1,2,3])
        elif y==0 and x!=0 and x!=3:  #第一列
            return random.choice([0,1,3])
        elif y==3 and x!=0 and x!=3:   #最后一列
            return random.choice([0,1,2])
        elif x==3 and y!=0 and y!=3:   #最后一行
            return random.choice([0,2,3])
        elif x==0and y==0:   #四个角
            return random.choice([1,3])
        elif x==0 and y==3:
            return random.choice([1,2])
        elif x==3 and y==0:
            return random.choice([0,3])
        elif x==3 and y==3:
            return random.choice([0,2])
        else:   #其他位置
            return random.randint(0,3)
    else:
        if (np.argwhere(qtable[prev_state]==max(qtable[prev_state]))).shape[0]>1:   #由于初始的时候qtable均为0，所以max有多个值   需要将不能走的点索引删去   然后在选择
            a=np.argwhere(qtable[prev_state]==max(qtable[prev_state]))
            if x == 0 and y != 0 and y != 3:  # 同上
                a = np.delete(a, np.where(a == 0))
                return random.choice(a)
            elif y == 0 and x != 0 and x != 3:
                a = np.delete(a, np.where(a == 2))
                return random.choice(a)
            elif y == 3 and x != 0 and x != 3:
                a = np.delete(a, np.where(a == 3))
                return random.choice(a)
            elif x == 3 and y != 0 and y != 3:
                a = np.delete(a, np.where(a == 1))
                return random.choice(a)
            elif x == 0 and y == 0:
                a = np.delete(a, np.where(a == 0))
                a = np.delete(a, np.where(a == 2))
                return random.choice(a)
            elif x == 0 and y == 3:
                a = np.delete(a, np.where(a == 0))
                a = np.delete(a, np.where(a == 3))
                return random.choice(a)
            elif x == 3 and y == 0:
                a = np.delete(a, np.where(a == 1))
                a = np.delete(a, np.where(a == 2))
                return random.choice(a)
            elif x == 3 and y == 3:
                a = np.delete(a, np.where(a == 1))
                a = np.delete(a, np.where(a == 3))
                return random.choice(a)
            else:
                return random.choice(a)
        else:
            return int(np.argwhere(qtable[prev_state]==max(qtable[prev_state])))    #只有一个最大值索引
def update_state(prev_state,action):
    x = prev_state // 4
    y = prev_state  % 4
    if action==0:
        return  (x-1)*4+y
    elif action==1:
        return  (x+1)*4+y
    elif action==2:
        return  x*4+y-1
    elif action==3:
        return x*4+y+1
first_state=0#这个指的是将迷宫展开成一位数组的横坐标索引
alpha = 0.3#学习系数
gamma = 0.9#gamma
epsilon = 0.3#epsilon
Maze,Reward=initmaze_r()
Qtable=initqtable()
pre_state=first_state
for m in range (100000):
    action=epsilon_greedy_choose_action(Qtable,pre_state,epsilon)
    state=update_state(pre_state,action)
    Qtable[pre_state][action]=Qtable[pre_state][action]+alpha*(Reward[state]+gamma*max(Qtable[state])-Qtable[pre_state][action])  #update
    if state==15:
        pre_state==first_state
    else :
        pre_state=state
# print(Maze)
# print(Reward)
print(Qtable)
P=np.array([first_state+1])
pre_state=first_state
action=int(np.argwhere(Qtable[pre_state]==max(Qtable[pre_state])))
state=update_state(pre_state,action)
while(state!=15):
    P=np.append(P,state+1)
    pre_state=state
    action = int(np.argwhere(Qtable[pre_state] == max(Qtable[pre_state])))
    state = update_state(pre_state, action)
P=np.append(P,16)
print(P)
