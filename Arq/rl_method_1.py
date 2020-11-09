

from autonomous_decision_system import Autonomous_Decision_System
import numpy as np
import matplotlib.pyplot as plt


class RL_Method_1(Autonomous_Decision_System): 
   
    def __init__(self):    
        Autonomous_Decision_System.__init__(self)      
        
        # datos reinforcement learning
        self.alfa = 0.10
        self.gamma = 0.90
        self.epsilon = 1
        self.epsilon_minimo = 0.01
        self.epsilon_decay = 0.99

        # numero de episodios 
        self.ep_maximo = 1000

        # numero de pasos
        self.t_maximo = 100
        
        # inicializar recompensa por episodio
        self.r_episodio = np.arange (self.ep_maximo, dtype=float)

        # inicializar acciones
        self.acciones = np.array([[0,0,0], [10,0,0], [-10,0,0], [10,10,0], [10,-10,0], 
                                  [-10,10,0], [-10,-10,0], [10,10,1], [10,10,-1], [10,-10,1],
                                  [10,-10,-1], [-10,10,1], [-10,10,-1], [-10,-10,1], [-10,-10,-1],
                                  [0,10,0], [0,-10,0], [0,10,1], [0,10,-1], [0,-10,1], [0,-10,-1],
                                  [0,0,1], [0,0,-1], [10,0,1], [10,0,-1], [-10,0,1], [-10,1,-1]])

        # inicializar S
        e1 = np.arange(60,310,10)
        e2 = np.repeat(e1,25)
        e3 = np.arange(10,60,10)
        e4 = np.tile(e3,125)
        e5 = np.arange(1,6,1)
        e6 = np.repeat(e5,5)
        e7 = np.tile (e6,25)
        e8 = np.column_stack((e2,e4))
        self.S = np.column_stack((e8,e7))

        # inicializar tabla Q
        self.Q = np.zeros((self.S.shape[0], self.acciones.shape[0]))
  
    #funcion elegir accion
    def elegir_accion (self, fila):
        p = np.random.random()
        if p < (1-self.epsilon):
            i = np.argmax (self.Q[fila,:])
        else:
            i = np.random.choice (2)
        return (i)

    #funcion rl- actualizar estados y matriz Q
    def process(self):
        for n in range (self.ep_maximo):
            S0 = self.S[0]
            t = 0
            r_acum = 0
            res0 = self.subscriber.update(S0)
            while t < self.t_maximo:                
                #buscar indice k del estado actual
                for k in range(625):
                    if S[k][0]==S0[0]:
                        if S[k][1]==S0[1]:
                            if S[k][2]==S0[2]:
                                break
                #elegir accion de la fila k
                j = self.elegir_accion(k)
                #actualizar estado
                Snew = S0 + self.acciones[j]
                if Snew[0] > 300:
                    Snew[0] -= 10
                elif Snew[0] < 60:
                    Snew[0] += 10
                elif Snew[1] > 50:
                    Snew[1] -= 10
                elif Snew[1] < 10:
                    Snew[1] += 10
                elif Snew[2] > 5:
                    Snew[2] -= 1
                elif Snew[2] < 1:
                    Snew[2] += 1
                #actualizar resultado simulacion
                res1 = self.subscriber.update(Snew)
                #recompensa
                if res1 < res0 :
                    r = 1
                else:
                    r = 0
                #buscar indice del estado nuevo S'
                for l in range(625):
                    if S[l][0]==Snew[0]:
                        if S[l][1]==Snew[1]:
                            if S[l][2]==Snew[2]:
                                break
                #actualizar matriz Q
                self.Q[k,j] = self.Q[k,j]+self.alfa*(r+self.gamma*np.max(self.Q[l,:])-self.Q[k,j])
                #actualizar parametros
                t +=1
                S0 = Snew
                r_acum = r_acum + r
            r_episodio[n] = r_acum
            if epsilon > epsilon_minimo:
                epsilon = epsilon*epsilon_decay
