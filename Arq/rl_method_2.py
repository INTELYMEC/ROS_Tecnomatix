
# ============================================================================
# IMPORTS
# ============================================================================

from autonomous_decision_system import Autonomous_Decision_System
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# ============================================================================


# Sarsa:
class RL_Method_2(Autonomous_Decision_System):

    def __init__(self):
        Autonomous_Decision_System.__init__(self)

        self.alfa = 0.01
        self.gamma = 0.5
        self.epsilon_inicial = 1
        self.epsilon_minimo = 0.1
        self.lim_episodios = 229
        self.epsilon_decay = 0.99

        # numero de episodios y pasos
        self.ep_maximo = 500
        self.t_maximo = 100

        # inicializar recompensa por episodio
        self.r_episodio = np.arange(self.ep_maximo, dtype=float)

        # inicializar estados y acciones
        e1 = np.arange(60, 310, 10)
        e2 = np.repeat(e1, 25)
        e3 = np.arange(10, 60, 10)
        e4 = np.tile(e3, 125)
        e5 = np.arange(1, 6, 1)
        e6 = np.repeat(e5, 5)
        e7 = np.tile(e6, 25)
        e8 = np.column_stack((e2, e4))
        self.S = np.column_stack((e8, e7))  # 625 estados
        self.acciones = np.column_stack((e8, e7))  # 625 acciones

        # inicializar tabla Q
        self.Q = np.zeros((self.S.shape[0], self.acciones.shape[0]))

    # funcion elegir accion
    def elegir_accion(self, fila):
        p = np.random.random()
        if p < (1-self.epsilon):
            i = np.argmax(self.Q[fila, :])
        else:
            i = np.random.choice(2)
        return (i)

    # funcion RL- actualizar estados y matriz Q
    def process(self):
        writer = SummaryWriter()
        global epsilon
        epsilon = self.epsilon_inicial
        for n in range(self.ep_maximo):
            S0 = self.S[0]
            A0 = self.elegir_accion(0)
            t = 0
            r_acum = 0
            res0 = self.subscriber.update(S0)
            r_tot = res0
            while t < self.t_maximo:
                print("Episodio", n, "Paso", t)
                # buscar indice k del estado actual
                for k in range(625):
                    if self.S[k][0] == S0[0]:
                        if self.S[k][1] == S0[1]:
                            if self.S[k][2] == S0[2]:
                                break
                # actualizar estado
                Snew = self.acciones[A0]
                # actualizar resultado simulacion
                res1 = self.subscriber.update(Snew)
                # recompensa
                r = 1 / res1
                # buscar indice del estado nuevo S'
                for l in range(625):
                    if self.S[l][0] == Snew[0]:
                        if self.S[l][1] == Snew[1]:
                            if self.S[l][2] == Snew[2]:
                                break
                # tomar otra accion A'
                Anew = self.elegir_accion(l)
                # actualizar matriz Q
                self.Q[k, A0] = self.Q[k, A0] + self.alfa * (r +
                    self.gamma * self.Q[l, Anew] - self.Q[k, A0])
                # actualizar parametros
                t += 1
                S0 = Snew
                A0 = Anew
                r_acum = r_acum + r
                r_tot = r_tot + res1
            self.r_episodio[n] = r_acum
            if n >= self.lim_episodios:
                self.epsilon = self.epsilon_minimo
            else:
                self.epsilon = self.epsilon * self.epsilon_decay
                writer.add_scalar('Recompensa acumulada por episodio',
                                  r_acum, n)
                writer.add_scalar('Recompensa media', r_acum / self.t_maximo,
                                  n)
                writer.add_scalar('Epsilon', self.epsilon, n)
                writer.add_scalar('Resultado total a minimizar', r_tot, n)
                writer.add_scalar('Resultado medio a minimizar',
                                  r_tot / self.t_maximo, n)
