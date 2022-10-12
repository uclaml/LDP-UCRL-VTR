import copy
import csv
import random

import numpy as np
from scipy.stats import bernoulli
from tqdm import tqdm
from multiprocessing import Manager, Pool, Process


class Environment(object):
    '''General RL environment'''

    def __init__(self):
        pass

    def reset(self):
        pass

    def advance(self, action):
        '''
        Moves one step in the environment.
        Args:
            action
        Returns:
            reward - double - reward
            newState - int - new state
            pContinue - 0/1 - flag for end of the episode
        '''
        return 0, 0, 0


class HardToLearn(Environment):
    def __init__(self, d, H, K):
        """
        Hard-to-learn MDP instance

        Args:
            d (int): dimension
            H (int): horizon
            K (int): num of episodes
        """        

        self.S = H + 2
        self.d = d
        self.H = H

        # for more details, see https://arxiv.org/pdf/2012.08507.pdf
        self.delta = 1 / H
        self.Delta = np.sqrt(self.delta / K) / (4 * np.sqrt(2))
        self.alpha = np.sqrt(1 / (1 + self.Delta * (d - 1)))
        self.beta = np.sqrt(self.Delta / (1 + self.Delta * (d - 1)))
        self.theta = self._get_theta()

        self.current_timestep = 0
        self.current_state = 0
             
    def _get_theta(self):
        mu = 2 * np.random.randint(low = 0, high = 2, size = (self.H, self.d - 1)) - np.ones(shape = (self.H, self.d - 1))
        mu = mu * self.Delta
        insert_alpha = np.ones(self.H) / self.alpha
        theta = np.insert(mu / self.beta, 0, insert_alpha, axis = 1)
        return theta
        
    def get_phi(self, s, action, s_prime):
        """
        get feature embeddings

        Args:
            s_prime (int): s_prime \in [self.S - 1], where self.S - 2 means x_{H + 1}, and self.S - 1 means x_{H + 2}
            action ([int]): for simplicity, action \in [0, 2^{d - 1} - 1] will be turned into a hypercube vector
        """        
        phi = np.zeros(self.d)
        bin_action = [int(idx) for idx in bin(action)[2:]]
        vec_action = np.zeros(self.d - 1)
        vec_action[-len(bin_action):] = bin_action
        vec_action = 2 * vec_action - np.ones(self.d - 1)
        if s not in [self.S - 2, self.S - 1]:
            if s_prime == s + 1:
                phi[0] = self.alpha * (1 - self.delta)
                phi[1:] = -self.beta * vec_action
            elif s_prime == self.S - 1:
                phi[0] = self.alpha * self.delta
                phi[1:] = self.beta * vec_action
        elif s_prime == s:
            phi[0] = self.alpha
        
        return phi

    def reset(self):
        "Resets the Environment"
        self.current_timestep = 0
        self.current_state = 0

    def advance(self, action):
        '''
        Move one step in the environment
        Args:
        action - int - chosen action
        Returns:
        reward - double - reward
        s_primt - int - new state
        episodeEnd - 0/1 - flag for end of the episode
        '''
        # get reward
        reward = 1 if self.current_state == (self.S - 1) else 0
        # get next state
        if self.current_state in [self.S - 2, self.S - 1]:
            s_prime = self.current_state
        else:
            phi_s_p_1 = self.get_phi(s = self.current_state, action = action, s_prime = self.current_state + 1)
            phi_absorb = self.get_phi(s = self.current_state, action = action, s_prime = self.S - 1)
            s_prime = np.random.choice([self.current_state + 1, self.S - 1], p = [np.dot(self.theta[self.current_timestep], phi_s_p_1), np.dot(self.theta[self.current_timestep], phi_absorb)])
        # update
        self.current_state = s_prime
        self.current_timestep += 1
        episodeEnd = 0
        if self.current_timestep == self.H:
            episodeEnd = 1
            self.reset()

        return reward, s_prime, episodeEnd


class LDP_UCRL_VTR(object):
    def __init__(self, env, K, epsilon = 'infty'):
        self.env = env
        self.K = K
        self.H = self.env.H
        self.d = self.env.d
        self.epsilon = epsilon
        self.S = self.env.S
        self.A = np.power(2, self.d - 1)

        # for users
        self.Q = np.zeros(shape = (self.H + 1, self.S, self.A))
        self.V = np.zeros(shape = (self.H + 1, self.S))
        
        # for the server
        self.Lambda = np.array([np.eye(self.d) for _ in range(self.H)])
        self.u = np.zeros((self.H, self.d))
        # self.beta = 9e-2
        self.beta = 3e0

    def get_phi_V(self, s, a, h):
        phi_V = np.zeros(self.d)
        for s_ in range(self.S):
            phi = self.env.get_phi(s = s, action = a, s_prime = s_, )
            phi_V += self.V[h + 1, s_] * phi
        return phi_V

    def _proj(self, x, lo, hi):
        return max(min(x, hi), lo)
    
    def get_sigma(self):
        return 4 * np.power(self.H, 3) * np.sqrt(2 * np.log(2.5 * self.H / 0.01)) / self.epsilon
     
    def update_user(self, k):
        for h in range(self.H - 1, -1, -1):
            if self.epsilon == 'infty':
                Sigma_inv = np.linalg.inv(self.Lambda[h])
            else:
                sigma = self.get_sigma()
                shift_r = 2 * np.sqrt(k + 1) * sigma * (2 * np.log(600 * self.H) + np.sqrt(4 * self.d))
                Sigma_inv = np.linalg.inv(self.Lambda[h] + shift_r * np.eye(self.d))
            theta_h = np.dot(Sigma_inv, self.u[h])
            for s in range(self.S):
                for a in range(self.A):
                    self._update_step(s, a, h, k, Sigma_inv, theta_h)
                    
    def _update_step(self, s, a, h, k, Sigma_inv, theta_h):
        X = self.get_phi_V(s, a, h)
        reward = 1 if s == self.S - 1 else 0
        UCB = self.UCB(h, k) * np.sqrt(np.dot(np.dot(np.transpose(X), Sigma_inv), X))
        if k == self.K - 1 and s == 0:
            print(UCB)
        self.Q[h, s, a] = self._proj(reward + np.dot(np.transpose(X), theta_h) + UCB, 0, self.H)
        self.V = np.max(self.Q, axis = -1)

    def update_server(self, s, a, s_, h):
        X = self.get_phi_V(s, a, h)
        y = self.V[h + 1, s_]
        if self.epsilon == 'infty':
            self.Lambda[h] = self.Lambda[h] + np.outer(X, X)
            self.u[h] = self.u[h] + y * X
        else:
            sigma = self.get_sigma()
            W = np.triu(np.random.normal(0, sigma, size = (self.d, self.d)))
            W = W + np.transpose(W) - np.diag(np.diag(W))
            xi = np.random.normal(0, sigma, size = self.d)
            self.Lambda[h] = self.Lambda[h] + np.outer(X, X) + W
            self.u[h] = self.u[h] + y * X + xi

    def UCB(self, h, k):
        if self.epsilon == 'infty':
            beta = self.beta * np.power(self.d, 3 / 4) * np.power(self.H - h, 3 / 2) * np.sqrt(np.log(k + 1)) * np.power(k, 1 / 4)
        else:
            beta = self.beta * np.power(self.d, 3 / 4) * np.power(self.H - h, 3 / 2) * np.sqrt(np.log(k + 1)) * np.power(k, 1 / 4) / np.sqrt(self.epsilon)
        return beta

    def current_regret(self):
        H = self.H
        S = self.S
        A = self.A
        Q_star = np.zeros((H + 1, S, A))
        Q_pi = np.zeros((H + 1, S, A))
        V_star = np.zeros((H + 1, S))
        V_pi = np.zeros((H + 1, S))
        for h in range(H - 1, -1, -1):
            for s in range(S):
                for a in range(A):
                    reward = 1 if s == S - 1 else 0
                    a_pi = np.argmax(self.Q[h, s])
                    if s in [S - 2, S - 1]:
                        Q_star[h, s, a] = reward + V_star[h + 1, s]
                        Q_pi[h, s, a] = reward + V_pi[h + 1, s]
                    else:
                        phi_s_p_1 = self.env.get_phi(s = s, action = a, s_prime = s + 1)
                        phi_absorb = self.env.get_phi(s = s, action = a, s_prime = S - 1)
                        p_sp1 = np.dot(self.env.theta[h], phi_s_p_1)
                        p_absorb = np.dot(self.env.theta[h], phi_absorb)
                        Q_star[h, s, a] = reward + p_sp1 * V_star[h + 1, s + 1] + p_absorb * V_star[h + 1, S - 1]
                        Q_pi[h, s, a] = reward + p_sp1 * V_pi[h + 1, s + 1] + p_absorb * V_pi[h + 1, S - 1]
                V_star[h, s] = Q_star[h, s].max()
                V_pi[h, s] = Q_pi[h, s, a_pi]
        return(V_star[0, 0] - V_pi[0, 0])

    def run(self):
        cumulative_Regret = []
        for k in tqdm(range(self.K)):
            self.env.reset()
            if k != 0:
                self.update_user(k)
            done = 0
            reward = 0
            while not done:
                s = self.env.current_state
                h = self.env.current_timestep
                a = np.argmax(self.Q[h, s])
                r, s_, done = self.env.advance(a)
                reward += r
                self.update_server(s, a, s_, h)
            
            last = 0
            if k > 0:
                last = cumulative_Regret[k - 1]
            regret = self.current_regret()
            cumulative_Regret.append(regret + last)
        return cumulative_Regret


if __name__ == '__main__':
    d = 5
    H = 6
    K = 10000
    env = HardToLearn(d = d, H = H, K = K)
    eval_rounds = 10
    for i in range(eval_rounds):
        agent = LDP_UCRL_VTR(env, K = K, epsilon = 0.1)
        # agent = LDP_UCRL_VTR(env, K = K, epsilon = 10)
        # agent = LDP_UCRL_VTR(env, K = K, epsilon = 1)
        # agent = LDP_UCRL_VTR(env, K = K, epsilon = 'infty')
        Regret_vec = agent.run()
        with open('UCRL01_{}.csv'.format(i), 'w', newline = '') as csvfile:
        # with open('UCRL10_{}.csv'.format(i), 'w', newline = '') as csvfile:
        # with open('UCRL1_{}.csv'.format(i), 'w', newline = '') as csvfile:
        # with open('UCRL_{}.csv'.format(i), 'w', newline = '') as csvfile:
            writer  = csv.writer(csvfile)
            writer.writerow(Regret_vec)
