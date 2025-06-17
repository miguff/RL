import gym
import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

N_GAMES = 3000
T_MAX = 5

class SharedAdam(T.optim.Adam):
    """
    Share the Adam optimizer between teh different synchronous runs
    """

    def __init__(self, params, lr = 1e-3, betas = (0.9, 0.99), eps = 1e-8, weight_decay = 0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)




        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)

                #Tell Torhc we want to share the parameters for gradient descent
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class ActorCritic(nn.Module):
    """
    The model that chooses the action in a given scenario
    
    """

    def __init__(self, input_dims, n_actions, gamma = 0.99):
        super(ActorCritic, self).__init__()

        self.gamma = gamma

        self.pi1 = nn.Linear(*input_dims, 128) #Action function (Policy)
        self.v1 = nn.Linear(*input_dims, 128) #Value function
        self.pi = nn.Linear(128, n_actions)
        self.v = nn.Linear(128, 1)

        #Network has some basic memory
        self.rewards = []
        self.actions = []
        self.states = []

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)


    def clear_memory(self):
        self.rewards = []
        self.actions = []
        self.states = []

    def forward(self, state):
        """
        Comes a state, and decides what action values should it do, based on the actions available number of actions, and the value function
        """
        pi1 = F.relu(self.pi1(state)) 
        v1 = F.relu(self.v1(state))

        #It is passed to the relevant policy and value outputs
        pi = self.pi(pi1)
        v = self.v(v1)

        return pi, v
    
    def calc_R(self, done):
        """
        Calculates the reward
        """
        #Here we calculate the states to be torch datatype
        states = T.tensor(self.states, dtype=T.float)
        _, v = self.forward(states)

        #We are going over backward on the list, by always using the last element
        R = v[-1]*(1-int(done)) #If the episode is over, it gets multiplied by zero

        batch_return = []

        #Iterate over the stored data
        for reward in self.rewards[::-1]:
            R = reward + self.gamma*R
            batch_return.append(R) #Append the reward to the list
        batch_return.reverse() #Make it so that it is in the good order
        batch_return = T.tensor(batch_return, dtype=T.float)

        return batch_return
    
    def calc_loss(self, done):
        states = T.tensor(self.states, dtype=T.float)
        actions = T.tensor([a.item() for a in self.actions], dtype=T.float)
        
        returns = self.calc_R(done)

        pi, values = self.forward(states)

        values = values.squeeze() #Without the squeeze, it would not give us an error, but it will make bad calculations because it will be a wrong shape.
        critic_loss = (returns-values)**2

        probs = T.softmax(pi, dim=1) #Every action has a finite values, and every probability adds up to one
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions) #Calculate the log probability distribution of our actions actually taken

        actor_loss = -log_probs*(returns-values)
        total_loss = (critic_loss-actor_loss).mean()

        return total_loss


    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float)

        pi, v = self.forward(state)

        probs = T.softmax(pi, dim=0)
        dist = Categorical(probs)
        action = dist.sample().numpy()

        return action
    

class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions, gamma, lr, name, global_ep_idx, env_id):
        super(Agent,self).__init__()

        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        self.global_actor_critic = global_actor_critic
        self.name = "w%02i" % name

        self.episode_idx = global_ep_idx
        self.env = gym.make(env_id)
        self.optimizer = optimizer

    def run(self):
        t_step = 1
        while self.episode_idx.value < N_GAMES:
            done = False
            observation = self.env.reset()
            observation = observation[0]
            score = 0
            self.local_actor_critic.clear_memory()
            while not done:
                action = self.local_actor_critic.choose_action(observation)
                observation_, reward, done, _ ,_ = self.env.step(action)
                score += reward
                self.local_actor_critic.remember(observation, action, reward)
                if t_step % T_MAX == 0 or done: #If it ends, or we reached the max training step, it evaluates the loss
                    loss = self.local_actor_critic.calc_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(
                                                    self.local_actor_critic.parameters(), 
                                                    self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad #Here we make out global optimizers gradients equal with our local ones
                    #Load the gradient into the global, then az optimizer make an optimization, and it global and local also trains, but the global has some other trainings as well, from other runnings
                    self.optimizer.step()
                    #Synchronize the network
                    self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict()) #So here we synchronize by loading the global values into the locals one

                    self.local_actor_critic.clear_memory()
                t_step += 1
                observation = observation_
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            print(self.name, 'episode', self.episode_idx.value, 'reward %.1f' % score)



if __name__ == '__main__':
    mp.set_start_method('spawn') 
    lr = 1e-4
    env_id = 'CartPole-v0'
    n_actions = 2
    input_dims = [4]
    global_actor_critic = ActorCritic(input_dims, n_actions)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr, betas=(0.91, 0.999))
    global_ep = mp.Value('i', 0)

    workers = [Agent(global_actor_critic, 
                     optim, 
                     input_dims, 
                     n_actions, 
                     gamma=0.99, 
                     lr=lr, 
                     name=i, 
                     global_ep_idx=global_ep, 
                     env_id=env_id) for i in range(mp.cpu_count())]

    [w.start() for w in workers]
    [w.join() for w in workers]