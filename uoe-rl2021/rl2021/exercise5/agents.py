from abc import ABC, abstractmethod
from collections import defaultdict
import random
import sys
from typing import List, Dict, DefaultDict
import itertools

import numpy as np
from gym.spaces import Space, Box
from gym.spaces.utils import flatdim

from rl2021.exercise5.matrix_game import actions_to_onehot

def obs_to_tuple(obs):
    return tuple([tuple(o) for o in obs])


class MultiAgent(ABC):
    """Base class for multi-agent reinforcement learning

    **DO NOT CHANGE THIS BASE CLASS**

    """

    def __init__(
        self,
        num_agents: int,
        action_spaces: List[Space],
        observation_spaces: List[Space],
        gamma: float,
        **kwargs
    ):
        """Constructor of base agent for Q-Learning

        Initializes basic variables of MARL agents
        namely epsilon, learning rate and discount rate.

        :param num_agents (int): number of agents
        :param action_spaces (List[Space]): action spaces of the environment for each agent
        :param observation_spaces (List[Space]): observation spaces of the environment for each agent
        :param gamma (float): discount factor (gamma)

        :attr n_acts (List[int]): number of actions for each agent
        """

        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.observation_spaces = observation_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]

        self.gamma: float = gamma

    @abstractmethod
    def act(self, obs: List[np.ndarray]) -> List[int]:
        """Chooses an action for all agents given observations

        :param obs (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :return (List[int]): index of selected action for each agent
        """
        ...

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def learn(self):
        ...


class IndependentQLearningAgents(MultiAgent):
    """Agent using the Independent Q-Learning algorithm

    ** YOU NEED TO IMPLEMENT THE FUNCTIONS IN THIS CLASS **
    """

    def __init__(self, learning_rate: float =0.5, epsilon: float =1.0, **kwargs):
        """Constructor of IndependentQLearningAgents

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents


        :attr q_tables (List[DefaultDict]): tables for Q-values mapping (OBS, ACT) pairs of observations
            and actions to respective Q-values for all agents

        Initializes some variables of the Independent Q-Learning agents, namely the epsilon, discount rate
        and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for i in range(self.num_agents)]


    def act(self, obss: List[np.ndarray]) -> List[int]:
        """Implement the epsilon-greedy action selection here

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :return (List[int]): index of selected action for each agent
        """
        actions = []
        ### PUT YOUR CODE HERE ###
        
        # Generate random uniform variable
        for i in range(self.num_agents):
            
            obs = obss[i]
            
            rand_prob = random.uniform(0,1)
            
            # Following the codes in the lectures
            act_vals = [self.q_tables[i][(obs, act)] for act in range(self.n_acts[i])]
            max_val = max(act_vals)
            max_acts = [idx for idx, act_val in enumerate(act_vals) if act_val == max_val]
    
            if rand_prob < self.epsilon:
                action = random.randint(0, self.n_acts[i] - 1)
            else:
                action = random.choice(max_acts)
                
            actions.append(action)
        
        return actions


    def learn(
        self, obss: List[np.ndarray], actions: List[int], rewards: List[float], n_obss: List[np.ndarray], dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables based on agents' experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param n_obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the next environmental state for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current observation-action pair of each agent
        """
        updated_values = []
        ### PUT YOUR CODE HERE ###
        
        
        for i in range(self.num_agents):
            obs, action, reward, n_obs, done = obss[i], actions[i], rewards[i], n_obss[i], dones[i]
         
            if done == 0:
                act_vals = [self.q_tables[i][(n_obs, act)] for act in range(self.n_acts[i])]
                max_val = max(act_vals)
     
            else:
                max_val = 0
            
            # Update Q-table
            self.q_tables[i][(obs, action)] += self.learning_rate*(reward + self.gamma * max_val - self.q_tables[i][(obs, action)])
            
            # Add updated values
            updated_values.append(self.q_tables[i][(obs, action)])
        
        return updated_values


    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        self.epsilon = max(self.epsilon* 0.99** timestep, 0.001)


class JointActionLearning(MultiAgent):
    """Agents using the Joint Action Learning algorithm with Opponent Modelling

    ** YOU NEED TO IMPLEMENT THE FUNCTIONS IN THIS CLASS **
    """

    def __init__(self, learning_rate: float =0.5, epsilon: float =1.0, **kwargs):
        """Constructor of JointActionLearning

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr q_tables (List[DefaultDict]): tables for Q-values mapping (OBS, ACT) pairs of
            observations and joint actions to respective Q-values for all agents
        :attr models (List[DefaultDict[DefaultDict]]): each agent holding model of other agent
            mapping observation to other agent actions to count of other agent action

        Initializes some variables of the Joint Action Learning agents, namely the epsilon, discount
        rate and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_acts = [flatdim(action_space) for action_space in self.action_spaces]

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for _ in range(self.num_agents)]

        # initialise models for each agent mapping state to other agent actions to count of other agent action
        # in state
        self.models = [defaultdict(lambda: defaultdict(lambda: 0)) for _ in range(self.num_agents)] 

        # count observations - count for each agent
        self.c_obss = [defaultdict(lambda: 0) for _ in range(self.num_agents)]


    def act(self, obss: List[np.ndarray]) -> List[int]:
        """Implement the epsilon-greedy action selection here

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :return (List[int]): index of selected action for each agent
        """
        joint_action = []
        ### PUT YOUR CODE HERE ###
                
        # Generate random uniform variable
        for i in range(self.num_agents):
            
            obs = obss[i]
            opp_model = self.models[i]
            
            rand_prob = random.uniform(0,1)
                                    
            if rand_prob < self.epsilon:
                action = random.randint(0, self.n_acts[i] - 1)
            else:
                
                # Get joint action values
                joint_act_vals = []
                for act1 in range(self.n_acts[0]):
                    act1_lst = []
                    for act2 in range(self.n_acts[1]):
                        act1_lst.append(self.q_tables[i][(obs, (act1, act2))])
                    joint_act_vals.append(act1_lst)
                
                # Get opponent's action counts
                opp_act_counts = [opp_model[obs][act] for act in range(self.n_acts[i])]
                
                N_s = sum([self.c_obss[agent][obs] for agent in range(self.num_agents)])
                
                if N_s == 0:
                    action = random.randint(0, self.n_acts[i] - 1)
                else:
                    
                    # Compute EV
                    EV = [np.dot(joint_act_vals[act], opp_act_counts)/N_s for act in range(self.n_acts[i])]
                    
                    max_EV = max(EV)
                    argmax_EV = [idx for idx, ev in enumerate(EV) if ev == max_EV]
    
                    action = random.choice(argmax_EV)
                
            joint_action.append(action)
        
        
        return joint_action


    def learn(
        self, obss: List[np.ndarray], actions: List[int], rewards: List[float], n_obss: List[np.ndarray], dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables and models based on agents' experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the current environmental state for each agent
        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param n_obss (List[np.ndarray] of float with dim (observation size)):
            received observations representing the next environmental state for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current observation-action pair of each agent
        """
        updated_values = []
        ### PUT YOUR CODE HERE ###
        
        
        for i in range(self.num_agents):
            obs, action, reward, n_obs, done = obss[i], actions[i], rewards[i], n_obss[i], dones[i]
            
            # Update count for the observation by the agent
            self.c_obss[i][obs] += 1 
            
            # Update mapping dictionary for opponent action counts
            for j in range(self.num_agents):
                if j == i:
                    continue
                else:
                    self.models[j][obs][action] += 1
            
            if done == 0:
                
                # Get joint action values
                joint_act_vals = []
                for act1 in range(self.n_acts[0]):
                    act1_lst = []
                    for act2 in range(self.n_acts[1]):
                        act1_lst.append(self.q_tables[i][(n_obs, (act1, act2))])
                    joint_act_vals.append(act1_lst)
                
                # Get opponent's action counts
                opp_act_counts = [self.models[i][n_obs][act] for act in range(self.n_acts[i])]
                
                N_s = sum([self.c_obss[agent][n_obs] for agent in range(self.num_agents)])
                
                # Compute EV
                EV = [np.dot(joint_act_vals[act], opp_act_counts)/N_s for act in range(self.n_acts[i])]
                
                # Compute Max EV
                max_EV = max(EV)
     
            else:
                max_EV = 0
            
            # Update Q-table
            joint_actions = (actions[0], actions[1])
            self.q_tables[i][(obs, joint_actions)] += self.learning_rate * (reward + self.gamma * max_EV - self.q_tables[i][(obs, joint_actions)])
            self.q_tables[i][(obs, joint_actions)] = float(self.q_tables[i][(obs, joint_actions)])
            
            updated_values.append(self.q_tables[i][(obs, joint_actions)])
            
        return updated_values


    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        # raise NotImplementedError("Needed for Q5")
        self.epsilon = max(self.epsilon* 0.99** (timestep), 0.001)