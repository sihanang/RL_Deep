from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
import random
from typing import List, Dict, DefaultDict
from gym.spaces import Space
from gym.spaces.utils import flatdim


class Agent(ABC):
    """Base class for Q-Learning agent

    **ONLY CHANGE THE BODY OF THE act() FUNCTION**

    """

    def __init__(
        self,
        action_space: Space,
        obs_space: Space,
        gamma: float,
        epsilon: float,
        **kwargs
    ):
        """Constructor of base agent for Q-Learning

        Initializes basic variables of the Q-Learning agent
        namely the epsilon, learning rate and discount rate.

        :param action_space (int): action space of the environment
        :param obs_space (int): observation space of the environment
        :param gamma (float): discount factor (gamma)
        :param epsilon (float): epsilon for epsilon-greedy action selection

        :attr n_acts (int): number of actions
        :attr q_table (DefaultDict): table for Q-values mapping (OBS, ACT) pairs of observations
            and actions to respective Q-values
        """

        self.action_space = action_space
        self.obs_space = obs_space
        self.n_acts = flatdim(action_space)

        self.epsilon: float = epsilon
        self.gamma: float = gamma

        self.q_table: DefaultDict = defaultdict(lambda: 0)

    def act(self, obs: np.ndarray) -> int:
        """Implement the epsilon-greedy action selection here

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        :param obs (np.ndarray of float with dim (observation size)):
            received observation representing the current environmental state
        :return (int): index of selected action
        """
        ### PUT YOUR CODE HERE ###
        
        
        # Generate random uniform variable
        rand_prob = random.uniform(0,1)
        
        # Following the codes in the lectures
        act_vals = [self.q_table[(obs, act)] for act in range(self.n_acts)]
        max_val = max(act_vals)
        max_acts = [idx for idx, act_val in enumerate(act_vals) if act_val == max_val]

        if rand_prob < self.epsilon:
            action = random.randint(0, self.n_acts - 1)
        else:
            action = random.choice(max_acts)
        
        
        #raise NotImplementedError("Needed for Q2")
        ### RETURN AN ACTION HERE ###
        return action

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


class QLearningAgent(Agent):
    """Agent using the Q-Learning algorithm

    """

    def __init__(self, alpha: float, **kwargs):
        """Constructor of QLearningAgent

        Initializes some variables of the Q-Learning agent, namely the epsilon, discount rate
        and learning rate alpha.

        :param alpha (float): learning rate alpha for Q-learning updates
        """

        super().__init__(**kwargs)
        self.alpha: float = alpha

    def learn(
        self, obs: np.ndarray, action: int, reward: float, n_obs: np.ndarray, done: bool
    ) -> float:
        """Updates the Q-table based on agent experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        :param obs (np.ndarray of float with dim (observation size)):
            received observation representing the current environmental state
        :param action (int): index of applied action
        :param reward (float): received reward
        :param n_obs (np.ndarray of float with dim (observation size)):
            received observation representing the next environmental state
        :param done (bool): flag indicating whether a terminal state has been reached
        :return (float): updated Q-value for current observation-action pair
        """
        ### PUT YOUR CODE HERE ###
        
        # Compute the maximum Q value for the next state
        
        if done == 0:
            act_vals = [self.q_table[(n_obs, act)] for act in range(self.n_acts)]
            max_val = max(act_vals)
 
        else:
            max_val = 0

        
        # Update Q-table
        self.q_table[(obs, action)] += self.alpha * (reward + self.gamma * max_val - self.q_table[(obs, action)])
        
        #raise NotImplementedError("Needed for Q2")
        return self.q_table[(obs, action)]

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q2")
        
        # As from references online such as https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
        # the hyperparameters should decrease over time
        
        # Define the final epsilon, gamma, and alpha to reach eventually
        final_ep = 0
        final_ga = 0.95
        final_al = 0.5
        
        self.epsilon = max(self.epsilon - (self.epsilon - final_ep) *(timestep/ max_timestep) ,0)
        self.gamma = max(self.gamma - (self.gamma - final_ga) *(timestep/ max_timestep), 0.95)
        self.alpha = max(self.alpha - (self.alpha - final_al) *(timestep/ max_timestep), 0.5)
        

class MonteCarloAgent(Agent):
    """Agent using the Monte-Carlo algorithm for training
    """

    def __init__(self, **kwargs):
        """Constructor of MonteCarloAgent

        Initializes some variables of the Monte-Carlo agent, namely epsilon,
        discount rate and an empty observation-action pair dictionary.

        :attr sa_counts (Dict[(Obs, Act), int]): dictionary to count occurrences observation-action pairs
        """
        super().__init__(**kwargs)
        self.sa_counts = {}

    def learn(
        self, obses: List[np.ndarray], actions: List[int], rewards: List[float]
    ) -> Dict:
        """Updates the Q-table based on agent experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        :param obses (List(np.ndarray) with numpy arrays of float with dim (observation size)):
            list of received observations representing environmental states of trajectory (in
            the order they were encountered)
        :param actions (List[int]): list of indices of applied actions in trajectory (in the
            order they were applied)
        :param rewards (List[float]): list of received rewards during trajectory (in the order
            they were received)
        :return (Dict): A dictionary containing the updated Q-value of all the updated state-action pairs
            indexed by the state action pair.
        """
        updated_values = {}
        ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q2")
        
        #  Within episode sa_counts to keep track of the counts
        within_ep_sa_counts = {}
        returns = 0
        
        first_visit = {}
        for sa_iteration in range(len(obses)):
            if first_visit.get((obses[sa_iteration], actions[sa_iteration]),-1) == -1:
                first_visit.update({(obses[sa_iteration], actions[sa_iteration]):sa_iteration})
        #print(first_visit)
        
        # Keep track how many times each observation-action pair occurs in sa_counts
        for sa_iteration in range(len(obses)-1,-1,-1):
            
            # Update counts for sa
            obs, action = obses[sa_iteration], actions[sa_iteration]
            
            curr_count = within_ep_sa_counts.get((obs, action),0)
            within_ep_sa_counts.update({(obs, action):curr_count+1})
            
            returns = self.gamma * returns + rewards[sa_iteration]
            
            if first_visit[(obs,action)] == sa_iteration:
                
                # Update Q by calling Q table and multiplying the sa_counts 
                # then adding the rewards of the first occurrence and divided
                # by the new number
                current_sa_count = self.sa_counts.get((obs, action),0)
                
                # Compute returns
                self.q_table[(obs, action)] = (self.q_table[(obs, action)] * current_sa_count + returns)/(current_sa_count +1)
                
                # Update updated_values
                updated_values.update({(obs, action): self.q_table[(obs, action)]})
                
                # Update sa_counts
                self.sa_counts[(obs, action)] = current_sa_count + 1
                
        
        return updated_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q2**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q2")
        
        self.epsilon = max(self.epsilon*0.9999999, 0.0001)
        #self.gamma = max(self.gamma*0.999999999999, 0.95)
        # if timestep % 100000 == 0:
        #     print("Current eps, timestep:",self.epsilon, timestep)
        #     print("Current eps, timestep:",self.gamma, timestep)
