from abc import ABC, abstractmethod
from copy import deepcopy
import gym
import numpy as np
import os.path
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn
from torch.optim import Adam
from typing import Dict, Iterable, List
import random
import torch
from rl2021.exercise3.networks import FCNetwork
from rl2021.exercise3.replay import Transition, ReplayBuffer


class Agent(ABC):
    """Base class for Deep RL Exercise 3 Agents

    **DO NOT CHANGE THIS CLASS**

    :attr action_space (gym.Space): action space of used environment
    :attr observation_space (gym.Space): observation space of used environment
    :attr saveables (Dict[str, torch.nn.Module]):
        mapping from network names to PyTorch network modules

    Note:
        see http://gym.openai.com/docs/#spaces for more information on Gym spaces
    """

    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        """The constructor of the Agent Class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        """
        self.action_space = action_space
        self.observation_space = observation_space

        self.saveables = {}

    def save(self, path: str, suffix: str = "") -> str:
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "models_{suffix}.pt"
        where suffix is given by the optional parameter (by default empty string "")

        :param path (str): path to directory where to save models
        :param suffix (str, optional): suffix given to models file
        :return (str): path to file of saved models file
        """
        torch.save(self.saveables, path)
        return path

    def restore(self, save_path: str):
        """Restores PyTorch models from models file given by path

        :param save_path (str): path to file containing saved models
        """
        dirname, _ = os.path.split(os.path.abspath(__file__))
        save_path = os.path.join(dirname, save_path)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    @abstractmethod
    def act(self, obs: np.ndarray):
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
    def update(self):
        ...


class DQN(Agent):
    """The DQN agent for exercise 3

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**

    :attr critics_net (FCNetwork): fully connected DQN to compute Q-value estimates
    :attr critics_target (FCNetwork): fully connected DQN target network
    :attr critics_optim (torch.optim): PyTorch optimiser for DQN critics_net
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr update_counter (int): counter of updates for target network updates
    :attr target_update_freq (int): update frequency (number of iterations after which the target
        networks should be updated)
    :attr batch_size (int): size of sampled batches of experience
    :attr gamma (float): discount rate gamma
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        target_update_freq: int,
        batch_size: int,
        gamma: float,
        **kwargs,
    ):
        """The constructor of the DQN agent class

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param target_update_freq (int): update frequency (number of iterations after which the target
            networks should be updated)
        :param batch_size (int): size of sampled batches of experience
        :param gamma (float): discount rate gamma
        """
        super().__init__(action_space, observation_space)

        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        self.critics_net = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=None
        )

        self.critics_target = deepcopy(self.critics_net)

        self.critics_optim = Adam(
            self.critics_net.parameters(), lr=learning_rate, eps=1e-3
        )

        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.update_counter = 0
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = 1

        # ######################################### #

        self.saveables.update(
            {
                "critics_net": self.critics_net,
                "critics_target": self.critics_target,
                "critic_optim": self.critics_optim,
            }
        )

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q3")        
        
        self.epsilon = max(self.epsilon * 0.98,0.001)
        
        # Update learning rate
        self.learning_rate = max(self.learning_rate* 0.99** ((timestep/max_timestep)*1000), 0.0001)
        self.critics_optim = Adam(
            self.critics_net.parameters(), lr=self.learning_rate, eps=1e-3
        )

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        When explore is False you should select the best action possible (greedy). However, during
        exploration, you should be implementing an exploration strategy (like e-greedy). Use
        schedule_hyperparameters() for any hyperparameters that you want to change over time.

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        ### PUT YOUR CODE HERE ###
        # raise NotImplementedError("Needed for Q3")
        
        # Generate random uniform variable
        rand_prob = random.uniform(0,1)
        
        # Convert to pytorch tensor
        obs_torch = torch.from_numpy(obs)
        obs_torch = obs_torch.float()     
        
        if explore == True:
    
            if rand_prob < self.epsilon:
                action = random.randint(0, self.action_space.n - 1)
            else:
                with torch.no_grad():
                    action = self.critics_net(obs_torch).argmax().numpy()
        else:
            with torch.no_grad():
                action = self.critics_net(obs_torch).argmax().numpy()
            
        return action
        

    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DQN

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        This function is called after storing a transition in the replay buffer. This happens
        every timestep. It should update your network and return the Q-loss in the form of a
        dictionary.

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        ### PUT YOUR CODE HERE ###
        #raise NotImplementedError("Needed for Q3")
                        
        state_batch = batch.states
        action_batch = batch.actions.type(torch.LongTensor)
        reward_batch = batch.rewards
        next_states = batch.next_states
        done_batch = batch.done
        
        # print(next_states)
    
        # Compute Q
        state_action_values = self.critics_net(state_batch).gather(1, action_batch)
    
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        next_state_values = torch.zeros(len(state_batch))
        next_state_values = self.critics_target(next_states).max(1)[0]
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values.reshape([self.batch_size,1]) * self.gamma * (1-done_batch.reshape([self.batch_size,1]))) + reward_batch.reshape([self.batch_size,1])
    
        # Compute MSE Loss
        loss = torch.nn.functional.mse_loss(state_action_values, expected_state_action_values)
    
        # Optimize the model
        self.critics_optim.zero_grad()
        loss.backward()
        
        # Clamping between -1 and 1
        for para in self.critics_net.parameters():
            para.grad.data.clamp_(-1, 1)
        
        self.critics_optim.step()
        
        # Hard update
        self.update_counter += 1
        
        if (self.update_counter % self.target_update_freq) == 0:
            self.critics_target.hard_update(self.critics_net)
            
        self.saveables.update(
            {
                "critics_net": self.critics_net,
                "critics_target": self.critics_target,
                "critic_optim": self.critics_optim,
            }
        )
        
        q_loss = loss.item()
        return {"q_loss": q_loss}


class Reinforce(Agent):
    """ The Reinforce Agent for Ex 3

    ** YOU NEED TO IMPLEMENT THE FUNCTIONS IN THIS CLASS **

    :attr critics_net (FCNetwork): fully connected critic network to compute Q-value estimates
    :attr critics_target (FCNetwork): fully connected target critic network
    :attr critics_optim (torch.optim): PyTorch optimiser for critics network
    :attr actors_net (FCNetwork): fully connected actor network for policy
    :attr actors_optim (torch.optim): PyTorch optimiser for actor network
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr gamma (float): discount rate gamma
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        gamma: float,
        **kwargs,
    ):
        """
        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param gamma (float): discount rate gamma
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #

        ### DO NOT CHANGE THE OUTPUT ACTIVATION OF THIS POLICY ###
        self.policy = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=torch.nn.modules.activation.Softmax
        )

        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate, eps=1e-3)

        # ############################################# #
        # WRITE ANY EXTRA HYPERPARAMETERS YOU NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.gamma = gamma

        # ############################### #
        # WRITE ANY AGENT PARAMETERS HERE #
        # ############################### #

        # ###############################################
        self.saveables.update(
            {
                "policy": self.policy,
            }
        )

    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        """Updates the hyperparameters 

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        # raise NotImplementedError("Needed for Q3")
        # self.gamma = max(self.gamma* 0.9999,0.85)
        

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        Select an action from the model's stochastic policy by sampling a discrete action
        from the distribution specified by the model output

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        ### PUT YOUR CODE HERE ###
        # raise NotImplementedError("Needed for Q3")
        
        # Convert to pytorch tensor
        obs_torch = torch.from_numpy(obs)
        obs_torch = obs_torch.float()
        
        # print(self.critics_net(obs_torch).argmax().numpy())        
        
        if explore == True:
            prob_arr = self.policy(obs_torch).detach().numpy()
            action = np.random.choice(np.arange(len(prob_arr)), p = prob_arr)
            
        else:
            with torch.no_grad():
                action = self.policy(obs_torch).argmax().numpy()
            
        return action
        

    def update(
        self, rewards: List[float], observations: List[np.ndarray], actions: List[int],
        ) -> Dict[str, float]:
        """Update function for policy gradients

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        :param rewards (List[float]): rewards of episode (from first to last)
        :param observations (List[np.ndarray]): observations of episode (from first to last)
        :param actions (List[int]): applied actions of episode (from first to last)
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
            losses
        """
        ### PUT YOUR CODE HERE ###
        # raise NotImplementedError("Needed for Q3")
        
        discounted_rewards = []
        log_probs = []

        for t in range(len(rewards)):
            G = 0 
            for r in rewards[t:][::-1]:
                G = G * self.gamma + r
                
            obs = observations[t]
            obs_torch = torch.from_numpy(obs)
            obs_torch = obs_torch.float()
            log_probs.append(self.policy(obs_torch)[actions[t]].log())
            discounted_rewards.append(G)
            
        discounted_rewards = torch.tensor(discounted_rewards)
        
        
        policy_gradient = []
        for log_prob, G in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * G)
        
        self.policy_optim.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.policy_optim.step()
        
        p_loss = policy_gradient
        
        return {"p_loss": p_loss}