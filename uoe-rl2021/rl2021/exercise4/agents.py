import os
import gym
import numpy as np
from torch.optim import Adam
from typing import Dict, Iterable
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from rl2021.exercise3.agents import Agent
from rl2021.exercise3.networks import FCNetwork, Tanh2
from rl2021.exercise3.replay import Transition


class DDPG(Agent):
    """ DDPG

        ** YOU NEED TO IMPLEMENT THE FUNCTIONS IN THIS CLASS **

        :attr critic (FCNetwork): fully connected critic network
        :attr critic_optim (torch.optim): PyTorch optimiser for critic network
        :attr policy (FCNetwork): fully connected actor network for policy
        :attr policy_optim (torch.optim): PyTorch optimiser for actor network
        :attr gamma (float): discount rate gamma
        """

    def __init__(
            self,
            action_space: gym.Space,
            observation_space: gym.Space,
            gamma: float,
            critic_learning_rate: float,
            policy_learning_rate: float,
            critic_hidden_size: Iterable[int],
            policy_hidden_size: Iterable[int],
            tau: float,
            **kwargs,
    ):
        """
        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q4**

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param gamma (float): discount rate gamma
        :param critic_learning_rate (float): learning rate for critic optimisation
        :param policy_learning_rate (float): learning rate for policy optimisation
        :param critic_hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected critic
        :param policy_hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected policy
        :param tau (float): step for the update of the target networks
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.shape[0]

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        self.actor = FCNetwork(
            (STATE_SIZE, *policy_hidden_size, ACTION_SIZE), output_activation=Tanh2
        )
        self.actor_target = FCNetwork(
            (STATE_SIZE, *policy_hidden_size, ACTION_SIZE), output_activation=Tanh2
        )

        self.actor_target.hard_update(self.actor)

        self.critic = FCNetwork(
            (STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None
        )
        self.critic_target = FCNetwork(
            (STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None
        )
        self.critic_target.hard_update(self.critic)

        self.policy_optim = Adam(self.actor.parameters(), lr=policy_learning_rate, eps=1e-3)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_learning_rate, eps=1e-3)


        # ############################################# #
        # WRITE ANY EXTRA HYPERPARAMETERS YOU NEED HERE #
        # ############################################# #
        self.gamma = gamma
        self.critic_learning_rate = critic_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.tau = tau

        # ################################################### #
        # DEFINE A GAUSSIAN THAT WILL BE USED FOR EXPLORATION #
        # ################################################### #

        ### PUT YOUR CODE HERE ###
        # raise NotImplementedError("Needed for Q4")
        
        # Gaussian noise with mean 0 and std 0.1I
        self.noise = Normal(0, 0.1)

        # ############################### #
        # WRITE ANY AGENT PARAMETERS HERE #
        # ############################### #

        self.saveables.update(
            {
                "actor": self.actor,
                "actor_target": self.actor_target,
                "critic": self.critic,
                "critic_target": self.critic_target,
                "policy_optim": self.policy_optim,
                "critic_optim": self.critic_optim,
            }
        )


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


    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q4**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        # raise NotImplementedError("Needed for Q4")
        self.gamma = max(self.gamma *0.999999999, 0.98)
        

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q4**

        When explore is False you should select the best action possible (greedy). However, during exploration,
        you should be implementing exporation using the self.noise variable that you should have declared in the __init__.
        Use schedule_hyperparameters() for any hyperparameters that you want to change over time.

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        ### PUT YOUR CODE HERE ###
        # raise NotImplementedError("Needed for Q4")
        
        # Convert to pytorch tensor
        obs_torch = torch.from_numpy(obs)
        obs_torch = obs_torch.float()
        
        if explore == True:
            action = self.actor(obs_torch).squeeze(0) + self.noise.sample()
            action = action.detach().item()
        else:
            with torch.no_grad():
                action = self.actor(obs_torch).squeeze(0).detach().item()
        
        action = np.clip(action, -2., 2.)
        
        return [action]
        

    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DQN
        
        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q4**

        This function is called after storing a transition in the replay buffer. This happens
        every timestep. It should update your networks and return the q_loss and the policy_loss in the form of a
        dictionary.

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        ### PUT YOUR CODE HERE ###
        # raise NotImplementedError("Needed for Q4")
        
        state_batch = batch.states
        next_state_batch = batch.next_states
        action_batch = batch.actions
        reward_batch = batch.rewards
        done_batch = batch.done
        
        # print(batch)
        
        # Prepare for the target q batch
        critic_target_input_tensor = torch.cat([next_state_batch,
            self.actor_target(next_state_batch),
        ], 1)
        # print(critic_target_input_tensor)
        next_q_values = self.critic_target(critic_target_input_tensor)
        # next_q_values.volatile=False

        target_q_batch = reward_batch + self.gamma*(1-done_batch)*next_q_values

        # Critic update
        self.critic.zero_grad()
        critic_update_input_tensor = torch.cat([state_batch, action_batch], 1)
        q_batch = self.critic(critic_update_input_tensor)
        
        value_loss = F.mse_loss(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()
        actor_update_input_tensor = torch.cat([state_batch, self.actor(state_batch)], 1)
        policy_loss = -self.critic(actor_update_input_tensor)

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.policy_optim.step()
        
        q_loss = value_loss.detach().numpy()
        p_loss = policy_loss.detach().numpy()
        
        # Soft update
        self.actor_target.soft_update(self.actor,self.tau)
        self.critic_target.soft_update(self.critic, self.tau)
        
        
        self.saveables.update(
            {
                "actor": self.actor,
                "actor_target": self.actor_target,
                "critic": self.critic,
                "critic_target": self.critic_target,
                "policy_optim": self.policy_optim,
                "critic_optim": self.critic_optim,
            }
        )
        
        
        return {"q_loss": q_loss,
                "p_loss": p_loss}
