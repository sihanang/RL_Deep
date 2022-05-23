from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Optional, Hashable

from rl2021.utils import MDP, Transition, State, Action


class MDPSolver(ABC):
    """Base class for MDP solvers

    **DO NOT CHANGE THIS CLASS**

    :attr mdp (MDP): MDP to solve
    :attr gamma (float): discount factor gamma to use
    :attr action_dim (int): number of actions in the MDP
    :attr state_dim (int): number of states in the MDP
    """

    def __init__(self, mdp: MDP, gamma: float):
        """Constructor of MDPSolver

        Initialises some variables from the MDP, namely the state and action dimension variables

        :param mdp (MDP): MDP to solve
        :param gamma (float): discount factor (gamma)
        """
        self.mdp: MDP = mdp
        self.gamma: float = gamma

        self.action_dim: int = len(self.mdp.actions)
        self.state_dim: int = len(self.mdp.states)

    def decode_policy(self, policy: Dict[int, np.ndarray]) -> Dict[State, Action]:
        """Generates greedy, deterministic policy dict

        Given a stochastic policy from state indeces to distribution over actions, the greedy,
        deterministic policy is generated choosing the action with highest probability

        :param policy (Dict[int, np.ndarray of float with dim (num of actions)]):
            stochastic policy assigning a distribution over actions to each state index
        :return (Dict[State, Action]): greedy, deterministic policy from states to actions
        """
        new_p = {}
        for state, state_idx in self.mdp._state_dict.items():
            new_p[state] = self.mdp.actions[np.argmax(policy[state_idx])]
        return new_p

    @abstractmethod
    def solve(self):
        """Solves the given MDP
        """
        ...


class ValueIteration(MDPSolver):
    """MDP solver using the Value Iteration algorithm
    """

    def _calc_value_func(self, theta: float) -> np.ndarray:
        """Calculates the value function

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        **DO NOT ALTER THE MDP HERE**

        Useful Variables:
        1. `self.mpd` -- Gives access to the MDP.
        2. `self.mdp.R` -- 3D NumPy array with the rewards for each transition.
            E.g. the reward of transition [3] -2-> [4] (going from state 3 to state 4 with action
            2) can be accessed with `self.R[3, 2, 4]`
        3. `self.mdp.P` -- 3D NumPy array with transition probabilities.
            *REMEMBER*: the sum of (STATE, ACTION, :) should be 1.0 (all actions lead somewhere)
            E.g. the transition probability of transition [3] -2-> [4] (going from state 3 to
            state 4 with action 2) can be accessed with `self.P[3, 2, 4]`

        :param theta (float): theta is the stop threshold for value iteration
        :return (np.ndarray of float with dim (num of states)):
            1D NumPy array with the values of each state.
            E.g. V[3] returns the computed value for state 3
        """
        V = np.zeros(self.state_dim)
        ### PUT YOUR CODE HERE ###
        
        while(True):
            delta = 0
            for s in np.arange(self.state_dim):
                v = V[s]
                
                vector_qty = self.mdp.P[s,:,:] * (self.mdp.R[s,:,:] + self.gamma * V)
                
                # Make the transition qty to itself be 0 as it shouldn't be in the sum
                vector_qty[:,s] = 0
                
                # Take the maximum across the actions (rows) after summing
                # across the state that has been transitioned to (columns).
                V[s] = np.max(np.sum(vector_qty, axis = 1))
                delta = max(delta, np.abs(v-V[s]))
            
            if delta < theta:
                break
        
        #raise NotImplementedError("Needed for Q1")
        return V

    def _calc_policy(self, V: np.ndarray) -> np.ndarray:
        """Calculates the policy

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        :param V (np.ndarray of float with dim (num of states)):
            A 1D NumPy array that encodes the computed value function (from _calc_value_func(...))
            It is indexed as (State) where V[State] is the value of state 'State'
        :return (np.ndarray of float with dim (num of states, num of actions):
            A 2D NumPy array that encodes the calculated policy.
            It is indexed as (STATE, ACTION) where policy[STATE, ACTION] has the probability of
            taking action 'ACTION' in state 'STATE'.
            REMEMBER: the sum of policy[STATE, :] should always be 1.0
            For deterministic policies the following holds for each state S:
            policy[S, BEST_ACTION] = 1.0
            policy[S, OTHER_ACTIONS] = 0
        """
        policy = np.zeros([self.state_dim, self.action_dim])
        ### PUT YOUR CODE HERE ###
        
        
        for s in np.arange(self.state_dim):
            vector_qty = self.mdp.P[s,:,:] * (self.mdp.R[s,:,:] + self.gamma * V)
            
            # Assign 0 to the transition to itself as this is not included in
            # the sum or argmax comparison.
            vector_qty[:,s] = 0
            
            # Get the best action
            best_action = np.argmax(np.sum(vector_qty, axis = 1))
            
            # Assign 1 to best action, the rest will default to 0
            policy[s, best_action] = 1
        
        #raise NotImplementedError("Needed for Q1")
        return policy

    def solve(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the MDP

        Compiles the MDP and then calls the calc_value_func and
        calc_policy functions to return the best policy and the
        computed value function

        **DO NOT CHANGE THIS FUNCTION**

        :param theta (float, optional): stop threshold, defaults to 1e-6
        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)):
            Tuple of calculated policy and value function
        """
        self.mdp.ensure_compiled()
        V = self._calc_value_func(theta)
        policy = self._calc_policy(V)

        return policy, V


class PolicyIteration(MDPSolver):
    """MDP solver using the Policy Iteration algorithm
    """

    def _policy_eval(self, policy: np.ndarray) -> np.ndarray:
        """Computes one policy evaluation step

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        :param policy (np.ndarray of float with dim (num of states, num of actions)):
            A 2D NumPy array that encodes the policy.
            It is indexed as (STATE, ACTION) where policy[STATE, ACTION] has the probability of
            taking action 'ACTION' in state 'STATE'.
            REMEMBER: the sum of policy[STATE, :] should always be 1.0
            For deterministic policies the following holds for each state S:
            policy[S, BEST_ACTION] = 1.0
            policy[S, OTHER_ACTIONS] = 0
        :return (np.ndarray of float with dim (num of states)): 
            A 1D NumPy array that encodes the computed value function
            It is indexed as (State) where V[State] is the value of state 'State'
        """
        V = np.zeros(self.state_dim)
        ### PUT YOUR CODE HERE ###
                
        while(True):
            delta = 0
            for s in np.arange(self.state_dim):
                v = V[s]
                
                vector_qty = self.mdp.P[s,:,:] * (self.mdp.R[s,:,:] + self.gamma * V)
                
                # Make the transition qty to itself be 0 as it shouldn't be in the sum
                vector_qty[:,s] = 0
                
                # Inner sum across the state that has been transitioned to (columns).
                # This inner_sum_vector will be of length: action_dim
                inner_sum_vector = np.sum(vector_qty, axis = 1)
                
                #print('Inner Sum ', inner_sum_vector)
            
                # Dot product of the action probabilities with inner_sum_vector
                V[s] = np.dot(policy[s,:], inner_sum_vector)
                
                delta = max(delta, np.abs(v-V[s]))
            
            if delta < self.theta:
                break
        
        
        #raise NotImplementedError("Needed for Q1")
        return np.array(V)

    def _policy_improvement(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes one policy improvement iteration

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        Useful Variables (As with Value Iteration):
        1. `self.mpd` -- Gives access to the MDP.
        2. `self.mdp.R` -- 3D NumPy array with the rewards for each transition.
            E.g. the reward of transition [3] -2-> [4] (going from state 3 to state 4 with action
            2) can be accessed with `self.R[3, 2, 4]`
        3. `self.mdp.P` -- 3D NumPy array with transition probabilities.
            *REMEMBER*: the sum of (STATE, ACTION, :) should be 1.0 (all actions lead somewhere)
            E.g. the transition probability of transition [3] -2-> [4] (going from state 3 to
            state 4 with action 2) can be accessed with `self.P[3, 2, 4]`

        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)):
            Tuple of calculated policy and value function
        """
        policy = np.zeros([self.state_dim, self.action_dim])
        V = np.zeros([self.state_dim])
        ### PUT YOUR CODE HERE ###
        
        # Do 2nd step: Policy Evaluation first
        V = self._policy_eval(policy)
        
        # 3rd Step: Policy Improvement
        
        # Initialise a counter for policy_stable, it should be accumulated
        # to state_dim before policy stable can be reassigned to True
        policy_stable_count = 0
        policy_stable = True
        
        while(True):
            
            # Reset counter
            policy_stable_count = 0
            
            for s in np.arange(self.state_dim):
                
                # Assign old action based on the policy
                a = np.argmax(policy[s,:])
                
                ######################
                # Compute new action
                ######################
                vector_qty = self.mdp.P[s,:,:] * (self.mdp.R[s,:,:] + self.gamma * V)
                
                # Assign 0 to the transition to itself as this is not included in
                # the sum or argmax comparison.
                vector_qty[:,s] = 0
                
                # Get the best action
                best_action = np.argmax(np.sum(vector_qty, axis = 1))
                
                # Assign 1 to best action, the rest will default to 0
                policy[s, best_action] = 1
                
                # If best_action not the same as a, assign policy-stable to false
                if best_action != a:
                    policy_stable = False
                
                else:
                    policy_stable_count += 1
            
            # If all states are policy stable, then we assign policy stable
            if policy_stable_count == self.state_dim:
                policy_stable = True
            
            if policy_stable == False:
                V = self._policy_eval(policy)
            else:
                break
        
        #raise NotImplementedError("Needed for Q1")
        return policy, V

    def solve(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the MDP

        This function compiles the MDP and then calls the
        policy improvement function that the student must implement
        and returns the solution

        **DO NOT CHANGE THIS FUNCTION**

        :param theta (float, optional): stop threshold, defaults to 1e-6
        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)]):
            Tuple of calculated policy and value function
        """
        self.mdp.ensure_compiled()
        self.theta = theta
        return self._policy_improvement()


if __name__ == "__main__":
    mdp = MDP()
    mdp.add_transition(
        #         start action end prob reward
        Transition("high", "wait", "high", 1, 2),
        Transition("high", "search", "high", 0.8, 5),
        Transition("high", "search", "low", 0.2, 5),
        Transition("high", "recharge", "high", 1, 0),
        Transition("low", "recharge", "high", 1, 0),
        Transition("low", "wait", "low", 1, 2),
        Transition("low", "search", "high", 0.6, -3),
        Transition("low", "search", "low", 0.4, 5),
    )

    solver = ValueIteration(mdp, 0.9)
    policy, valuefunc = solver.solve()
    print("---Value Iteration---")
    print("Policy:")
    print(solver.decode_policy(policy))
    print("Value Function")
    print(valuefunc)

    solver = PolicyIteration(mdp, 0.9)
    policy, valuefunc = solver.solve()
    print("---Policy Iteration---")
    print("Policy:")
    print(solver.decode_policy(policy))
    print("Value Function")
    print(valuefunc)