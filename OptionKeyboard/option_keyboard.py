import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
from dataclasses import dataclass
import random

# gridworld implementation from online example
@dataclass
class State:
    features: np.ndarray
    terminal: bool = False

@dataclass
class TransitionData:
    state: State
    action: int
    reward: float
    next_state: State
    done: bool

class Environment:
    def __init__(self, n_objectives: int):
        self.n_objectives = n_objectives
    
    def reset(self) -> State:
        raise NotImplementedError
    
    def step(self, action: int) -> Tuple[State, np.ndarray, bool]:
        raise NotImplementedError
    
    @property
    def action_space(self) -> int:
        raise NotImplementedError

class GridWorld(Environment):
    def __init__(self, size: int = 8, n_objectives: int = 2):
        super().__init__(n_objectives)
        self.size = size
        self.position = [0, 0]
        self.grid = np.zeros((size, size, n_objectives))
        self._place_objectives()
    
    def _place_objectives(self):
        for obj in range(self.n_objectives):
            for _ in range(3):
                x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
                self.grid[x, y, obj] = 1.0
    
    def reset(self) -> State:
        self.position = [0, 0]
        return State(features=np.array(self.position))
    
    def step(self, action: int) -> Tuple[State, np.ndarray, bool]:
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = moves[action]
        
        new_x = max(0, min(self.size-1, self.position[0] + dx))
        new_y = max(0, min(self.size-1, self.position[1] + dy))
        self.position = [new_x, new_y]
        
        rewards = self.grid[new_x, new_y].copy()
        
        if np.any(rewards > 0):
            self.grid[new_x, new_y] = 0
            
        done = not np.any(self.grid > 0)
        
        return State(features=np.array(self.position)), rewards, done
    
    @property
    def action_space(self) -> int:
        return 4
    

# OptionKeyboard implementation from paper
class OptionKeyboard:
    def __init__(
        self,
        n_objectives: int,
        n_actions: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1
    ):
        self.n_objectives = n_objectives
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.weights = np.random.uniform(-0.1, 0.1, (n_objectives, n_actions))
        self.traces = np.zeros_like(self.weights)

    # randomly select action based on epsilon greedy policy    
    def select_action(self, state: State, objective_idx: int) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
            
        q_values = self.compute_q_values(state, objective_idx)
        return np.argmax(q_values)
    
    # getter function for q_values
    def compute_q_values(self, state: State, objective_idx: int) -> np.ndarray:
        return self.weights[objective_idx]
    
    # based on rl_step function from coursera
    def update(self, transition: TransitionData, objective_idx: int):
        current_q = self.weights[objective_idx, transition.action]
        
        if transition.done:
            next_q = 0
        else:
            next_q_values = self.compute_q_values(transition.next_state, objective_idx)
            next_q = np.max(next_q_values)
        
        td_target = transition.reward + self.gamma * next_q
        td_error = td_target - current_q
        self.weights[objective_idx, transition.action] += self.learning_rate * td_error

# Add training and evaluation loops