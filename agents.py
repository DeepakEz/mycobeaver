"""
Agents Module
Implements beaver agents with complete behaviors (§8-12, §15)
Part 1: Agent class and basic behaviors
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from config import (
    AgentConfig, TaskType, ActionType, AgentRole,
    PolicyConfig
)
from environment import Environment
from physarum import PhysarumNetwork
from pheromones import PheromoneField
from projects import ProjectManager, Project


@dataclass
class AgentState:
    """Internal agent state (§8)"""
    # Position
    position: int
    
    # Internal variables
    energy: float  # e_k^t
    satiety: float  # s_k^t
    wetness: float  # w_k^t
    
    # Task response thresholds (one per task type)
    thresholds: Dict[TaskType, float]  # θ_{k,m}
    
    # Role
    role: AgentRole  # r_k^t
    
    # Assigned project
    assigned_project: Optional[int]  # p_k^{proj,t}
    
    # Task history (for threshold adaptation)
    tasks_performed: Dict[TaskType, int]
    
    # Alive status
    alive: bool


class BeaverAgent:
    """
    Beaver agent with complete behavior specification (§8-12, §15)
    
    Features:
    - Energy, satiety, wetness dynamics (§12)
    - Task response thresholds (§9)
    - Role switching (§8)
    - Project assignment (§10)
    - Movement with pheromone and Physarum guidance (§11)
    - Local actions (§12)
    - Mortality (§12)
    """
    
    def __init__(
        self,
        agent_id: int,
        initial_position: int,
        config: AgentConfig
    ):
        self.id = agent_id
        self.cfg = config
        
        # Initialize state
        self.state = self._initialize_state(initial_position)
    
    def _initialize_state(self, position: int) -> AgentState:
        """Initialize agent state"""
        # Initial energy and satiety
        energy = self.cfg.initial_energy
        satiety = self.cfg.initial_satiety
        wetness = 0.0
        
        # Initialize thresholds (normal distribution)
        thresholds = {}
        for task in TaskType:
            threshold = np.random.normal(
                self.cfg.theta_mean,
                self.cfg.theta_std
            )
            threshold = max(0.1, threshold)  # Keep positive
            thresholds[task] = threshold
        
        # Initial role (worker)
        role = AgentRole.WORKER
        
        # No project assigned initially
        assigned_project = None
        
        # Task history
        tasks_performed = {task: 0 for task in TaskType}
        
        # Alive
        alive = True
        
        return AgentState(
            position=position,
            energy=energy,
            satiety=satiety,
            wetness=wetness,
            thresholds=thresholds,
            role=role,
            assigned_project=assigned_project,
            tasks_performed=tasks_performed,
            alive=alive
        )
    
    def is_alive(self) -> bool:
        """Check if agent is alive"""
        return self.state.alive
    
    def check_mortality(self) -> bool:
        """
        Check mortality conditions (§12)
        
        Returns True if agent dies
        """
        if not self.state.alive:
            return True
        
        # Death from energy depletion
        if self.state.energy <= self.cfg.energy_death_threshold:
            self.state.alive = False
            return True
        
        # Death from starvation (extended low satiety)
        if self.state.satiety <= self.cfg.satiety_death_threshold:
            self.state.alive = False
            return True
        
        return False
    
    def select_task(self, task_stimuli: Dict[TaskType, float]) -> TaskType:
        """
        Select task using response threshold mechanism (§9)
        
        P_k(take task m | S_m) = (S_m^n) / (S_m^n + θ_{k,m}^n)
        """
        n = self.cfg.n  # Steepness parameter
        
        # Compute probabilities
        probabilities = {}
        total = 0.0
        
        for task, stimulus in task_stimuli.items():
            threshold = self.state.thresholds[task]
            
            # Response probability
            prob = (stimulus ** n) / ((stimulus ** n) + (threshold ** n))
            probabilities[task] = prob
            total += prob
        
        # Normalize
        if total < 1e-8:
            # If no strong signal, return random task
            return np.random.choice(list(TaskType))
        
        for task in probabilities:
            probabilities[task] /= total
        
        # Sample task
        tasks = list(probabilities.keys())
        probs = [probabilities[t] for t in tasks]
        selected_task = np.random.choice(tasks, p=probs)
        
        # Update task history
        self.state.tasks_performed[selected_task] += 1
        
        return selected_task
    
    def assign_project(
        self,
        project_manager: ProjectManager,
        beta_R: float
    ) -> None:
        """
        Assign agent to project using waggle dance softmax (§10)
        """
        project_id = project_manager.select_project(beta_R)
        self.state.assigned_project = project_id
    
    def select_action(
        self,
        environment: Environment,
        pheromone_field: PheromoneField,
        physarum_network: PhysarumNetwork,
        project_manager: ProjectManager,
        task: TaskType
    ) -> Tuple[ActionType, Optional[int]]:
        """
        Select action based on task and local state (§11, §12)
        
        Returns:
            (action_type, target_cell)
        """
        current_cell = self.state.position
        
        # If at assigned project location, perform work
        if self.state.assigned_project is not None:
            project = project_manager.get_project(self.state.assigned_project)
            if project and current_cell in project.region:
                # Perform task-specific action
                return self._select_work_action(task, environment), None
        
        # Otherwise, move toward project or explore
        return self._select_movement_action(
            environment, pheromone_field, physarum_network, project_manager
        )
    
    def _select_work_action(
        self,
        task: TaskType,
        environment: Environment
    ) -> ActionType:
        """Select work action based on task"""
        current_cell = self.state.position
        
        if task == TaskType.FORAGE:
            # Check if vegetation available
            if environment.state.v[current_cell] > 0.5:
                return ActionType.FORAGE
            else:
                return ActionType.STAY
        
        elif task == TaskType.BUILD_DAM:
            # Check if dam can be built (permeability high)
            if environment.state.d[current_cell] > 0.3:
                return ActionType.BUILD_DAM
            else:
                return ActionType.STAY
        
        elif task == TaskType.REPAIR_DAM:
            # Check if dam exists and needs repair
            if environment.state.d[current_cell] < 0.7:
                return ActionType.PATCH_DAM
            else:
                return ActionType.STAY
        
        elif task == TaskType.LODGE_WORK:
            # Lodge work (simplified as staying)
            return ActionType.STAY
        
        elif task == TaskType.GUARD:
            return ActionType.STAY
        
        elif task == TaskType.SCOUT:
            # Scouts explore
            return ActionType.STAY
        
        else:
            return ActionType.STAY
    
    def _select_movement_action(
        self,
        environment: Environment,
        pheromone_field: PheromoneField,
        physarum_network: PhysarumNetwork,
        project_manager: ProjectManager
    ) -> Tuple[ActionType, Optional[int]]:
        """
        Select movement action using pheromone and Physarum guidance (§11)
        """
        current_cell = self.state.position
        neighbors = environment.state.neighbors[current_cell]
        
        if len(neighbors) == 0:
            return ActionType.STAY, None
        
        # Compute heuristic desirability for each neighbor
        heuristics = {}
        
        for neighbor in neighbors:
            # Physarum guidance
            physarum_score = physarum_network.get_edge_desirability(
                current_cell, neighbor
            )
            
            # Distance to assigned project (if any)
            project_score = 0.0
            if self.state.assigned_project is not None:
                project = project_manager.get_project(self.state.assigned_project)
                if project:
                    # Simple heuristic: closer to project center is better
                    proj_row, proj_col = project.center
                    neigh_row, neigh_col = environment._index_to_coords(neighbor)
                    distance = np.sqrt(
                        (neigh_row - proj_row) ** 2 +
                        (neigh_col - proj_col) ** 2
                    )
                    max_dist = np.sqrt(environment.H ** 2 + environment.W ** 2)
                    project_score = 1.0 - (distance / max_dist)
            
            # Safety (avoid deep water)
            water_depth = environment.state.h[neighbor]
            safety_score = max(0.0, 1.0 - water_depth / 3.0)
            
            # Combined heuristic
            eta = physarum_score + project_score + safety_score
            heuristics[neighbor] = max(1e-8, eta)
        
        # Get movement probabilities using pheromone field
        probabilities = pheromone_field.get_movement_probability(
            current_cell, heuristics
        )
        
        # Sample next cell
        next_cells = list(probabilities.keys())
        probs = [probabilities[c] for c in next_cells]
        next_cell = np.random.choice(next_cells, p=probs)
        
        # Determine movement action
        current_row, current_col = environment._index_to_coords(current_cell)
        next_row, next_col = environment._index_to_coords(next_cell)
        
        if next_row < current_row:
            action = ActionType.MOVE_UP
        elif next_row > current_row:
            action = ActionType.MOVE_DOWN
        elif next_col < current_col:
            action = ActionType.MOVE_LEFT
        elif next_col > current_col:
            action = ActionType.MOVE_RIGHT
        else:
            action = ActionType.STAY
        
        return action, next_cell
    
    def execute_action(
        self,
        action: ActionType,
        target_cell: Optional[int],
        environment: Environment,
        pheromone_field: PheromoneField,
        dt: float
    ) -> Dict[str, float]:
        """
        Execute action and update internal state (§12)
        
        Returns metrics about action execution
        """
        metrics = {}
        
        # Execute action
        if action == ActionType.MOVE_UP or \
           action == ActionType.MOVE_DOWN or \
           action == ActionType.MOVE_LEFT or \
           action == ActionType.MOVE_RIGHT:
            # Movement
            if target_cell is not None:
                old_position = self.state.position
                self.state.position = target_cell
                
                # Deposit pheromone on traversed edge
                pheromone_field.deposit(
                    old_position,
                    target_cell,
                    self.cfg.rho_tree  # Use as base deposition amount
                )
                
                metrics['moved'] = 1.0
            
            # Movement cost
            energy_cost = self.cfg.c_move_base
        
        elif action == ActionType.FORAGE:
            # Forage for food
            current_cell = self.state.position
            vegetation = environment.state.v[current_cell]
            
            if vegetation > 0:
                # Consume vegetation
                consumed = min(vegetation, self.cfg.rho_tree)
                environment.add_vegetation_consumption(current_cell, consumed)
                
                # Gain satiety
                self.state.satiety += self.cfg.beta_s * consumed
                self.state.satiety = min(1.0, self.state.satiety)
                
                # Gain energy
                self.state.energy += self.cfg.eta_food * consumed
                
                metrics['food_gathered'] = consumed
            
            energy_cost = self.cfg.c_work_base
        
        elif action == ActionType.FELL_TREE:
            # Fell tree (consume vegetation for dam building)
            current_cell = self.state.position
            environment.add_vegetation_consumption(current_cell, self.cfg.rho_tree)
            
            metrics['trees_felled'] = 1.0
            energy_cost = self.cfg.c_work_base * 1.5
        
        elif action == ActionType.BUILD_DAM or action == ActionType.PATCH_DAM:
            # Build or repair dam
            current_cell = self.state.position
            environment.add_building_effort(current_cell, self.cfg.rho_tree)
            
            metrics['dam_work'] = 1.0
            energy_cost = self.cfg.c_work_base * 2.0
        
        elif action == ActionType.HARVEST_MUD:
            # Harvest mud (simplified)
            energy_cost = self.cfg.c_work_base
        
        else:  # STAY
            energy_cost = self.cfg.c_move_base * 0.5
        
        # Update energy
        self.state.energy -= energy_cost * dt
        
        # Update satiety (decays over time)
        self.state.satiety -= self.cfg.alpha_s * dt
        self.state.satiety = max(-5.0, self.state.satiety)
        
        # Update wetness
        self._update_wetness(environment, dt)
        
        return metrics
    
    def _update_wetness(self, environment: Environment, dt: float) -> None:
        """
        Update wetness state (§12)
        
        w_{k}^{t+1} = (1 - α_w) * w_k + β_w * 1[h_{p_k} > h_wet] - γ_w * 1[p_k ∈ L]
        """
        current_cell = self.state.position
        
        # Decay
        self.state.wetness *= (1 - self.cfg.alpha_w * dt)
        
        # Increase if in deep water
        h_wet = 1.5  # Wetness threshold (from HydrologyConfig)
        if environment.state.h[current_cell] > h_wet:
            self.state.wetness += self.cfg.beta_w * dt
        
        # Decrease if in lodge
        if environment.state.L[current_cell] == 1:
            self.state.wetness -= self.cfg.gamma_w * dt
        
        # Clip to [0, 1]
        self.state.wetness = np.clip(self.state.wetness, 0, 1)
    
    def get_local_reward(self, environment: Environment, policy_config: PolicyConfig) -> float:
        """
        Compute local reward for greedy policy (§15.1)
        
        r_k^local = λ_E * Δe_k + λ_S * Δs_k + λ_safe * 1[safe] - λ_effort * c_move
        """
        # Energy term
        r_energy = policy_config.lambda_E * self.state.energy
        
        # Satiety term
        r_satiety = policy_config.lambda_S * self.state.satiety
        
        # Safety term (inverse of wetness and water danger)
        current_cell = self.state.position
        water_depth = environment.state.h[current_cell]
        is_lodge = environment.state.L[current_cell] == 1
        
        safe = 1.0 if (water_depth < 2.0 or is_lodge) else 0.0
        r_safe = policy_config.lambda_safe * safe
        
        # Effort penalty (wetness is discomfort)
        r_effort = -policy_config.lambda_effort * self.state.wetness
        
        return r_energy + r_satiety + r_safe + r_effort
    
    def get_state_dict(self) -> Dict:
        """Get agent state as dictionary for logging"""
        return {
            'id': self.id,
            'position': self.state.position,
            'energy': self.state.energy,
            'satiety': self.state.satiety,
            'wetness': self.state.wetness,
            'role': self.state.role.name,
            'assigned_project': self.state.assigned_project,
            'alive': self.state.alive
        }


class AgentPopulation:
    """
    Manager for population of beaver agents
    """
    
    def __init__(
        self,
        environment: Environment,
        config: AgentConfig
    ):
        self.env = environment
        self.cfg = config
        
        # Create agents
        self.agents = self._initialize_agents()
    
    def _initialize_agents(self) -> List[BeaverAgent]:
        """Initialize agent population"""
        agents = []
        
        # Place agents in core habitat region
        core_cells = list(self.env.core_habitat)
        
        for i in range(self.cfg.num_agents):
            # Random position in core habitat
            position = np.random.choice(core_cells)
            
            agent = BeaverAgent(
                agent_id=i,
                initial_position=position,
                config=self.cfg
            )
            agents.append(agent)
        
        return agents
    
    def get_alive_agents(self) -> List[BeaverAgent]:
        """Get list of alive agents"""
        return [a for a in self.agents if a.is_alive()]
    
    def get_num_alive(self) -> int:
        """Get number of alive agents"""
        return len(self.get_alive_agents())
    
    def get_agent_positions(self) -> List[Tuple[int, int]]:
        """Get positions of all alive agents for visualization"""
        positions = []
        for agent in self.get_alive_agents():
            row, col = self.env._index_to_coords(agent.state.position)
            positions.append((row, col))
        return positions
    
    def get_population_metrics(self) -> Dict[str, float]:
        """Get population-level metrics"""
        alive_agents = self.get_alive_agents()
        
        if len(alive_agents) == 0:
            return {
                'num_alive': 0,
                'mean_energy': 0.0,
                'mean_satiety': 0.0,
                'mean_wetness': 0.0
            }
        
        energies = [a.state.energy for a in alive_agents]
        satieties = [a.state.satiety for a in alive_agents]
        wetnesses = [a.state.wetness for a in alive_agents]
        
        return {
            'num_alive': len(alive_agents),
            'mean_energy': np.mean(energies),
            'mean_satiety': np.mean(satieties),
            'mean_wetness': np.mean(wetnesses),
            'min_energy': np.min(energies),
            'max_energy': np.max(energies)
        }
