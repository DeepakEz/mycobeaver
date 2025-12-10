"""
Overmind Module
Implements Contemplative Overmind with Architect Cognitive Prior (§14, §18)
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

from config import OvermindConfig


@dataclass
class MetaParameters:
    """Meta-parameters controlled by Overmind (§14.4)"""
    rho: float  # Pheromone evaporation rate
    beta_R: float  # Recruitment sharpness
    gamma_dance: float  # Recruitment gain
    task_stimuli: Dict  # Task stimulus intensities


class ContemplativeOvermind:
    """
    Contemplative Overmind with Architect Cognitive Prior (§14, §18)
    
    Features:
    - Wisdom signal aggregation (§14.1, §14.2)
    - Meta-parameter adaptation (§14.4)
    - ACP-informed decision making (§18.2, §18.3)
    - Brittleness avoidance (§18.1)
    """
    
    def __init__(self, config: OvermindConfig):
        self.cfg = config
        
        # Initialize meta-parameters at midpoints
        self.meta_params = self._initialize_meta_params()
        
        # Wisdom history
        self.wisdom_history = []
        self.max_history_length = 100
    
    def _initialize_meta_params(self) -> MetaParameters:
        """Initialize meta-parameters at reasonable defaults"""
        rho = (self.cfg.rho_min + self.cfg.rho_max) / 2
        beta_R = (self.cfg.beta_R_min + self.cfg.beta_R_max) / 2
        gamma_dance = (self.cfg.gamma_dance_min + self.cfg.gamma_dance_max) / 2
        
        # Initial task stimuli (will be updated based on needs)
        from config import TaskType
        task_stimuli = {task: 5.0 for task in TaskType}
        
        return MetaParameters(
            rho=rho,
            beta_R=beta_R,
            gamma_dance=gamma_dance,
            task_stimuli=task_stimuli
        )
    
    def update_meta_parameters(
        self,
        wisdom_signal: float,
        metrics: Dict[str, float]
    ) -> MetaParameters:
        """
        Update meta-parameters based on wisdom signal and metrics (§14.4, §18.3)
        
        Overmind adapts parameters to:
        1. Increase exploration when brittleness is high
        2. Sharpen consensus when system is stable
        3. Shift labor allocation based on needs
        """
        # Store wisdom
        self.wisdom_history.append(wisdom_signal)
        if len(self.wisdom_history) > self.max_history_length:
            self.wisdom_history.pop(0)
        
        # Extract key metrics
        structural_entropy = metrics.get('structural_entropy', 0.0)
        brittleness = metrics.get('brittleness', 0.0)
        network_degenerate = metrics.get('network_degenerate', 0.0)
        h_std_core = metrics.get('h_std_core', 0.0)
        num_flood = metrics.get('num_flood_cells', 0)
        
        # Adaptation rules (§18.3)
        
        # 1. Pheromone evaporation
        # Increase when structure is too rigid or brittle
        if structural_entropy < 2.0 or brittleness > 5.0:
            # Need more exploration
            target_rho = self.cfg.rho_max
        else:
            # Normal exploitation
            target_rho = self.cfg.rho_min + 0.3 * (self.cfg.rho_max - self.cfg.rho_min)
        
        # Smooth adaptation
        rho_new = 0.9 * self.meta_params.rho + 0.1 * target_rho
        rho_new = np.clip(rho_new, self.cfg.rho_min, self.cfg.rho_max)
        
        # 2. Recruitment sharpness
        # Lower when need to explore more options
        if network_degenerate > 0.5 or brittleness > 5.0:
            # Reduce consensus sharpness
            target_beta_R = self.cfg.beta_R_min
        elif h_std_core < 0.5:
            # System is stable, can increase focus
            target_beta_R = self.cfg.beta_R_max
        else:
            # Moderate sharpness
            target_beta_R = (self.cfg.beta_R_min + self.cfg.beta_R_max) / 2
        
        beta_R_new = 0.9 * self.meta_params.beta_R + 0.1 * target_beta_R
        beta_R_new = np.clip(beta_R_new, self.cfg.beta_R_min, self.cfg.beta_R_max)
        
        # 3. Recruitment gain
        # Increase when need more coordination
        if h_std_core > 1.0:
            # Need more coordinated action
            target_gamma = self.cfg.gamma_dance_max
        else:
            target_gamma = self.cfg.gamma_dance_min + 0.5 * (
                self.cfg.gamma_dance_max - self.cfg.gamma_dance_min
            )
        
        gamma_dance_new = 0.9 * self.meta_params.gamma_dance + 0.1 * target_gamma
        gamma_dance_new = np.clip(
            gamma_dance_new,
            self.cfg.gamma_dance_min,
            self.cfg.gamma_dance_max
        )
        
        # 4. Task stimuli
        # Adjust based on environmental needs
        task_stimuli_new = self._adapt_task_stimuli(metrics)
        
        # Create new meta-parameters
        new_params = MetaParameters(
            rho=rho_new,
            beta_R=beta_R_new,
            gamma_dance=gamma_dance_new,
            task_stimuli=task_stimuli_new
        )
        
        self.meta_params = new_params
        return new_params
    
    def _adapt_task_stimuli(self, metrics: Dict[str, float]) -> Dict:
        """
        Adapt task stimulus intensities based on needs (§14.4, §18.3)
        """
        from config import TaskType
        
        # Extract relevant metrics
        h_std_core = metrics.get('h_std_core', 0.0)
        num_flood = metrics.get('num_flood_cells', 0)
        mean_satiety = metrics.get('mean_satiety', 0.5)
        
        # Start with current stimuli
        new_stimuli = self.meta_params.task_stimuli.copy()
        
        # Foraging stimulus
        if mean_satiety < 0.3:
            # Colony is hungry, increase forage stimulus
            new_stimuli[TaskType.FORAGE] = min(10.0, new_stimuli[TaskType.FORAGE] + 0.5)
        else:
            # Reduce foraging
            new_stimuli[TaskType.FORAGE] = max(1.0, new_stimuli[TaskType.FORAGE] - 0.1)
        
        # Dam building stimulus
        if h_std_core > 1.0:
            # Water is unstable, increase dam building
            new_stimuli[TaskType.BUILD_DAM] = min(10.0, new_stimuli[TaskType.BUILD_DAM] + 0.5)
        else:
            new_stimuli[TaskType.BUILD_DAM] = max(1.0, new_stimuli[TaskType.BUILD_DAM] - 0.1)
        
        # Dam repair stimulus
        if num_flood > 5:
            # Flooding detected, increase repair urgency
            new_stimuli[TaskType.REPAIR_DAM] = min(10.0, new_stimuli[TaskType.REPAIR_DAM] + 0.5)
        else:
            new_stimuli[TaskType.REPAIR_DAM] = max(1.0, new_stimuli[TaskType.REPAIR_DAM] - 0.1)
        
        # Scout stimulus (exploration)
        brittleness = metrics.get('brittleness', 0.0)
        if brittleness > 5.0:
            # Need more exploration
            new_stimuli[TaskType.SCOUT] = min(10.0, new_stimuli[TaskType.SCOUT] + 0.3)
        else:
            new_stimuli[TaskType.SCOUT] = max(1.0, new_stimuli[TaskType.SCOUT] - 0.1)
        
        # Lodge work and guard (lower priority)
        new_stimuli[TaskType.LODGE_WORK] = 2.0
        new_stimuli[TaskType.GUARD] = 2.0
        
        return new_stimuli
    
    def get_wisdom_trend(self) -> float:
        """Get trend in wisdom signal (positive = improving)"""
        if len(self.wisdom_history) < 10:
            return 0.0
        
        recent = self.wisdom_history[-10:]
        
        # Simple linear regression
        x = np.arange(len(recent))
        y = np.array(recent)
        
        # Slope
        slope = np.polyfit(x, y, 1)[0]
        
        return slope
    
    def should_intervene(self, metrics: Dict[str, float]) -> Tuple[bool, str]:
        """
        Determine if Overmind should make major intervention (§18.3)
        
        Returns:
            (should_intervene, reason)
        """
        # Check for critical conditions
        
        # 1. Population collapse
        num_alive = metrics.get('num_alive', 0)
        if num_alive < 5:
            return True, "population_collapse"
        
        # 2. Severe brittleness
        brittleness = metrics.get('brittleness', 0.0)
        if brittleness > 10.0:
            return True, "severe_brittleness"
        
        # 3. Network degeneration
        network_degenerate = metrics.get('network_degenerate', 0.0)
        if network_degenerate > 0.5:
            return True, "network_degenerate"
        
        # 4. Persistent declining wisdom
        if len(self.wisdom_history) >= 20:
            recent_trend = self.get_wisdom_trend()
            if recent_trend < -0.5:
                return True, "declining_wisdom"
        
        return False, ""
    
    def execute_intervention(self, reason: str) -> MetaParameters:
        """
        Execute major intervention to reset system (§18.3)
        
        This is Overmind's "catastrophic reset" when system is stuck
        """
        from config import TaskType
        
        if reason == "population_collapse":
            # Emergency: maximize survival
            new_params = MetaParameters(
                rho=self.cfg.rho_min,  # Preserve trails
                beta_R=self.cfg.beta_R_min,  # Explore all options
                gamma_dance=self.cfg.gamma_dance_max,  # High coordination
                task_stimuli={
                    TaskType.FORAGE: 10.0,  # Priority: food
                    TaskType.BUILD_DAM: 1.0,
                    TaskType.REPAIR_DAM: 1.0,
                    TaskType.LODGE_WORK: 1.0,
                    TaskType.GUARD: 1.0,
                    TaskType.SCOUT: 5.0  # High exploration
                }
            )
        
        elif reason == "severe_brittleness":
            # Increase exploration, reduce commitment
            new_params = MetaParameters(
                rho=self.cfg.rho_max,  # High evaporation
                beta_R=self.cfg.beta_R_min,  # Low consensus
                gamma_dance=self.cfg.gamma_dance_min,  # Low recruitment
                task_stimuli={task: 5.0 for task in TaskType}  # Balanced
            )
        
        elif reason == "network_degenerate":
            # Force network diversity
            new_params = MetaParameters(
                rho=self.cfg.rho_max,  # High evaporation
                beta_R=self.cfg.beta_R_min,  # Explore projects
                gamma_dance=self.cfg.gamma_dance_min,  # Reduce lock-in
                task_stimuli={
                    TaskType.SCOUT: 10.0,  # High scouting
                    TaskType.FORAGE: 5.0,
                    TaskType.BUILD_DAM: 5.0,
                    TaskType.REPAIR_DAM: 5.0,
                    TaskType.LODGE_WORK: 2.0,
                    TaskType.GUARD: 2.0
                }
            )
        
        elif reason == "declining_wisdom":
            # Reset to defaults and increase exploration
            new_params = self._initialize_meta_params()
            new_params.rho = self.cfg.rho_max * 0.7
            new_params.beta_R = self.cfg.beta_R_min
            new_params.task_stimuli[TaskType.SCOUT] = 8.0
        
        else:
            # Generic reset
            new_params = self._initialize_meta_params()
        
        self.meta_params = new_params
        return new_params
    
    def get_overmind_state(self) -> Dict:
        """Get Overmind state for logging"""
        return {
            'rho': self.meta_params.rho,
            'beta_R': self.meta_params.beta_R,
            'gamma_dance': self.meta_params.gamma_dance,
            'wisdom_trend': self.get_wisdom_trend(),
            'task_stimuli': {
                task.name: value
                for task, value in self.meta_params.task_stimuli.items()
            }
        }
