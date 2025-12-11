"""
Overmind Module - IMPROVED VERSION
Implements Contemplative Overmind with Architect Cognitive Prior (§14, §18)

Enhanced with mathematical specification improvements:
- Exponential smoothing of wisdom (§1.7)
- Bounded parameter updates (§3.1)
- Two-mode operation: EXPLORE vs CONSOLIDATE (§4)
- Meta-self-critique module
- Edge-of-chaos preference
"""

import numpy as np
import logging
from typing import Dict, Tuple, List
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
    Contemplative Overmind with Architect Cognitive Prior - IMPROVED
    
    Key improvements from mathematical specification:
    1. Exponential smoothing prevents reactive jumping
    2. Bounded updates prevent parameter spikes
    3. Two-mode operation (EXPLORE vs CONSOLIDATE)
    4. Strategic behavior instead of reactive
    5. Edge-of-chaos preference (entropy targeting)
    """
    
    def __init__(self, config: OvermindConfig):
        self.cfg = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize meta-parameters
        self.meta_params = self._initialize_meta_params()
        
        # Wisdom tracking with exponential smoothing (§1.7)
        self.wisdom_history: List[float] = []
        self.max_history_length = 100
        self.wisdom_smoothed: float = 0.0
        self.prev_wisdom_smoothed: float = 0.0
        self.smoothing_alpha: float = 0.05  # Small for stability
        
        # Mode tracking (§4 - Architect Cognitive Prior)
        self.mode: str = "EXPLORE"  # Start exploring
        self.mode_counter: int = 0
        self.steps_since_improvement: int = 0
        
        # Step counter for periodic checks
        self.step_count: int = 0
        
        # Entropy target band (edge-of-chaos preference)
        self.H_min_target: float = 8.0
        self.H_max_target: float = 10.0
    
    def _initialize_meta_params(self) -> MetaParameters:
        """Initialize meta-parameters at reasonable defaults"""
        rho = (self.cfg.rho_min + self.cfg.rho_max) / 2
        beta_R = (self.cfg.beta_R_min + self.cfg.beta_R_max) / 2
        gamma_dance = (self.cfg.gamma_dance_min + self.cfg.gamma_dance_max) / 2
        
        # Initial task stimuli
        from config import TaskType
        task_stimuli = {task: 5.0 for task in TaskType}
        task_stimuli[TaskType.BUILD_DAM] = 7.0
        task_stimuli[TaskType.REPAIR_DAM] = 6.0
        
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
        Update meta-parameters with smoothing and strategic behavior
        
        NEW APPROACH (from math spec):
        1. Smooth wisdom signal exponentially
        2. Compute wisdom drift (delta_w)
        3. Determine operating mode (EXPLORE/CONSOLIDATE)
        4. Get mode-specific targets
        5. Smoothly update toward targets
        """
        self.step_count += 1
        
        # Store wisdom
        self.wisdom_history.append(wisdom_signal)
        if len(self.wisdom_history) > self.max_history_length:
            self.wisdom_history.pop(0)
        
        # 1. Exponential smoothing (§1.7)
        self._update_smoothed_wisdom(wisdom_signal)
        
        # 2. Compute drift
        delta_w = self._compute_wisdom_drift()
        
        # 3. Extract metrics
        brittleness = self._compute_brittleness(metrics)
        H_struct = metrics.get('structural_entropy', 8.5)
        
        # 4. Determine mode (EXPLORE vs CONSOLIDATE)
        self._determine_mode(delta_w, brittleness, H_struct)
        
        # 5. Get mode-specific targets
        targets = self._get_mode_targets()
        
        # 6. Smoothly update toward targets (bounded updates)
        new_rho = self._smooth_param_update(
            self.meta_params.rho,
            targets['rho'],
            self.cfg.rho_min,
            self.cfg.rho_max,
            step_size=0.01,
            momentum=0.9
        )
        
        new_beta_R = self._smooth_param_update(
            self.meta_params.beta_R,
            targets['beta_R'],
            self.cfg.beta_R_min,
            self.cfg.beta_R_max,
            step_size=0.05,
            momentum=0.9
        )
        
        new_gamma_dance = self._smooth_param_update(
            self.meta_params.gamma_dance,
            targets['gamma_dance'],
            self.cfg.gamma_dance_min,
            self.cfg.gamma_dance_max,
            step_size=0.05,
            momentum=0.9
        )
        
        # 7. Update task stimuli toward mode targets
        new_stimuli = self._adapt_task_stimuli_smooth(
            metrics,
            targets['stimuli']
        )
        
        # 8. Periodic counterfactual check
        if self._run_counterfactual_check():
            # Force back to EXPLORE if current regime is brittle
            if self.mode == "CONSOLIDATE":
                self.mode = "EXPLORE"
                self.mode_counter = 0
                print("Overmind: Counterfactual check triggered EXPLORE mode")
        
        # Create new meta-parameters
        new_params = MetaParameters(
            rho=new_rho,
            beta_R=new_beta_R,
            gamma_dance=new_gamma_dance,
            task_stimuli=new_stimuli
        )
        
        self.meta_params = new_params
        return new_params
    
    def _update_smoothed_wisdom(self, raw_wisdom: float):
        """
        Exponential smoothing of wisdom (§1.7)
        w̄_t = (1-α)w̄_{t-1} + αw_t
        """
        self.wisdom_smoothed = (
            (1 - self.smoothing_alpha) * self.wisdom_smoothed +
            self.smoothing_alpha * raw_wisdom
        )
    
    def _compute_wisdom_drift(self) -> float:
        """
        Compute wisdom drift: Δw_t = w̄_t - w̄_{t-1}
        """
        delta_w = self.wisdom_smoothed - self.prev_wisdom_smoothed
        self.prev_wisdom_smoothed = self.wisdom_smoothed
        return delta_w
    
    def _smooth_param_update(
        self,
        current: float,
        target: float,
        min_val: float,
        max_val: float,
        step_size: float = 0.05,
        momentum: float = 0.9
    ) -> float:
        """
        Smoothly update parameter toward target (§3.1)
        
        Prevents wild oscillations by:
        1. Limiting step size
        2. Applying momentum (weighted average)
        3. Clipping to bounds
        """
        # Compute proposed step
        delta = np.sign(target - current) * step_size
        proposed = current + delta
        
        # Clip to bounds
        clipped = np.clip(proposed, min_val, max_val)
        
        # Apply momentum (90% current, 10% new)
        smoothed = momentum * current + (1 - momentum) * clipped
        
        return smoothed
    
    def _determine_mode(
        self,
        delta_w: float,
        brittleness: float,
        H_struct: float
    ):
        """
        Determine operating mode: EXPLORE vs CONSOLIDATE (§4)
        
        EXPLORE: When stuck, brittle, or boring
        CONSOLIDATE: When improving and robust
        """
        # Track improvement
        if delta_w < 0:
            self.steps_since_improvement += 1
        else:
            self.steps_since_improvement = 0
        
        # Triggers for EXPLORE
        explore_conditions = [
            self.steps_since_improvement > 50,  # Stuck
            brittleness > 0.3,                  # Brittle
            H_struct < self.H_min_target,       # Too simple
            H_struct > self.H_max_target        # Too chaotic
        ]
        
        # Triggers for CONSOLIDATE
        consolidate_conditions = [
            delta_w > 0,                        # Improving
            brittleness < 0.1,                  # Robust
            self.H_min_target <= H_struct <= self.H_max_target  # Sweet spot
        ]
        
        # Mode switching
        if any(explore_conditions) and self.mode == "CONSOLIDATE":
            self.mode = "EXPLORE"
            self.mode_counter = 0
            self.steps_since_improvement = 0
            print(f"[Overmind] Switching to EXPLORE mode (step {self.step_count})")
        
        elif all(consolidate_conditions) and self.mode == "EXPLORE":
            self.mode = "CONSOLIDATE"
            self.mode_counter = 0
            print(f"[Overmind] Switching to CONSOLIDATE mode (step {self.step_count})")
        
        self.mode_counter += 1
    
    def _get_mode_targets(self) -> Dict:
        """
        Get target parameter values for current mode (§4)
        
        EXPLORE: Higher exploration, flatter recruitment
        CONSOLIDATE: Lower exploration, sharper recruitment
        """
        from config import TaskType
        
        if self.mode == "EXPLORE":
            return {
                'rho': 0.15,      # Higher evaporation (clear stale trails)
                'beta_R': 1.5,    # Flatter recruitment (diversity)
                'gamma_dance': 1.5,
                'stimuli': {
                    TaskType.SCOUT: 8.0,
                    TaskType.REPAIR_DAM: 7.0,
                    TaskType.BUILD_DAM: 6.0,
                    TaskType.FORAGE: 5.0,
                    TaskType.LODGE_WORK: 4.0,
                    TaskType.GUARD: 3.0
                }
            }
        else:  # CONSOLIDATE
            return {
                'rho': 0.05,      # Lower evaporation (preserve trails)
                'beta_R': 3.5,    # Sharper recruitment (focus)
                'gamma_dance': 0.8,
                'stimuli': {
                    TaskType.REPAIR_DAM: 8.0,
                    TaskType.LODGE_WORK: 7.0,
                    TaskType.GUARD: 6.5,
                    TaskType.BUILD_DAM: 5.0,
                    TaskType.FORAGE: 5.0,
                    TaskType.SCOUT: 4.0
                }
            }
    
    def _adapt_task_stimuli_smooth(
        self,
        metrics: Dict[str, float],
        mode_targets: Dict
    ) -> Dict:
        """
        Smoothly adapt task stimuli toward mode targets
        Also responds to immediate needs (flooding, hunger)
        """
        from config import TaskType
        
        # Extract metrics
        h_std_core = metrics.get('h_std_core', 0.0)
        h_mean_core = metrics.get('h_mean_core', 0.0)
        num_flood = metrics.get('num_flood_cells', 0)
        mean_satiety = metrics.get('mean_satiety', 0.5)
        
        # Start with mode targets
        new_stimuli = mode_targets.copy()
        
        # Override with urgent needs
        
        # Foraging stimulus (hunger overrides mode)
        if mean_satiety < 0.3:
            new_stimuli[TaskType.FORAGE] = min(10.0, new_stimuli[TaskType.FORAGE] + 2.0)
        
        # Dam building stimulus (flooding overrides mode)
        dam_urgency = 0.0
        if h_std_core > 0.3:
            dam_urgency += 0.5
        if h_mean_core > 0.6:
            dam_urgency += 0.5
        if num_flood > 5:
            dam_urgency += 1.0
        
        if dam_urgency > 0:
            new_stimuli[TaskType.BUILD_DAM] = min(
                10.0,
                new_stimuli[TaskType.BUILD_DAM] + dam_urgency
            )
        
        # Smooth transition from current to target
        smoothed_stimuli = {}
        for task in TaskType:
            current = self.meta_params.task_stimuli.get(task, 5.0)
            target = new_stimuli.get(task, 5.0)
            # 80% current, 20% target
            smoothed = 0.8 * current + 0.2 * target
            smoothed_stimuli[task] = np.clip(smoothed, 1.0, 10.0)
        
        return smoothed_stimuli
    
    def _run_counterfactual_check(self) -> bool:
        """
        Meta-self-critique: Check if perturbations would be better (§4)
        
        Returns True if current regime seems brittle
        (many perturbations look better)
        """
        if self.step_count % 100 != 0:  # Every 100 steps
            return False
        
        # Get current targets
        targets = self._get_mode_targets()
        
        # Perturb and check
        num_samples = 5
        better_count = 0
        
        for _ in range(num_samples):
            # Random perturbation
            perturbed_rho = self.meta_params.rho * (1 + np.random.normal(0, 0.1))
            perturbed_beta = self.meta_params.beta_R * (1 + np.random.normal(0, 0.1))
            
            # Check if perturbation is closer to target
            rho_improvement = (
                abs(perturbed_rho - targets['rho']) <
                abs(self.meta_params.rho - targets['rho'])
            )
            beta_improvement = (
                abs(perturbed_beta - targets['beta_R']) <
                abs(self.meta_params.beta_R - targets['beta_R'])
            )
            
            if rho_improvement or beta_improvement:
                better_count += 1
        
        # If >60% perturbations are better, regime is brittle
        is_brittle = (better_count / num_samples) > 0.6
        
        if is_brittle:
            print(f"[Overmind] Counterfactual check: Regime is brittle ({better_count}/{num_samples} perturbations better)")
        
        return is_brittle
    
    def _compute_brittleness(self, metrics: Dict[str, float]) -> float:
        """
        Compute brittleness from metrics
        
        Brittleness indicators:
        - High variance in reward
        - Many dam failures
        - Unstable water levels
        """
        # Extract indicators
        h_std = metrics.get('h_std_core', 0.0)
        num_failures = metrics.get('num_failure_cells', 0)
        
        # Normalize to [0, 1]
        h_brittleness = min(1.0, h_std / 2.0)  # High std = brittle
        failure_brittleness = min(1.0, num_failures / 10.0)  # Many failures = brittle
        
        # Combine
        brittleness = 0.5 * h_brittleness + 0.5 * failure_brittleness
        
        return brittleness
    
    def compute_structural_entropy(self, physarum_network) -> float:
        """
        Compute structural entropy from Physarum network (§18.1)
        
        H_struct = -Σ p_ij log(p_ij)
        where p_ij = D_ij / Σ D_uv
        """
        try:
            from config import CommodityType
            
            # Collect all conductivities for water commodity
            conductivities = []
            for edge in physarum_network.edges:
                D_water = physarum_network.state.D[edge][CommodityType.WATER]
                if D_water > 1e-8:
                    conductivities.append(D_water)
            
            if len(conductivities) < 2:
                return 0.0
            
            # Normalize to probabilities
            total = sum(conductivities)
            if total < 1e-8:
                return 0.0
            
            probs = [D / total for D in conductivities]
            
            # Compute entropy
            entropy = -sum(p * np.log(p + 1e-10) for p in probs)
            
            return entropy
            
        except Exception as e:
            # If error, return neutral value
            return 8.5
    
    def detect_degenerate_network(self, physarum_network) -> bool:
        """
        Detect if network has collapsed to single path (§18.2)
        """
        try:
            from config import CommodityType
            
            conductivities = []
            for edge in physarum_network.edges:
                D = physarum_network.state.D[edge][CommodityType.WATER]
                conductivities.append(D)
            
            if len(conductivities) < 2:
                return True
            
            max_D = max(conductivities)
            avg_D = np.mean(conductivities)
            
            # Degenerate if max >> average (single dominant path)
            is_degenerate = max_D > 10 * avg_D
            
            return is_degenerate
            
        except Exception as e:
            return False
    
    def get_state_dict(self) -> Dict:
        """Get current state for logging"""
        return {
            'mode': self.mode,
            'mode_counter': self.mode_counter,
            'rho': self.meta_params.rho,
            'beta_R': self.meta_params.beta_R,
            'gamma_dance': self.meta_params.gamma_dance,
            'wisdom_smoothed': self.wisdom_smoothed,
            'steps_since_improvement': self.steps_since_improvement,
            'task_stimuli': {
                str(task): value
                for task, value in self.meta_params.task_stimuli.items()
            }
        }
    
    def get_wisdom_trend(self) -> float:
        """Get wisdom trend from recent history"""
        if len(self.wisdom_history) < 10:
            return 0.0
        
        recent = self.wisdom_history[-10:]
        return np.mean(recent)
    
    def should_intervene(self, metrics: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if Overmind should intervene
        
        Intervention triggers:
        - Catastrophic failure (many dead agents)
        - Extreme brittleness
        - Stuck in bad regime
        """
        num_alive = metrics.get('num_alive', 30)
        brittleness = metrics.get('brittleness', 0.0)
        
        # Catastrophic failure
        if num_alive < 10:
            return True, "catastrophic_failure"
        
        # Extreme brittleness
        if brittleness > 0.5:
            return True, "extreme_brittleness"
        
        # Stuck in bad regime
        if self.steps_since_improvement > 100:
            return True, "stuck_regime"
        
        return False, ""
    
    def execute_intervention(self, reason: str) -> MetaParameters:
        """
        Execute intervention by forcing exploration
        """
        self.mode = "EXPLORE"
        self.mode_counter = 0
        self.steps_since_improvement = 0
        
        self.logger.warning(f"Overmind intervention: {reason}")
        
        return self.meta_params
    
    def get_overmind_state(self) -> Dict:
        """Get current overmind state (alias for get_state_dict)"""
        return self.get_state_dict()
