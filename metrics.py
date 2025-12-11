"""
Metrics Module
Implements global reward function (§13) and wisdom signals (§14, §18)
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass

from config import RewardConfig, OvermindConfig, DamConfig
from environment import Environment
from agents import AgentPopulation
from physarum import PhysarumNetwork


@dataclass
class RewardComponents:
    """Breakdown of reward components"""
    survival: float
    stability: float
    habitat: float
    flood_penalty: float
    drought_penalty: float
    failure_penalty: float
    total: float


class MetricsCalculator:
    """
    Calculate global reward and wisdom signals
    
    Features:
    - Global reward function (§13)
    - Wisdom signal computation (§14)
    - ACP-informed wisdom (§18)
    - Brittleness detection (§18.1)
    """
    
    def __init__(
        self,
        reward_config: RewardConfig,
        overmind_config: OvermindConfig,
        dam_config: DamConfig
    ):
        self.reward_cfg = reward_config
        self.overmind_cfg = overmind_config
        self.dam_cfg = dam_config
        
        # Tracking for brittleness
        self.reward_history = []
        self.max_history_length = 100
    
    def compute_reward(
        self,
        environment: Environment,
        population: AgentPopulation
    ) -> RewardComponents:
        """
        Compute global reward (§13)
        
        R_t = R_survival + R_stability + R_habitat - C_flood - C_drought - C_failure
        """
        # Survival reward (§13.1)
        num_alive = population.get_num_alive()
        num_total = len(population.agents)
        R_survival = self.reward_cfg.alpha_1 * (num_alive / num_total)
        
        # Stability reward (§13.2)
        R_stability = self._compute_stability_reward(environment)
        
        # Habitat complexity reward (§13.3)
        R_habitat = self._compute_habitat_reward(environment)
        
        # Flood penalty (§13.4)
        C_flood = self._compute_flood_penalty(environment)
        
        # Drought penalty (§13.4)
        C_drought = self._compute_drought_penalty(environment)
        
        # Structural failure penalty (§13.4)
        C_failure = self._compute_failure_penalty(environment)
        
        # Total reward
        total = R_survival + R_stability + R_habitat - C_flood - C_drought - C_failure
        
        # Store in history for brittleness calculation
        self.reward_history.append(total)
        if len(self.reward_history) > self.max_history_length:
            self.reward_history.pop(0)
        
        return RewardComponents(
            survival=R_survival,
            stability=R_stability,
            habitat=R_habitat,
            flood_penalty=C_flood,
            drought_penalty=C_drought,
            failure_penalty=C_failure,
            total=total
        )
    
    def _compute_stability_reward(self, environment: Environment) -> float:
        """
        Compute hydrological stability reward (§13.2)
        
        R_stability = -α_2 * σ_{h,core}
        """
        core_metrics = environment.get_core_habitat_metrics()
        sigma_h = core_metrics['h_std_core']
        
        return -self.reward_cfg.alpha_2 * sigma_h
    
    def _compute_habitat_reward(self, environment: Environment) -> float:
        """
        Compute habitat complexity reward (§13.3)
        
        R_habitat = α_3 * (1/|I|) * Σ_i H(i, S_t)
        H(i) = exp(-λ_h * (h_i - h*)^2 - λ_v * (v_i - v*)^2)
        """
        h = environment.state.h
        v = environment.state.v
        
        h_star = self.reward_cfg.h_star
        v_star = self.reward_cfg.v_star
        lambda_h = self.reward_cfg.lambda_h_habitat
        lambda_v = self.reward_cfg.lambda_v_habitat
        
        # Habitat suitability function
        H = np.exp(
            -lambda_h * (h - h_star) ** 2 -
            lambda_v * (v - v_star) ** 2
        )
        
        # Average over all cells
        mean_H = np.mean(H)
        
        return self.reward_cfg.alpha_3 * mean_H
    
    def _compute_flood_penalty(self, environment: Environment) -> float:
        """
        Compute flood penalty (§13.4)
        
        C_flood = β_1 * Σ_{i ∈ I_down} 1[h_i > h_flood]
        """
        downstream_metrics = environment.get_downstream_metrics()
        num_flood = downstream_metrics['num_flood_cells']
        
        return self.reward_cfg.beta_1 * num_flood
    
    def _compute_drought_penalty(self, environment: Environment) -> float:
        """
        Compute drought penalty (§13.4)
        
        C_drought = β_2 * Σ_{i ∈ I_core} 1[h_i < h_drought]
        """
        core_indices = list(environment.core_habitat)
        h_core = environment.state.h[core_indices]
        
        num_drought = np.sum(h_core < environment.hydro_cfg.h_drought)
        
        return self.reward_cfg.beta_2 * num_drought
    
    def _compute_failure_penalty(self, environment: Environment) -> float:
        """
        Compute structural failure penalty (§13.4)
        
        C_failure = β_3 * |B_t|
        
        Where B_t is the set of dam cells showing breakage
        """
        # This requires tracking dam permeability changes
        # For now, approximate by checking for very low permeability that increased
        # (indicates erosion breakthrough)
        
        # Count cells with dam permeability between 0.3 and 0.7
        # (indicates partial dam that may be failing)
        d = environment.state.d
        num_failures = np.sum((d > 0.3) & (d < 0.7))
        
        return self.reward_cfg.beta_3 * num_failures
    
    def compute_wisdom_signal(
        self,
        environment: Environment,
        population: AgentPopulation,
        physarum_network: PhysarumNetwork,
        reward_components: RewardComponents
    ) -> float:
        """
        Compute wisdom signal with ACP (§14, §18)
        
        w_ACP = w + λ_{Hs} * H_struct - λ_B * B - λ_simp * degenerate - λ_mono * monotony
        """
        # Base wisdom signal (§14.2)
        core_metrics = environment.get_core_habitat_metrics()
        downstream_metrics = environment.get_downstream_metrics()
        
        sigma_h = core_metrics['h_std_core']
        C_flood = reward_components.flood_penalty
        C_drought = reward_components.drought_penalty
        C_failure = reward_components.failure_penalty
        R_habitat = reward_components.habitat
        
        w_base = (
            -self.overmind_cfg.lambda_sigma * sigma_h -
            self.overmind_cfg.lambda_F * C_flood -
            self.overmind_cfg.lambda_D * C_drought -
            self.overmind_cfg.lambda_B * C_failure +
            self.overmind_cfg.lambda_H * R_habitat
        )
        
        # ACP enhancements (§18.2)
        
        # Structural entropy (§18.1)
        H_struct = physarum_network.get_structural_entropy()
        
        # Brittleness (§18.1)
        B_brittle = self._compute_brittleness()
        
        # Degenerate network penalty (§18.1)
        is_degenerate = physarum_network.detect_degenerate_network()
        degenerate_penalty = 1.0 if is_degenerate else 0.0
        
        # Monotony penalty (low exploration diversity)
        # For now, set to 0 (would need project recruitment variance tracking)
        monotony_penalty = 0.0
        
        # Total ACP wisdom
        w_ACP = (
            w_base +
            self.overmind_cfg.lambda_Hs * H_struct -
            self.overmind_cfg.lambda_B_brittle * B_brittle -
            self.overmind_cfg.lambda_simp * degenerate_penalty -
            self.overmind_cfg.lambda_mono * monotony_penalty
        )
        
        return w_ACP
    
    def _compute_brittleness(self) -> float:
        """
        Compute brittleness indicator (§18.1)
        
        Measures sensitivity to perturbations
        For now, use variance in recent rewards as proxy
        """
        if len(self.reward_history) < 10:
            return 0.0
        
        # Use recent reward variance as brittleness proxy
        recent_rewards = self.reward_history[-10:]
        variance = np.var(recent_rewards)
        
        # Normalize
        brittleness = np.sqrt(variance)
        
        return brittleness
    
    def compute_all_metrics(
        self,
        environment: Environment,
        population: AgentPopulation,
        physarum_network: PhysarumNetwork
    ) -> Dict[str, float]:
        """Compute all metrics for logging"""
        # Reward components
        reward = self.compute_reward(environment, population)
        
        # Wisdom signal
        wisdom = self.compute_wisdom_signal(
            environment, population, physarum_network, reward
        )
        
        # Environment metrics
        core_metrics = environment.get_core_habitat_metrics()
        downstream_metrics = environment.get_downstream_metrics()
        
        # Population metrics
        pop_metrics = population.get_population_metrics()
        
        # Physarum metrics
        H_struct = physarum_network.get_structural_entropy()
        is_degenerate = physarum_network.detect_degenerate_network()
        
        # Combine all
        all_metrics = {
            # Reward
            'reward_survival': reward.survival,
            'reward_stability': reward.stability,
            'reward_habitat': reward.habitat,
            'reward_flood_penalty': reward.flood_penalty,
            'reward_drought_penalty': reward.drought_penalty,
            'reward_failure_penalty': reward.failure_penalty,
            'reward_total': reward.total,
            
            # Wisdom
            'wisdom_signal': wisdom,
            
            # Environment
            'h_mean_core': core_metrics['h_mean_core'],
            'h_std_core': core_metrics['h_std_core'],
            'num_flood_cells': downstream_metrics['num_flood_cells'],
            
            # Population
            'num_alive': pop_metrics['num_alive'],
            'mean_energy': pop_metrics['mean_energy'],
            'mean_satiety': pop_metrics['mean_satiety'],
            'mean_wetness': pop_metrics['mean_wetness'],
            
            # Physarum
            'structural_entropy': H_struct,
            'network_degenerate': 1.0 if is_degenerate else 0.0,
            
            # Brittleness
            'brittleness': self._compute_brittleness()
        }
        
        return all_metrics


class WisdomNormalizer:
    """
    Wisdom signal normalizer (§3.6)
    
    Normalizes raw wisdom values to z-scores in [-3, 3] range
    using sliding window statistics.
    
    This makes wisdom values interpretable and prevents
    Overmind from being confused by changing scales.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.wisdom_history: List[float] = []
    
    def normalize(self, raw_wisdom: float) -> float:
        """
        Normalize wisdom to z-score
        
        z = (w - μ) / σ
        
        Clipped to [-3, 3] range for stability
        """
        # Add to history
        self.wisdom_history.append(raw_wisdom)
        if len(self.wisdom_history) > self.window_size:
            self.wisdom_history.pop(0)
        
        # Need at least 2 values for std
        if len(self.wisdom_history) < 2:
            return 0.0
        
        # Compute statistics
        mean = np.mean(self.wisdom_history)
        std = np.std(self.wisdom_history)
        
        # Avoid division by zero
        if std < 1e-8:
            return 0.0
        
        # Z-score
        z = (raw_wisdom - mean) / std
        
        # Clip to [-3, 3]
        z_clipped = np.clip(z, -3.0, 3.0)
        
        return z_clipped
    
    def get_stats(self) -> Dict[str, float]:
        """Get current normalization statistics"""
        if len(self.wisdom_history) < 2:
            return {'mean': 0.0, 'std': 1.0, 'n': 0}
        
        return {
            'mean': np.mean(self.wisdom_history),
            'std': np.std(self.wisdom_history),
            'n': len(self.wisdom_history)
        }
