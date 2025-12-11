"""
Beaver Ecosystem Configuration
Complete parameter specification following mathematical formulation (§1-19)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum, auto
import numpy as np


class TaskType(Enum):
    """Task types for division of labor (§9)"""
    FORAGE = auto()
    BUILD_DAM = auto()
    REPAIR_DAM = auto()
    LODGE_WORK = auto()
    GUARD = auto()
    SCOUT = auto()


class ActionType(Enum):
    """Local action types (§12)"""
    STAY = auto()
    MOVE_UP = auto()
    MOVE_DOWN = auto()
    MOVE_LEFT = auto()
    MOVE_RIGHT = auto()
    FELL_TREE = auto()
    BUILD_DAM = auto()
    PATCH_DAM = auto()
    HARVEST_MUD = auto()
    FORAGE = auto()


class AgentRole(Enum):
    """Agent roles (§8)"""
    SCOUT = auto()
    WORKER = auto()
    GUARDIAN = auto()


class CommodityType(Enum):
    """Multi-commodity Physarum types (§17.5)"""
    WATER = auto()
    LOG_TRANSPORT = auto()
    FOOD_TRANSPORT = auto()


@dataclass
class WorldConfig:
    """World configuration (§1)"""
    # Grid dimensions
    grid_height: int = 50
    grid_width: int = 50
    
    # Time discretization
    dt: float = 0.1
    max_steps: int = 5000
    
    # Random seed
    random_seed: int = 42
    
    # Elevation field (will be generated)
    elevation_scale: float = 5.0
    elevation_roughness: float = 0.3


@dataclass
class HydrologyConfig:
    """Hydrology dynamics parameters (§3)"""
    # Base conductance - INCREASED for faster drainage
    g0: float = 2.0  # Was 0.5, now 4x higher
    
    # Loss rates - DRAMATICALLY INCREASED to balance water budget
    alpha_evap: float = 0.025  # Second fix: Reduced from 0.04 (2.5% loss per step)
    alpha_seep: float = 0.015  # Second fix: Reduced from 0.03 (1.5% loss per step)
    
    # Rainfall parameters - REDUCED to prevent flooding
    mean_rainfall: float = 0.04  # Fourth adjustment: Increased from 0.03 to ensure h>0
    rainfall_std: float = 0.01   # Slightly increased for variability
    
    # Boundary inflow (for river source cells) - INCREASED AGAIN
    boundary_inflow: float = 0.40  # Fourth adjustment: Increased from 0.30
    
    # Thresholds
    h_wet: float = 1.0  # Second fix: Lowered from 1.5
    h_flood: float = 2.5  # Second fix: Lowered from 3.0
    h_drought: float = 0.15  # Second fix: Lowered from 0.3


@dataclass
class VegetationConfig:
    """Vegetation and soil moisture parameters (§4)"""
    # Growth parameters
    gamma_v: float = 0.1  # Growth rate
    V_max: float = 10.0  # Maximum biomass
    
    # Soil moisture parameters
    beta_m: float = 0.05  # Moisture decay
    beta_infil: float = 0.1  # Infiltration rate
    beta_drain: float = 0.02  # Drainage rate
    
    # Initial conditions
    initial_vegetation_mean: float = 5.0
    initial_vegetation_std: float = 2.0
    initial_moisture: float = 0.5


@dataclass
class DamConfig:
    """Dam and structural memory parameters (§5)"""
    # Dam dynamics
    eta_build: float = 0.05  # Building effectiveness
    eta_erode: float = 0.01  # Erosion rate
    
    # Erosion function
    phi_max: float = 0.5  # Maximum erosion pressure
    
    # Dam permeability
    rho_build: float = 0.1  # Building effort per action
    d_break_thr: float = 0.7  # Threshold for dam breaking
    delta_break: float = 0.3  # Change indicating breakage


@dataclass
class PheromoneConfig:
    """
    Multi-channel ant-style pheromone parameters (§6 + Enhancement #3)
    
    Supports specialized pheromone channels for different behaviors
    """
    # Base evaporation rate (used if channel-specific not provided)
    rho: float = 0.05
    
    # Deposition parameters
    delta: float = 1.0  # Base deposition amount (was delta_0)
    delta_0: float = 1.0  # Backward compat alias
    
    # Movement probability parameters
    alpha: float = 1.0  # Pheromone influence exponent
    beta: float = 2.0  # Heuristic influence exponent
    
    # MULTI-CHANNEL: Channel-specific evaporation rates
    # Different channels persist for different durations
    channel_evaporation: Dict = field(default_factory=lambda: {
        'FOOD': 0.1,          # Fast evaporation (food sources change)
        'CONSTRUCTION': 0.05,  # Slow (construction sites persist)
        'DANGER': 0.15,        # Very fast (danger is transient)
        'RETURN_HOME': 0.08,   # Medium (stable path home)
        'EXPLORATION': 0.12    # Fast (explore new areas)
    })
    
    # Success-based deposition bonuses
    food_success_bonus: float = 2.0    # Extra deposit when carrying food
    build_success_bonus: float = 1.5   # Extra deposit when building
    danger_intensity: float = 3.0      # Strong warning signal
    
    # Directional bias (boost trails aligned with target direction)
    use_directional_bias: bool = True
    directional_boost: float = 1.5  # Multiplier when aligned with target


@dataclass
class ProjectConfig:
    """Bee-style recruitment parameters (§7)"""
    # Number of projects
    num_projects: int = 5
    
    # Recruitment dynamics
    kappa: float = 0.1  # Recruitment decay
    gamma_dance: float = 0.5  # Recruitment gain
    lambda_Q: float = 1.0  # Quality exponent
    
    # Project quality weights
    w1: float = 1.0  # ResourceAbundance weight
    w2: float = 0.8  # HydroImpact weight
    w3: float = 0.5  # Safety weight
    w4: float = 0.3  # DistanceCost weight
    
    # Selection sharpness
    beta_R: float = 2.0  # Initial recruitment sharpness
    
    # Cross-inhibition
    chi: float = 0.1  # Cross-inhibition strength


@dataclass
class AgentConfig:
    """Agent (beaver) parameters (§8, §12)"""
    # Population
    num_agents: int = 30
    
    # Energy and satiety
    initial_energy: float = 100.0
    initial_satiety: float = 0.8
    
    # Energy costs
    c_move_base: float = 0.5
    c_work_base: float = 1.0
    
    # Food parameters
    eta_food: float = 5.0
    beta_s: float = 0.2
    alpha_s: float = 0.01
    
    # Wetness parameters
    alpha_w: float = 0.1
    beta_w: float = 0.2
    gamma_w: float = 0.3
    
    # Tree felling and dam building
    rho_tree: float = 0.5
    
    # Task response parameters
    n: float = 2.0  # Response steepness
    
    # Initial threshold distribution
    theta_mean: float = 5.0
    theta_std: float = 2.0
    
    # Mortality thresholds
    energy_death_threshold: float = 0.0
    satiety_death_threshold: float = -5.0


@dataclass
class PhysarumConfig:
    """Physarum network parameters (§17)"""
    # Conductivity dynamics
    alpha_D: float = 0.5  # Reinforcement rate
    beta_D: float = 0.1  # Decay rate
    gamma_flux: float = 1.0  # Flux exponent
    
    # Source/sink parameters
    I_source: float = 10.0
    I_sink: float = 10.0
    
    # Edge cost parameters
    lambda_h: float = 0.5  # Water depth cost multiplier
    lambda_z: float = 1.0  # Elevation cost multiplier
    
    # Multi-commodity parameters
    enable_multicommodity: bool = True
    
    # Solver parameters
    max_solver_iterations: int = 500
    solver_tolerance: float = 1e-3  # Relaxed for better convergence


@dataclass
class OvermindConfig:
    """Contemplative Overmind parameters (§14, §18)"""
    # Wisdom signal weights
    lambda_sigma: float = 1.0
    lambda_F: float = 2.0
    lambda_D: float = 2.0
    lambda_B: float = 1.5
    lambda_H: float = 1.0
    
    # ACP (Architect Cognitive Prior) weights
    lambda_Hs: float = 1.0  # Structural entropy reward
    lambda_B_brittle: float = 2.0  # Brittleness penalty
    lambda_simp: float = 1.5  # Simplicity penalty
    lambda_mono: float = 1.0  # Monotony penalty
    
    # Discount factor
    Gamma: float = 0.99
    
    # Meta-parameter bounds
    rho_min: float = 0.01
    rho_max: float = 0.2
    beta_R_min: float = 0.5
    beta_R_max: float = 5.0
    gamma_dance_min: float = 0.1
    gamma_dance_max: float = 2.0
    
    # Brittleness testing
    perturbation_std: float = 0.1
    num_perturbation_samples: int = 5
    
    # Structural entropy parameters
    epsilon: float = 1e-8  # Numerical stability


@dataclass
class RewardConfig:
    """Global reward function parameters (§13)"""
    # Survival weight
    alpha_1: float = 10.0
    
    # Stability weight
    alpha_2: float = 5.0
    
    # Habitat complexity weight
    alpha_3: float = 5.0  # Second fix: Increased from 3.0 (boost habitat reward)
    
    # Penalty weights
    beta_1: float = 5.0  # Second fix: Reduced from 10.0 (flood penalty)
    beta_2: float = 4.0  # Second fix: Reduced from 8.0 (drought penalty)
    beta_3: float = 8.0  # Second fix: Reduced from 15.0 (failure penalty)
    
    # Habitat suitability targets
    h_star: float = 0.7  # Second fix: Lowered from 1.5 (target water depth)
    v_star: float = 5.0
    lambda_h_habitat: float = 1.0
    lambda_v_habitat: float = 0.5
    
    # Discount factor
    gamma: float = 0.99


@dataclass
class PolicyConfig:
    """Policy parameters (§15)"""
    # Greedy policy weights
    lambda_E: float = 1.0
    lambda_S: float = 2.0
    lambda_safe: float = 3.0
    lambda_effort: float = 0.5
    
    # Contemplative policy parameters
    horizon: int = 10
    lambda_W: float = 1.0  # Wisdom weight
    beta_Q: float = 1.0  # Action selection temperature


@dataclass
class VisualizationConfig:
    """Visualization parameters"""
    # Display settings
    cell_size: int = 10
    update_interval: int = 10
    
    # Color schemes
    water_cmap: str = "Blues"
    vegetation_cmap: str = "Greens"
    pheromone_cmap: str = "Reds"
    
    # Save settings
    save_interval: int = 50
    dpi: int = 150


# ============================================================================
# ENHANCEMENT CONFIGURATION SECTIONS
# ============================================================================

@dataclass
class AgentMemoryConfig:
    """Agent memory system configuration (Enhancement #1)"""
    hidden_dim: int = 50
    use_gru_memory: bool = True
    buffer_size: int = 20
    use_experience_replay: bool = True
    track_visited_cells: bool = True
    max_visited_history: int = 100
    track_heuristic_performance: bool = True


@dataclass
class RoleSpecializationConfig:
    """Role-specific network configuration (Enhancement #6)"""
    use_role_specific_networks: bool = True
    policy_hidden_dim: int = 128
    gru_hidden_dim: int = 50
    scout_exploration_bias: float = 0.5
    worker_construction_bias: float = 0.5
    guardian_stay_bias: float = 0.5
    frustration_threshold: float = 10.0
    frustration_decay_rate: float = 1.0
    frustration_growth_rate: float = 2.0
    allow_role_switching: bool = True
    min_steps_before_switch: int = 100


@dataclass
class PhysarumAgentCouplingConfig:
    """Physarum-agent integration configuration (Enhancement #2)"""
    enable_physarum_guidance: bool = True
    pheromone_weight: float = 0.3
    physarum_weight: float = 0.4
    heuristic_weight: float = 0.3
    use_physarum_pathfinding: bool = True
    max_path_search_steps: int = 100
    track_physarum_usage: bool = True


@dataclass
class LocalPredictionConfig:
    """Local predictive modeling configuration (Enhancement #5)"""
    use_local_prediction: bool = True
    local_obs_dim: int = 25
    hidden_dim: int = 64
    lstm_hidden_dim: int = 32
    prediction_loss_weight: float = 0.1
    train_predictor: bool = True
    enable_terrain_rewards: bool = True
    flood_prevention_reward: float = -10.0
    drought_prevention_reward: float = -5.0
    vegetation_reward: float = 2.0
    stability_reward: float = -3.0
    infrastructure_reward: float = 15.0


@dataclass
class PopulationDynamicsConfig:
    """Population dynamics configuration (Enhancement #4 + #9)"""
    enable_population_dynamics: bool = False  # Disabled by default (experimental)
    max_population: int = 50
    min_population: int = 10
    initial_population: int = 30
    enable_reproduction: bool = True
    reproduction_threshold_population: int = 15
    reproduction_energy_cost: float = 20.0
    check_mortality: bool = True
    starvation_threshold: float = 0.0
    drowning_depth: float = 3.0
    drowning_wetness: float = 0.8
    enable_evolution: bool = True
    mutation_rate: float = 0.05
    crossover_rate: float = 0.7
    tournament_size: int = 5
    fitness_survival_weight: float = 1.0
    fitness_dams_weight: float = 10.0
    fitness_food_weight: float = 5.0
    fitness_reward_weight: float = 0.1


@dataclass
class EnhancedAgentConfigExtensions:
    """Agent enhancement configurations"""
    memory: AgentMemoryConfig = field(default_factory=AgentMemoryConfig)
    roles: RoleSpecializationConfig = field(default_factory=RoleSpecializationConfig)
    physarum_coupling: PhysarumAgentCouplingConfig = field(default_factory=PhysarumAgentCouplingConfig)
    prediction: LocalPredictionConfig = field(default_factory=LocalPredictionConfig)
    population: PopulationDynamicsConfig = field(default_factory=PopulationDynamicsConfig)


@dataclass
class SimulationConfig:
    """Master configuration"""
    world: WorldConfig = field(default_factory=WorldConfig)
    hydrology: HydrologyConfig = field(default_factory=HydrologyConfig)
    vegetation: VegetationConfig = field(default_factory=VegetationConfig)
    dam: DamConfig = field(default_factory=DamConfig)
    pheromone: PheromoneConfig = field(default_factory=PheromoneConfig)
    project: ProjectConfig = field(default_factory=ProjectConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    physarum: PhysarumConfig = field(default_factory=PhysarumConfig)
    overmind: OvermindConfig = field(default_factory=OvermindConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # ENHANCEMENTS (new)
    agent_enhancements: EnhancedAgentConfigExtensions = field(default_factory=EnhancedAgentConfigExtensions)
    
    # Experiment settings
    experiment_name: str = "beaver_ecosystem"
    save_directory: str = "./output"
    enable_visualization: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    
    # Policy mode
    use_contemplative_policy: bool = True


def create_default_config() -> SimulationConfig:
    """Create default configuration"""
    return SimulationConfig()


def create_greedy_config() -> SimulationConfig:
    """Create configuration for greedy baseline"""
    config = SimulationConfig()
    config.use_contemplative_policy = False
    config.experiment_name = "beaver_ecosystem_greedy"
    return config


def create_ablation_config(
    disable_physarum: bool = False,
    disable_overmind: bool = False,
    disable_acp: bool = False
) -> SimulationConfig:
    """Create ablation study configuration"""
    config = SimulationConfig()
    
    if disable_physarum:
        config.physarum.alpha_D = 0.0
        config.experiment_name += "_no_physarum"
    
    if disable_overmind:
        config.overmind.lambda_sigma = 0.0
        config.overmind.lambda_F = 0.0
        config.overmind.lambda_D = 0.0
        config.overmind.lambda_B = 0.0
        config.overmind.lambda_H = 0.0
        config.experiment_name += "_no_overmind"
    
    if disable_acp:
        config.overmind.lambda_Hs = 0.0
        config.overmind.lambda_B_brittle = 0.0
        config.overmind.lambda_simp = 0.0
        config.overmind.lambda_mono = 0.0
        config.experiment_name += "_no_acp"
    
    return config
