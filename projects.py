"""
Projects Module
Implements bee-style recruitment and project selection (§7, §10)
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass

from config import ProjectConfig
from environment import Environment


@dataclass
class Project:
    """Project definition (§7)"""
    id: int
    region: Set[int]  # Region of interest I_p
    quality: float  # Q_p^t
    recruitment: float  # R_p^t
    center: Tuple[int, int]  # Central location (row, col)


@dataclass
class ProjectState:
    """State of all projects"""
    projects: List[Project]
    num_scouts_per_project: Dict[int, int]  # Track scouts advertising each project


class ProjectManager:
    """
    Bee-style recruitment and project selection (§7, §10)
    
    Features:
    - Project quality estimation (§7)
    - Recruitment dynamics with decay and gain (§7)
    - Cross-inhibition for consensus (§7)
    - Waggle dance softmax selection (§10)
    """
    
    def __init__(
        self,
        environment: Environment,
        config: ProjectConfig
    ):
        self.env = environment
        self.cfg = config
        
        # Initialize projects
        self.state = self._initialize_projects()
    
    def _initialize_projects(self) -> ProjectState:
        """Initialize project locations"""
        projects = []
        
        # Distribute projects across grid
        H, W = self.env.H, self.env.W
        
        for p_id in range(self.cfg.num_projects):
            # Random center location
            row = np.random.randint(0, H)
            col = np.random.randint(0, W)
            
            # Define region (5x5 area around center)
            region = self._define_region(row, col, radius=2)
            
            # Initial quality and recruitment
            quality = 0.0
            recruitment = 0.0
            
            project = Project(
                id=p_id,
                region=region,
                quality=quality,
                recruitment=recruitment,
                center=(row, col)
            )
            projects.append(project)
        
        num_scouts_per_project = {p.id: 0 for p in projects}
        
        return ProjectState(
            projects=projects,
            num_scouts_per_project=num_scouts_per_project
        )
    
    def _define_region(self, row: int, col: int, radius: int) -> Set[int]:
        """Define project region around center"""
        region = set()
        
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r = row + dr
                c = col + dc
                
                if 0 <= r < self.env.H and 0 <= c < self.env.W:
                    idx = self.env._coords_to_index(r, c)
                    region.add(idx)
        
        return region
    
    def update_project_qualities(self) -> None:
        """
        Update project qualities based on environment (§7)
        
        Q_p^t = w1 * ResourceAbundance + w2 * HydroImpact +
                w3 * Safety - w4 * DistanceCost
        """
        for project in self.state.projects:
            resource_abundance = self._compute_resource_abundance(project)
            hydro_impact = self._compute_hydro_impact(project)
            safety = self._compute_safety(project)
            distance_cost = self._compute_distance_cost(project)
            
            quality = (
                self.cfg.w1 * resource_abundance +
                self.cfg.w2 * hydro_impact +
                self.cfg.w3 * safety -
                self.cfg.w4 * distance_cost
            )
            
            project.quality = quality
    
    def _compute_resource_abundance(self, project: Project) -> float:
        """Compute resource abundance in project region"""
        region_indices = list(project.region)
        
        if len(region_indices) == 0:
            return 0.0
        
        # Sum vegetation biomass
        vegetation = self.env.state.v[region_indices]
        total_vegetation = np.sum(vegetation)
        
        # Normalize by region size
        abundance = total_vegetation / len(region_indices)
        
        return abundance
    
    def _compute_hydro_impact(self, project: Project) -> float:
        """
        Compute potential hydrological impact of dam building
        
        Higher if building here would stabilize water levels
        """
        region_indices = list(project.region)
        
        if len(region_indices) == 0:
            return 0.0
        
        # Check water depth variance in region
        water_depths = self.env.state.h[region_indices]
        variance = np.var(water_depths)
        
        # High variance = high potential for stabilization
        impact = np.sqrt(variance)
        
        return impact
    
    def _compute_safety(self, project: Project) -> float:
        """Compute safety score (inverse of water depth risk)"""
        region_indices = list(project.region)
        
        if len(region_indices) == 0:
            return 0.0
        
        # Avoid very deep water
        water_depths = self.env.state.h[region_indices]
        max_depth = np.max(water_depths)
        
        # Safety decreases with depth
        h_safe = 2.0  # Safe depth threshold
        safety = max(0.0, 1.0 - max_depth / h_safe)
        
        return safety
    
    def _compute_distance_cost(self, project: Project) -> float:
        """
        Compute average distance from core habitat
        
        Projects closer to core are more valuable
        """
        # Get core habitat center
        core_cells = list(self.env.core_habitat)
        if len(core_cells) == 0:
            return 0.0
        
        # Compute average core location
        core_rows = [self.env._index_to_coords(c)[0] for c in core_cells]
        core_cols = [self.env._index_to_coords(c)[1] for c in core_cells]
        core_center_row = np.mean(core_rows)
        core_center_col = np.mean(core_cols)
        
        # Distance from project center to core center
        proj_row, proj_col = project.center
        distance = np.sqrt(
            (proj_row - core_center_row) ** 2 +
            (proj_col - core_center_col) ** 2
        )
        
        # Normalize
        max_distance = np.sqrt(self.env.H ** 2 + self.env.W ** 2)
        cost = distance / max_distance
        
        return cost
    
    def update_recruitment(self, dt: float, scout_returns: Dict[int, int]) -> None:
        """
        Update recruitment dynamics (§7)
        
        R_p^{t+1} = (1 - κ) * R_p^t + Σ_{k ∈ S_p^t} γ_dance * f(Q_p^t)
        
        Args:
            dt: Time step
            scout_returns: Dict mapping project_id -> number of returning scouts
        """
        kappa = self.cfg.kappa
        gamma_dance = self.cfg.gamma_dance
        lambda_Q = self.cfg.lambda_Q
        
        # Update each project
        for project in self.state.projects:
            # Decay
            project.recruitment *= (1 - kappa * dt)
            
            # Scout contribution
            num_scouts = scout_returns.get(project.id, 0)
            if num_scouts > 0:
                # f(Q) = exp(λ * Q)
                f_Q = np.exp(lambda_Q * project.quality)
                project.recruitment += gamma_dance * num_scouts * f_Q * dt
        
        # Apply cross-inhibition (optional)
        if self.cfg.chi > 0:
            self._apply_cross_inhibition()
    
    def _apply_cross_inhibition(self) -> None:
        """
        Apply cross-inhibition between projects (§7)
        
        R_q ← R_q - χ * R_p for q ≠ p
        """
        chi = self.cfg.chi
        
        recruitments = [p.recruitment for p in self.state.projects]
        
        for i, project in enumerate(self.state.projects):
            inhibition = 0.0
            
            for j, other_recruitment in enumerate(recruitments):
                if i != j:
                    inhibition += chi * other_recruitment
            
            project.recruitment = max(0.0, project.recruitment - inhibition)
    
    def select_project(self, beta_R: float) -> int:
        """
        Select project using waggle dance softmax (§10)
        
        P_k(p | R) = exp(β_R * R_p) / Σ_q exp(β_R * R_q)
        
        Args:
            beta_R: Recruitment sharpness (inverse temperature)
        
        Returns:
            Selected project ID
        """
        recruitments = np.array([p.recruitment for p in self.state.projects])
        
        # Add small epsilon to avoid numerical issues
        epsilon = 1e-8
        recruitments = recruitments + epsilon
        
        # Compute softmax
        exp_terms = np.exp(beta_R * recruitments)
        probabilities = exp_terms / np.sum(exp_terms)
        
        # Sample
        project_ids = [p.id for p in self.state.projects]
        selected_id = np.random.choice(project_ids, p=probabilities)
        
        return selected_id
    
    def get_project(self, project_id: int) -> Optional[Project]:
        """Get project by ID"""
        for project in self.state.projects:
            if project.id == project_id:
                return project
        return None
    
    def get_project_metrics(self) -> Dict[str, float]:
        """Get project recruitment metrics"""
        recruitments = [p.recruitment for p in self.state.projects]
        qualities = [p.quality for p in self.state.projects]
        
        # Variance of recruitments
        recruitment_variance = np.var(recruitments)
        
        # Mean quality
        mean_quality = np.mean(qualities)
        
        return {
            'recruitment_variance': recruitment_variance,
            'mean_project_quality': mean_quality,
            'max_recruitment': max(recruitments) if recruitments else 0.0,
            'min_recruitment': min(recruitments) if recruitments else 0.0
        }
    
    def get_recruitment_distribution(self) -> Dict[int, float]:
        """Get recruitment values for all projects"""
        return {p.id: p.recruitment for p in self.state.projects}
