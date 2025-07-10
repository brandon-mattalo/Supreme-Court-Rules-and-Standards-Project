"""
ASAP Active Sampling Integration
Handles active sampling for pairwise comparisons using the ASAP library
"""

import numpy as np
import json
from typing import List, Tuple, Dict, Optional
from scipy.stats import spearmanr
import logging

logger = logging.getLogger(__name__)

class ASAPSampler:
    """
    Wrapper class for ASAP active sampling functionality.
    Handles the active selection of pairs for comparison and convergence checking.
    """
    
    def __init__(self, n_items: int):
        """
        Initialize the ASAP sampler.
        
        Args:
            n_items: Number of items (cases) to compare
        """
        self.n_items = n_items
        self.comparison_matrix = np.zeros((n_items, n_items), dtype=int)
        self.current_scores = None
        self.previous_scores = None
        self.iteration = 0
        self.asap = None
        
        # Try to import ASAP, fall back to random sampling if not available
        try:
            from asap_cpu import ASAP
            self.asap = ASAP(n_items)
            self.use_asap = True
        except ImportError:
            logger.warning("ASAP library not found. Using fallback random sampling.")
            self.use_asap = False
    
    def get_next_pair(self) -> Optional[Tuple[int, int]]:
        """
        Get the next pair of items to compare using active sampling.
        
        Returns:
            Tuple of (item1_idx, item2_idx) or None if converged
        """
        if self.check_convergence():
            return None
        
        # Use fallback for initial coverage, then switch to ASAP
        # Switch to ASAP once we have good coverage (50% of cases compared)
        coverage_stats = self.get_coverage_stats()
        use_fallback = coverage_stats['coverage_percent'] < 50
        
        if use_fallback:
            # Use smart fallback that prioritizes uncompared cases
            result = self._get_least_compared_pair()
            return result
        
        # Use ASAP for more intelligent sampling after initial coverage
        if self.use_asap and self.asap:
            try:
                pairs = self.asap.run_asap(self.comparison_matrix, mst_mode=True)
                if pairs and len(pairs) > 0:
                    return (pairs[0][0], pairs[0][1])
            except Exception as e:
                logger.error(f"ASAP failed: {e}")
                # Fall through to fallback
        
        # Final fallback
        return self._get_least_compared_pair()
    
    def get_sampling_method(self) -> str:
        """
        Get the current sampling method being used.
        
        Returns:
            String describing the current sampling method
        """
        # Check coverage to determine which method is active
        coverage_stats = self.get_coverage_stats()
        
        if coverage_stats['coverage_percent'] < 50:
            return "Random Sampling (Building Coverage)"
        elif self.use_asap and self.asap:
            return "ASAP Active Sampling"
        else:
            return "Random Sampling (Fallback)"
        
        # ASAP code temporarily disabled
        # if self.use_asap and self.asap:
        #     try:
        #         pairs = self.asap.run_asap(self.comparison_matrix, mst_mode=True)
        #         if pairs and len(pairs) > 0:
        #             return (pairs[0][0], pairs[0][1])
        #     except Exception as e:
        #         logger.error(f"ASAP failed: {e}")
        #         # Fall through to fallback
    
    def update_comparison(self, item1_idx: int, item2_idx: int, winner_idx: int):
        """
        Update the comparison matrix with a new result.
        
        Args:
            item1_idx: Index of first item
            item2_idx: Index of second item  
            winner_idx: Index of winning item (must be item1_idx or item2_idx)
        """
        if winner_idx == item1_idx:
            self.comparison_matrix[item1_idx, item2_idx] += 1
        elif winner_idx == item2_idx:
            self.comparison_matrix[item2_idx, item1_idx] += 1
        else:
            raise ValueError(f"Winner index {winner_idx} must be either {item1_idx} or {item2_idx}")
        
        self.iteration += 1
        
        # Update scores
        if self.current_scores is not None:
            # Ensure current_scores is a numpy array before copying
            if hasattr(self.current_scores, 'copy'):
                self.previous_scores = self.current_scores.copy()
            else:
                self.previous_scores = np.array(self.current_scores)
        else:
            self.previous_scores = None
            
        self.current_scores = self.calculate_scores()
    
    def calculate_scores(self) -> np.ndarray:
        """
        Calculate current scores using Bradley-Terry model or ASAP scores.
        
        Returns:
            Array of scores for each item
        """
        if self.use_asap and self.asap:
            # ASAP returns (means, stdevs) tuple - we just want the means
            asap_result = self.asap.get_scores()
            if isinstance(asap_result, tuple):
                return asap_result[0]  # Return just the means
            else:
                return asap_result
        
        # Fallback: Simple win ratio
        wins = np.sum(self.comparison_matrix, axis=1)
        losses = np.sum(self.comparison_matrix, axis=0)
        total_comparisons = wins + losses
        
        # Avoid division by zero
        scores = np.zeros(self.n_items)
        mask = total_comparisons > 0
        scores[mask] = wins[mask] / total_comparisons[mask]
        
        return scores
    
    def check_convergence(self) -> bool:
        """
        Check if the ranking has converged using Spearman correlation.
        Requires every case to be compared at least once before checking convergence.
        
        Returns:
            True if converged (Spearman > 0.99), False otherwise
        """
        if self.current_scores is None or self.previous_scores is None:
            return False
        
        # First requirement: Every case must have been compared at least once
        # Check based on comparison matrix, not scores (scores can be negative/zero)
        wins = np.sum(self.comparison_matrix, axis=1)
        losses = np.sum(self.comparison_matrix, axis=0)
        total_comparisons_per_case = wins + losses
        uncompared_cases = np.sum(total_comparisons_per_case == 0)
        
        if uncompared_cases > 0:
            return False  # Can't converge until all cases have been compared
        
        # Second requirement: Need some variation in scores across the full sample
        if np.std(self.current_scores) < 0.01 or np.std(self.previous_scores) < 0.01:
            return False
        
        # Calculate Spearman correlation on the full ranking of all cases
        current_ranking = np.argsort(-self.current_scores)  # Descending order
        previous_ranking = np.argsort(-self.previous_scores)
        
        correlation, _ = spearmanr(current_ranking, previous_ranking)
        
        return correlation > 0.99
    
    def get_spearman_correlation(self) -> Optional[float]:
        """
        Get the current Spearman correlation between current and previous rankings.
        Returns correlation for all cases if every case has been compared, 
        otherwise returns None to indicate insufficient data.
        
        Returns:
            Spearman correlation coefficient or None if not enough data
        """
        if self.current_scores is None or self.previous_scores is None:
            return None
        
        # Only calculate correlation if every case has been compared at least once
        # Check based on comparison matrix, not scores (scores can be negative/zero)
        wins = np.sum(self.comparison_matrix, axis=1)
        losses = np.sum(self.comparison_matrix, axis=0)
        total_comparisons_per_case = wins + losses
        uncompared_cases = np.sum(total_comparisons_per_case == 0)
        
        if uncompared_cases > 0:
            return None  # Not ready for full correlation yet
        
        # Need some variation in scores
        if np.std(self.current_scores) < 0.01 or np.std(self.previous_scores) < 0.01:
            return None
        
        # Calculate correlation on full ranking of all cases
        current_ranking = np.argsort(-self.current_scores)
        previous_ranking = np.argsort(-self.previous_scores)
        
        correlation, _ = spearmanr(current_ranking, previous_ranking)
        return correlation
    
    def get_coverage_stats(self) -> Dict[str, int]:
        """
        Get coverage statistics for the comparison process.
        
        Returns:
            Dictionary with coverage statistics
        """
        # Base coverage on comparison matrix, not scores (scores can be negative or zero)
        # A case is "compared" if it appears in any comparison (wins OR losses)
        wins = np.sum(self.comparison_matrix, axis=1)
        losses = np.sum(self.comparison_matrix, axis=0)
        total_comparisons_per_case = wins + losses
        
        compared_cases = np.sum(total_comparisons_per_case > 0)
        uncompared_cases = self.n_items - compared_cases
        
        return {
            'compared': int(compared_cases),
            'total': self.n_items,
            'uncompared': int(uncompared_cases),
            'coverage_percent': (compared_cases / self.n_items) * 100
        }
    
    def _get_least_compared_pair(self) -> Optional[Tuple[int, int]]:
        """
        Fallback method: Get the pair with the fewest comparisons.
        Prioritizes cases that haven't been compared at all, then pairs with fewest comparisons.
        
        Returns:
            Tuple of (item1_idx, item2_idx) or None if all pairs compared
        """
        import random
        
        # First priority: Find cases that haven't been compared at all
        if self.current_scores is not None:
            uncompared_cases = [i for i in range(self.n_items) if self.current_scores[i] == 0]
            
            if uncompared_cases:
                # Pick an uncompared case
                case_1 = random.choice(uncompared_cases)
                
                # Try to pair with another uncompared case first
                other_uncompared = [c for c in uncompared_cases if c != case_1]
                if other_uncompared:
                    case_2 = random.choice(other_uncompared)
                    return (min(case_1, case_2), max(case_1, case_2))
                
                # Otherwise pair with any compared case
                compared_cases = [i for i in range(self.n_items) if self.current_scores[i] > 0]
                if compared_cases:
                    case_2 = random.choice(compared_cases)
                    return (min(case_1, case_2), max(case_1, case_2))
        
        # Second priority: Find pairs that have never been compared
        uncompared_pairs = []
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                comparisons = self.comparison_matrix[i, j] + self.comparison_matrix[j, i]
                if comparisons == 0:
                    uncompared_pairs.append((i, j))
        
        # If we have uncompared pairs, randomly select from them
        if uncompared_pairs:
            return random.choice(uncompared_pairs)
        
        # Last priority: Find pairs with minimum comparisons
        min_comparisons = float('inf')
        candidates = []
        
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                comparisons = self.comparison_matrix[i, j] + self.comparison_matrix[j, i]
                if comparisons < min_comparisons:
                    min_comparisons = comparisons
                    candidates = [(i, j)]  # Reset candidates list
                elif comparisons == min_comparisons:
                    candidates.append((i, j))
        
        # Randomly select from candidates with minimum comparisons
        if candidates:
            return random.choice(candidates)
        
        return None
    
    def get_total_comparisons(self) -> int:
        """
        Get the total number of comparisons made so far.
        
        Returns:
            Total comparison count
        """
        return np.sum(self.comparison_matrix)
    
    def get_state_dict(self) -> Dict:
        """
        Get the current state as a dictionary for persistence.
        
        Returns:
            Dictionary containing the sampler state
        """
        return {
            'comparison_matrix': self.comparison_matrix.tolist(),
            'current_scores': self.current_scores.tolist() if self.current_scores is not None else None,
            'previous_scores': self.previous_scores.tolist() if self.previous_scores is not None else None,
            'iteration': self.iteration,
            'spearman_correlation': self.get_spearman_correlation()
        }
    
    def load_state_dict(self, state_dict: Dict):
        """
        Load state from a dictionary.
        
        Args:
            state_dict: Dictionary containing the sampler state
        """
        self.comparison_matrix = np.array(state_dict['comparison_matrix'])
        self.current_scores = np.array(state_dict['current_scores']) if state_dict['current_scores'] else None
        self.previous_scores = np.array(state_dict['previous_scores']) if state_dict['previous_scores'] else None
        self.iteration = state_dict['iteration']
    
    def load_state_from_database(self, experiment_id: int):
        """
        Load ASAP sampling state from the database.
        
        Args:
            experiment_id: ID of the experiment
        """
        from config import execute_sql
        
        # Query for existing active sampling state
        state_data = execute_sql("""
            SELECT comparison_matrix, current_scores, spearman_correlation, iteration_number
            FROM v2_active_sampling_state 
            WHERE experiment_id = ?
        """, (experiment_id,), fetch=True)
        
        if state_data and state_data[0][0]:  # If state exists and has comparison_matrix
            matrix_json, scores_json, spearman_corr, iteration = state_data[0]
            
            try:
                # Load comparison matrix
                if matrix_json:
                    self.comparison_matrix = np.array(json.loads(matrix_json))
                
                # Load current scores
                if scores_json:
                    self.current_scores = np.array(json.loads(scores_json))
                
                # Set iteration
                self.iteration = iteration or 0
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to load ASAP state from database: {e}")
                # Keep default initialized state
        else:
            # No saved state - rebuild matrix from existing comparisons
            print(f"DEBUG: No saved ASAP state found, rebuilding from comparisons...")
            self._rebuild_matrix_from_comparisons(experiment_id)
            print(f"DEBUG: Matrix rebuilt - sum: {self.comparison_matrix.sum()}, iteration: {self.iteration}")
    
    def _rebuild_matrix_from_comparisons(self, experiment_id: int):
        """
        Rebuild the comparison matrix from existing database comparisons.
        
        Args:
            experiment_id: ID of the experiment
        """
        from config import execute_sql
        
        # Get all existing comparisons with extraction indices
        comparisons = execute_sql("""
            SELECT ec.extraction_id_1, ec.extraction_id_2, ec.winner_id,
                   ee1.extraction_id as ext1_id, ee2.extraction_id as ext2_id
            FROM v2_experiment_comparisons ec
            JOIN v2_experiment_extractions ee1 ON ec.extraction_id_1 = ee1.extraction_id  
            JOIN v2_experiment_extractions ee2 ON ec.extraction_id_2 = ee2.extraction_id
            WHERE ec.experiment_id = ?
            ORDER BY ee1.extraction_id, ee2.extraction_id
        """, (experiment_id,), fetch=True)
        
        if comparisons:
            # Create extraction_id to matrix index mapping
            all_extractions = execute_sql("""
                SELECT extraction_id FROM v2_experiment_extractions 
                WHERE experiment_id = ? 
                ORDER BY extraction_id
            """, (experiment_id,), fetch=True)
            
            id_to_index = {ext_id[0]: idx for idx, ext_id in enumerate(all_extractions)}
            
            # Rebuild matrix from comparisons
            for comp in comparisons:
                extraction_id_1, extraction_id_2, winner_id = comp[:3]
                
                idx_1 = id_to_index.get(extraction_id_1)
                idx_2 = id_to_index.get(extraction_id_2) 
                
                if idx_1 is not None and idx_2 is not None:
                    # Update matrix based on winner
                    if winner_id == extraction_id_1:
                        self.comparison_matrix[idx_1, idx_2] += 1
                    elif winner_id == extraction_id_2:
                        self.comparison_matrix[idx_2, idx_1] += 1
                        
                    self.iteration += 1
            
            # Calculate initial scores if we have comparisons
            if self.iteration > 0:
                self.current_scores = self.calculate_scores()
                
            logger.info(f"Rebuilt ASAP matrix from {len(comparisons)} existing comparisons")
    
    def save_state_to_database(self, experiment_id: int):
        """
        Save current ASAP sampling state to the database.
        
        Args:
            experiment_id: ID of the experiment
        """
        from config import execute_sql
        
        # Prepare state data
        matrix_json = json.dumps(self.comparison_matrix.tolist())
        scores_json = json.dumps(self.current_scores.tolist()) if self.current_scores is not None else None
        spearman_corr = self.get_spearman_correlation()
        
        # Upsert state (insert or update)
        execute_sql("""
            INSERT INTO v2_active_sampling_state 
            (experiment_id, comparison_matrix, current_scores, spearman_correlation, iteration_number, last_updated)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (experiment_id) DO UPDATE SET
                comparison_matrix = EXCLUDED.comparison_matrix,
                current_scores = EXCLUDED.current_scores,
                spearman_correlation = EXCLUDED.spearman_correlation,
                iteration_number = EXCLUDED.iteration_number,
                last_updated = CURRENT_TIMESTAMP
        """, (experiment_id, matrix_json, scores_json, spearman_corr, self.iteration))
    
    def add_comparison_result(self, item1_idx: int, item2_idx: int, winner: int):
        """
        Add a comparison result. Alias for update_comparison for compatibility.
        
        Args:
            item1_idx: Index of first item
            item2_idx: Index of second item
            winner: 1 if item1 wins, 0 if item2 wins
        """
        winner_idx = item1_idx if winner == 1 else item2_idx
        self.update_comparison(item1_idx, item2_idx, winner_idx)


def estimate_required_comparisons(n_items: int) -> int:
    """
    Estimate the number of comparisons needed for active sampling.
    
    For active sampling, a good rule of thumb is O(n log n) comparisons.
    This is much more efficient than full pairwise O(n^2).
    
    Args:
        n_items: Number of items to compare
        
    Returns:
        Estimated number of comparisons
    """
    if n_items <= 1:
        return 0
    
    # Use n * log2(n) * 2 as a conservative estimate
    # The factor of 2 provides some buffer for convergence
    import math
    return int(n_items * math.log2(n_items) * 2)


def calculate_information_gain(comparison_matrix: np.ndarray, item1_idx: int, item2_idx: int) -> float:
    """
    Calculate the information gain for comparing two items.
    
    This is a simplified version - the actual ASAP library uses more sophisticated methods.
    
    Args:
        comparison_matrix: Current comparison matrix
        item1_idx: Index of first item
        item2_idx: Index of second item
        
    Returns:
        Information gain value
    """
    # Count existing comparisons for this pair
    existing_comparisons = comparison_matrix[item1_idx, item2_idx] + comparison_matrix[item2_idx, item1_idx]
    
    # Less information gain if we've already compared these items many times
    if existing_comparisons == 0:
        return 1.0
    else:
        return 1.0 / (1.0 + existing_comparisons)