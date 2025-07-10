"""
ASAP Active Sampling Integration
Handles active sampling for pairwise comparisons using the ASAP library
"""

import numpy as np
import json
from typing import List, Tuple, Dict, Optional, Any
from scipy.stats import spearmanr
import logging

logger = logging.getLogger(__name__)

# Try to import choix for Bradley-Terry calculations
try:
    import choix
    CHOIX_AVAILABLE = True
except ImportError:
    CHOIX_AVAILABLE = False
    logger.warning("choix library not available - Bradley-Terry convergence disabled")

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
        self.asap_approx = None
        self.asap_full = None
        
        # Bradley-Terry tracking (parallel to TrueSkill)
        self.bt_scores = None
        self.bt_previous_scores = None
        self.bt_comparisons = []  # List of (winner_idx, loser_idx) tuples for choix
        self.bt_convergence_history = []
        
        # Absorbing class tracking for progress indication
        self.absorbing_classes_history = []  # Track count over time
        
        # Persistence tracking for adaptive sampling
        self.absorbing_class_persistence = {}  # {case_idx: {'first_seen': iteration, 'comparisons': count, 'wins': count}}
        self.sampling_mode = 'normal'  # 'normal' or 'absorbing_targeted'
        self.last_mode_switch_iteration = 0
        
        # Try to import ASAP, fall back to random sampling if not available
        try:
            from asap_cpu import ASAP
            # Create both approximate and full ASAP instances
            self.asap_approx = ASAP(n_items, approx=True)
            self.asap_full = ASAP(n_items, approx=False)
            self.use_asap = True
            logger.info(f"ASAP initialized with {n_items} items (both approx and full modes)")
        except ImportError:
            logger.warning("ASAP library not found. Using fallback random sampling.")
            self.use_asap = False
    
    def get_next_pair(self) -> Optional[Tuple[int, int]]:
        """
        Get the next pair of items to compare using adaptive sampling.
        
        Adaptive approach:
        1. Random sampling until 100% coverage
        2. Normal ASAP mode for general exploration
        3. Absorbing-targeted mode when persistent absorbing classes detected
        
        Returns:
            Tuple of (item1_idx, item2_idx) or None if converged
        """
        if self.check_convergence():
            return None
        
        coverage_stats = self.get_coverage_stats()
        coverage_pct = coverage_stats['coverage_percent']
        
        # Phase 1: Use random sampling until we have 100% coverage
        if coverage_pct < 100:
            result = self._get_least_compared_pair()
            self._set_last_method("Random")
            return result
        
        # Check if we should use absorbing-targeted sampling
        if self.sampling_mode == 'absorbing_targeted':
            result = self._get_absorbing_targeted_pair()
            if result is not None:
                self._set_last_method("Absorbing-Targeted")
                return result
            # Fall back to normal ASAP if no absorbing-targeted pairs available
        
        # Phase 2 & 3: Use ASAP after 100% coverage
        if self.use_asap and (self.asap_approx or self.asap_full):
            try:
                import time
                
                # Decide whether to use ASAP-full or ASAP-approx
                # Start with approx, then test if full would be fast enough
                # ASAP computational complexity depends on n_items and comparison density
                total_comparisons = self.get_total_comparisons()
                comparison_density = total_comparisons / (self.n_items * (self.n_items - 1) / 2)
                
                # Use full for small datasets or low comparison density
                # These heuristics prevent expensive computations
                use_full = (
                    self.n_items <= 30 or  # Very small dataset
                    (self.n_items <= 50 and comparison_density < 0.1) or  # Small dataset with sparse comparisons
                    (self.n_items <= 100 and total_comparisons < 100)  # Medium dataset with few comparisons
                )
                
                if use_full and self.asap_full:
                    # Phase 3: Use ASAP-full for small datasets
                    start_time = time.time()
                    pairs = self.asap_full.run_asap(self.comparison_matrix, mst_mode=True)
                    elapsed = time.time() - start_time
                    
                    if elapsed > 1.0:
                        # If it took too long, switch back to approx for next time
                        logger.info(f"ASAP-full took {elapsed:.2f}s, switching to approx")
                        use_full = False
                    else:
                        logger.info(f"ASAP-full completed in {elapsed:.2f}s")
                        
                    if pairs is not None and len(pairs) > 0:
                        self._set_last_method("ASAP Full")
                        return (pairs[0][0], pairs[0][1])
                        
                # Phase 2: Use ASAP-approx for larger datasets
                if not use_full and self.asap_approx:
                    start_time = time.time()
                    pairs = self.asap_approx.run_asap(self.comparison_matrix, mst_mode=True)
                    elapsed = time.time() - start_time
                    logger.info(f"ASAP-approx completed in {elapsed:.2f}s")
                    
                    if pairs is not None and len(pairs) > 0:
                        self._set_last_method("ASAP Approx")
                        return (pairs[0][0], pairs[0][1])
                
                # ASAP returned no pairs, fall back to random
                logger.info("ASAP returned no pairs, using fallback")
                self._set_last_method("ASAP Random Fallback")
                    
            except Exception as e:
                logger.error(f"ASAP failed: {e}")
                self._set_last_method("ASAP Random Fallback")
                # Fall through to fallback
        
        # Final fallback
        result = self._get_least_compared_pair()
        if not hasattr(self, '_last_method_used'):
            self._set_last_method("Random")
        return result
    
    def get_sampling_method(self) -> str:
        """
        Get the current sampling method being used.
        
        Returns:
            String describing the current sampling method (one of four options):
            - "Random"
            - "ASAP Full"
            - "ASAP Approx"
            - "ASAP Random Fallback"
        """
        # Return the last method that was actually used
        if hasattr(self, '_last_method_used'):
            return self._last_method_used
        
        # Default based on coverage
        coverage_stats = self.get_coverage_stats()
        if coverage_stats['coverage_percent'] < 100:
            return "Random"
        else:
            return "ASAP Approx"  # Default assumption
    
    def _set_last_method(self, method: str):
        """Set the method that was actually used in the last iteration."""
        self._last_method_used = method
    
    def _update_bradley_terry(self, winner_idx: int, loser_idx: int):
        """
        Update Bradley-Terry scores using choix library with connectivity checking.
        
        Args:
            winner_idx: Index of winning item
            loser_idx: Index of losing item
        """
        if not CHOIX_AVAILABLE:
            return
            
        # Add comparison to choix format
        self.bt_comparisons.append((winner_idx, loser_idx))
        
        # Check graph connectivity before running choix (using same logic as choix_analysis)
        connectivity_info = self.check_graph_connectivity()
        is_bt_connected = connectivity_info.get('is_connected', False)
        
        if not is_bt_connected:
            # Graph has absorbing classes - skip choix calculation to avoid errors
            # Set BT scores to None to indicate disconnected state
            self.bt_scores = None
            
            # Better logging with absorbing class information and progress tracking
            absorbing_info = connectivity_info.get('absorbing_classes', {})
            if absorbing_info.get('has_absorbing_classes'):
                n_absorbing = absorbing_info.get('n_absorbing_classes', 0)
                
                # Track absorbing class progress
                self.absorbing_classes_history.append({
                    'iteration': len(self.bt_comparisons),
                    'count': n_absorbing,
                    'members': []  # Not available in this context
                })
                
                # Show progress if classes have been reduced
                if len(self.absorbing_classes_history) > 1:
                    previous_entry = self.absorbing_classes_history[-2]
                    previous_count = previous_entry.get('count', 0) if isinstance(previous_entry, dict) else previous_entry
                    if n_absorbing < previous_count:
                        improvement = previous_count - n_absorbing
                        print(f"DEBUG: After comparison {len(self.bt_comparisons)}: {n_absorbing} absorbing classes remain (down {improvement} from {previous_count})")
                    elif len(self.bt_comparisons) % 20 == 0:  # Periodic update every 20 comparisons
                        print(f"DEBUG: After comparison {len(self.bt_comparisons)}: {n_absorbing} absorbing classes remain (no change)")
                else:
                    print(f"DEBUG: After comparison {len(self.bt_comparisons)}: {n_absorbing} absorbing classes detected")
            else:
                print(f"DEBUG: After comparison {len(self.bt_comparisons)}: Graph connectivity issue")
            return
        else:
            # True BT connectivity achieved - log the success
            if self.bt_scores is None:
                print(f"DEBUG: After comparison {len(self.bt_comparisons)}: Bradley-Terry connectivity ACHIEVED! (No more absorbing classes)")
            elif len(self.bt_comparisons) % 10 == 0:  # Log every 10 comparisons
                print(f"DEBUG: After comparison {len(self.bt_comparisons)}: BT connectivity maintained")
        
        # Store previous BT scores
        if self.bt_scores is not None:
            self.bt_previous_scores = self.bt_scores.copy()
        
        # Calculate new BT scores using choix (only when connected)
        try:
            # choix expects comparisons as list of (winner, loser) tuples
            self.bt_scores = choix.ilsr_pairwise(
                n_items=self.n_items,
                data=self.bt_comparisons,
                initial_params=None
            )
            
            # Track convergence
            if self.bt_previous_scores is not None:
                score_change = np.max(np.abs(self.bt_scores - self.bt_previous_scores))
                self.bt_convergence_history.append(score_change)
                
        except Exception as e:
            # Even with connectivity check, choix might still fail due to numerical issues
            logger.info(f"Bradley-Terry calculation failed despite connectivity check: {e}")
            self.bt_scores = None
            
    def get_bradley_terry_scores(self) -> Optional[np.ndarray]:
        """Get current Bradley-Terry scores."""
        return self.bt_scores
    
    def get_bradley_terry_convergence(self, tolerance: float = 1e-4) -> bool:
        """
        Check if Bradley-Terry scores have converged.
        
        Args:
            tolerance: Maximum score change to consider converged
            
        Returns:
            True if BT scores have converged
        """
        if not CHOIX_AVAILABLE or len(self.bt_convergence_history) < 3:
            return False
            
        # Check if recent score changes are below tolerance
        recent_changes = self.bt_convergence_history[-3:]
        return all(change < tolerance for change in recent_changes)
    
    def get_bradley_terry_ranking_correlation(self) -> Optional[float]:
        """
        Get Spearman correlation between current and previous BT rankings.
        
        Returns:
            Spearman correlation or None if insufficient data
        """
        if (not CHOIX_AVAILABLE or 
            self.bt_scores is None or 
            self.bt_previous_scores is None):
            return None
            
        try:
            # Calculate rankings (higher scores = better ranks)
            current_ranking = np.argsort(-self.bt_scores)
            previous_ranking = np.argsort(-self.bt_previous_scores)
            
            correlation, _ = spearmanr(current_ranking, previous_ranking)
            return correlation
        except Exception:
            return None
    
    def update_comparison(self, item1_idx: int, item2_idx: int, winner_idx: int):
        """
        Update the comparison matrix with a new result.
        Updates both TrueSkill (for ASAP) and Bradley-Terry (for convergence) in parallel.
        
        Args:
            item1_idx: Index of first item
            item2_idx: Index of second item  
            winner_idx: Index of winning item (must be item1_idx or item2_idx)
        """
        if winner_idx == item1_idx:
            self.comparison_matrix[item1_idx, item2_idx] += 1
            loser_idx = item2_idx
        elif winner_idx == item2_idx:
            self.comparison_matrix[item2_idx, item1_idx] += 1
            loser_idx = item1_idx
        else:
            raise ValueError(f"Winner index {winner_idx} must be either {item1_idx} or {item2_idx}")
        
        self.iteration += 1
        
        # Update TrueSkill scores (for ASAP pair selection)
        if self.current_scores is not None:
            # Ensure current_scores is a numpy array before copying
            if hasattr(self.current_scores, 'copy'):
                self.previous_scores = self.current_scores.copy()
            else:
                self.previous_scores = np.array(self.current_scores)
        else:
            self.previous_scores = None
            
        self.current_scores = self.calculate_scores()
        
        # Update Bradley-Terry tracking (for convergence detection)
        self._update_bradley_terry(winner_idx, loser_idx)
        
        # Update absorbing class persistence tracking
        self._update_absorbing_class_persistence()
    
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
        Check if the ranking has converged using dual criteria:
        1. TrueSkill Spearman correlation (for ASAP)
        2. Bradley-Terry statistical convergence (for final analysis)
        
        Returns:
            True if either convergence method indicates completion
        """
        # First requirement: Every case must have been compared at least once
        wins = np.sum(self.comparison_matrix, axis=1)
        losses = np.sum(self.comparison_matrix, axis=0)
        total_comparisons_per_case = wins + losses
        uncompared_cases = np.sum(total_comparisons_per_case == 0)
        
        if uncompared_cases > 0:
            return False  # Can't converge until all cases have been compared
        
        # Check TrueSkill convergence (existing method)
        trueskill_converged = self._check_trueskill_convergence()
        
        # Check Bradley-Terry convergence (new method)
        bt_converged = self.get_bradley_terry_convergence()
        
        # Check Bradley-Terry ranking stability
        bt_ranking_stable = False
        bt_correlation = self.get_bradley_terry_ranking_correlation()
        if bt_correlation is not None:
            bt_ranking_stable = bt_correlation > 0.99
        
        # CRITICAL: Check for absorbing classes - can't converge if they exist
        connectivity_info = self.check_graph_connectivity()
        bt_connected = connectivity_info.get('is_connected', False)
        
        if not bt_connected:
            # Even if TrueSkill converged, can't stop if absorbing classes remain
            absorbing_info = connectivity_info.get('absorbing_classes', {})
            if absorbing_info.get('has_absorbing_classes'):
                n_absorbing = absorbing_info.get('n_absorbing_classes', 0)
                if n_absorbing > 0:
                    print(f"DEBUG: Convergence blocked - {n_absorbing} absorbing classes remain")
                    return False  # Block convergence until absorbing classes resolved
        
        # Return True if either method indicates convergence AND no absorbing classes
        return (trueskill_converged or bt_converged or bt_ranking_stable) and bt_connected
    
    def _check_trueskill_convergence(self) -> bool:
        """
        Check TrueSkill convergence using Spearman correlation.
        
        Returns:
            True if TrueSkill scores have converged
        """
        if self.current_scores is None or self.previous_scores is None:
            return False
        
        # Need some variation in scores across the full sample
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
    
    def _get_absorbing_targeted_pair(self) -> Optional[Tuple[int, int]]:
        """
        Get a strategically selected pair when in absorbing-targeted sampling mode.
        Prioritizes comparisons involving absorbing class members to help resolve them.
        
        Returns:
            Tuple of (item1_idx, item2_idx) or None if no good targets found
        """
        import random
        
        # Get current absorbing class members
        persistent_absorbing = self.get_persistent_absorbing_classes()
        if not persistent_absorbing:
            return None
        
        absorbing_indices = list(persistent_absorbing.keys())
        
        # Strategy 1: Compare absorbing class members with high-connectivity cases
        high_connectivity_cases = self._get_high_connectivity_cases(exclude=absorbing_indices)
        
        comparison_candidates = []
        
        # Priority 1: Absorbing vs. high-connectivity cases they haven't been compared with
        for abs_idx in absorbing_indices:
            for conn_idx in high_connectivity_cases:
                existing_comps = self.comparison_matrix[abs_idx, conn_idx] + self.comparison_matrix[conn_idx, abs_idx]
                if existing_comps == 0:  # Not compared yet
                    comparison_candidates.append((abs_idx, conn_idx, 'abs_vs_high_conn', 3))
                elif existing_comps <= 2:  # Compared few times
                    comparison_candidates.append((abs_idx, conn_idx, 'abs_vs_high_conn', 2))
        
        # Priority 2: Inter-absorbing class comparisons (to establish hierarchy)
        for i, abs_idx_1 in enumerate(absorbing_indices):
            for abs_idx_2 in absorbing_indices[i+1:]:
                existing_comps = self.comparison_matrix[abs_idx_1, abs_idx_2] + self.comparison_matrix[abs_idx_2, abs_idx_1]
                if existing_comps <= 1:  # Few or no comparisons between absorbing classes
                    comparison_candidates.append((abs_idx_1, abs_idx_2, 'inter_absorbing', 2))
        
        # Priority 3: Absorbing vs. cases they haven't been compared with at all
        for abs_idx in absorbing_indices:
            for other_idx in range(self.n_items):
                if other_idx == abs_idx or other_idx in absorbing_indices:
                    continue
                existing_comps = self.comparison_matrix[abs_idx, other_idx] + self.comparison_matrix[other_idx, abs_idx]
                if existing_comps == 0:
                    comparison_candidates.append((abs_idx, other_idx, 'abs_vs_uncompared', 1))
        
        # Sort by priority and select randomly from highest priority group
        if comparison_candidates:
            comparison_candidates.sort(key=lambda x: x[3], reverse=True)
            max_priority = comparison_candidates[0][3]
            top_candidates = [c for c in comparison_candidates if c[3] == max_priority]
            
            selected = random.choice(top_candidates)
            return (selected[0], selected[1])
        
        return None
    
    def _get_high_connectivity_cases(self, exclude: List[int] = None, top_n: int = 10) -> List[int]:
        """
        Get cases with high connectivity (many comparisons) for strategic pairing.
        
        Args:
            exclude: List of case indices to exclude
            top_n: Number of top connected cases to return
            
        Returns:
            List of case indices sorted by connectivity (descending)
        """
        if exclude is None:
            exclude = []
        
        connectivity_scores = []
        for i in range(self.n_items):
            if i not in exclude:
                total_comps = np.sum(self.comparison_matrix[i, :]) + np.sum(self.comparison_matrix[:, i])
                connectivity_scores.append((i, total_comps))
        
        # Sort by connectivity (descending) and return top N
        connectivity_scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in connectivity_scores[:top_n]]
    
    def get_total_comparisons(self) -> int:
        """
        Get the total number of comparisons made so far.
        
        Returns:
            Total comparison count
        """
        return np.sum(self.comparison_matrix)
    
    def check_graph_connectivity(self) -> Dict[str, Any]:
        """
        Check if the comparison graph is Bradley-Terry connected using same logic as choix_analysis.
        
        Returns:
            Dictionary with detailed connectivity information including absorbing classes
        """
        if len(self.bt_comparisons) == 0:
            return {
                'is_connected': False,
                'is_basic_connected': False,
                'reason': 'no_comparisons',
                'absorbing_classes_info': {'has_absorbing_classes': True, 'n_absorbing_classes': self.n_items}
            }
        
        # Import the choix connectivity checker to use exact same logic
        try:
            from utils.choix_analysis import check_graph_connectivity
            # Create dummy extraction IDs for compatibility
            extraction_ids = [str(i) for i in range(self.n_items)]
            connectivity_info = check_graph_connectivity(self.bt_comparisons, self.n_items, extraction_ids)
            return connectivity_info
        except ImportError:
            # Fallback to simple connectivity if choix_analysis not available
            return self._simple_connectivity_check()
    
    def _simple_connectivity_check(self) -> Dict[str, Any]:
        """
        Fallback simple connectivity check.
        
        Returns:
            Basic connectivity information
        """        
        # Build adjacency set for undirected graph
        adjacency = {i: set() for i in range(self.n_items)}
        
        for winner_idx, loser_idx in self.bt_comparisons:
            if 0 <= winner_idx < self.n_items and 0 <= loser_idx < self.n_items:
                adjacency[winner_idx].add(loser_idx)
                adjacency[loser_idx].add(winner_idx)
        
        # Find connected components using DFS
        visited = set()
        components = []
        
        def dfs(node, component):
            if node in visited:
                return
            visited.add(node)
            component.append(node)
            for neighbor in adjacency[node]:
                dfs(neighbor, component)
        
        for i in range(self.n_items):
            if i not in visited:
                component = []
                dfs(i, component)
                if component:  # Only add non-empty components
                    components.append(component)
        
        # Handle case where some nodes have no connections at all
        if len(visited) < self.n_items:
            # Add isolated nodes as single-node components
            for i in range(self.n_items):
                if i not in visited:
                    components.append([i])
        
        # Graph is connected only if there's exactly one component with all nodes
        is_basic_connected = len(components) == 1 and len(components[0]) == self.n_items
        
        # For fallback, assume basic connectivity = BT connectivity (not accurate but safe)
        return {
            'is_connected': is_basic_connected,
            'is_basic_connected': is_basic_connected,
            'n_components': len(components),
            'components': components,
            'component_sizes': [len(comp) for comp in components],
            'largest_component_size': max([len(comp) for comp in components]) if components else 0,
            'absorbing_classes': {'has_absorbing_classes': not is_basic_connected, 'fallback': True}
        }
    
    def is_graph_connected(self) -> bool:
        """
        Public method to check if comparison graph is Bradley-Terry connected.
        Used by UI to show connectivity warnings.
        
        Returns:
            True if BT connected (choix can compute scores), False if absorbing classes exist
        """
        connectivity_info = self.check_graph_connectivity()
        return connectivity_info.get('is_connected', False)
    
    def get_connectivity_info(self) -> Dict[str, Any]:
        """
        Get detailed connectivity information for UI display.
        
        Returns:
            Dictionary with connectivity details including absorbing class info
        """
        return self.check_graph_connectivity()
    
    def get_state_dict(self) -> Dict:
        """
        Get the current state as a dictionary for persistence.
        
        Returns:
            Dictionary containing the sampler state including BT data
        """
        # Convert all data to JSON-safe native Python types
        spearman_corr = self.get_spearman_correlation()
        bt_ranking_corr = self.get_bradley_terry_ranking_correlation()
        
        return {
            'comparison_matrix': [[int(val) for val in row] for row in self.comparison_matrix.tolist()],
            'current_scores': [float(score) for score in self.current_scores.tolist()] if self.current_scores is not None else None,
            'previous_scores': [float(score) for score in self.previous_scores.tolist()] if self.previous_scores is not None else None,
            'iteration': int(self.iteration),
            'spearman_correlation': float(spearman_corr) if spearman_corr is not None else None,
            'bt_scores': [float(score) for score in self.bt_scores.tolist()] if self.bt_scores is not None else None,
            'bt_previous_scores': [float(score) for score in self.bt_previous_scores.tolist()] if self.bt_previous_scores is not None else None,
            'bt_comparisons': [(int(winner), int(loser)) for winner, loser in self.bt_comparisons] if self.bt_comparisons else [],
            'bt_convergence_history': [float(change) for change in self.bt_convergence_history] if self.bt_convergence_history else [],
            'bt_ranking_correlation': float(bt_ranking_corr) if bt_ranking_corr is not None else None,
            'absorbing_classes_history': [int(count) for count in self.absorbing_classes_history] if self.absorbing_classes_history else []
        }
    
    def load_state_dict(self, state_dict: Dict):
        """
        Load state from a dictionary.
        
        Args:
            state_dict: Dictionary containing the sampler state including BT data
        """
        self.comparison_matrix = np.array(state_dict['comparison_matrix'])
        self.current_scores = np.array(state_dict['current_scores']) if state_dict['current_scores'] else None
        self.previous_scores = np.array(state_dict['previous_scores']) if state_dict['previous_scores'] else None
        self.iteration = state_dict['iteration']
        
        # Load Bradley-Terry state if available
        if 'bt_scores' in state_dict and state_dict['bt_scores']:
            self.bt_scores = np.array(state_dict['bt_scores'])
        if 'bt_previous_scores' in state_dict and state_dict['bt_previous_scores']:
            self.bt_previous_scores = np.array(state_dict['bt_previous_scores'])
        if 'bt_comparisons' in state_dict:
            self.bt_comparisons = state_dict['bt_comparisons']
        if 'bt_convergence_history' in state_dict:
            self.bt_convergence_history = state_dict['bt_convergence_history']
        if 'absorbing_classes_history' in state_dict:
            self.absorbing_classes_history = state_dict['absorbing_classes_history']
    
    def load_state_from_database(self, experiment_id: int):
        """
        Load ASAP sampling state from the database and sync with actual comparisons.
        
        Args:
            experiment_id: ID of the experiment
        """
        from config import execute_sql
        
        # Query for existing active sampling state including BT fields
        state_data = execute_sql("""
            SELECT comparison_matrix, current_scores, spearman_correlation, iteration_number,
                   bt_scores, bt_comparisons, bt_convergence_history, bt_ranking_correlation
            FROM v2_active_sampling_state 
            WHERE experiment_id = ?
        """, (experiment_id,), fetch=True)
        
        if state_data and state_data[0][0]:  # If state exists and has comparison_matrix
            (matrix_json, scores_json, spearman_corr, iteration, 
             bt_scores_json, bt_comparisons_json, bt_conv_json, bt_ranking_corr) = state_data[0]
            
            try:
                # Load comparison matrix
                if matrix_json:
                    self.comparison_matrix = np.array(json.loads(matrix_json))
                
                # Load current scores
                if scores_json:
                    self.current_scores = np.array(json.loads(scores_json))
                
                # Set iteration
                self.iteration = iteration or 0
                
                # Load Bradley-Terry state
                if bt_scores_json:
                    self.bt_scores = np.array(json.loads(bt_scores_json))
                if bt_comparisons_json:
                    self.bt_comparisons = json.loads(bt_comparisons_json)
                if bt_conv_json:
                    self.bt_convergence_history = json.loads(bt_conv_json)
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to load ASAP state from database: {e}")
                # Keep default initialized state
        else:
            # No saved state - rebuild matrix from existing comparisons
            print(f"DEBUG: No saved ASAP state found, rebuilding from comparisons...")
            self._rebuild_matrix_from_comparisons(experiment_id)
            print(f"DEBUG: Matrix rebuilt - sum: {self.comparison_matrix.sum()}, iteration: {self.iteration}")
        
        # CRITICAL: Always sync BT comparisons with actual database regardless of saved state
        self._sync_bt_comparisons_with_database(experiment_id)
        
        # Force BT connectivity check and score recalculation after sync
        if len(self.bt_comparisons) > 0:
            print(f"DEBUG: Synced {len(self.bt_comparisons)} BT comparisons, checking connectivity...")
            is_connected = self.check_graph_connectivity()
            print(f"DEBUG: Graph connectivity after sync: {is_connected}")
            
            # Force recalculation of BT scores if connected
            if is_connected:
                self._force_bt_recalculation()
            else:
                print(f"DEBUG: Graph disconnected, skipping BT score calculation")
        else:
            print(f"DEBUG: No BT comparisons found after sync")
    
    def _rebuild_matrix_from_comparisons(self, experiment_id: int):
        """
        Rebuild the comparison matrix from existing database comparisons.
        
        Args:
            experiment_id: ID of the experiment
        """
        from config import execute_sql
        
        # First get all extractions to build the extraction_id to matrix_index mapping
        all_extractions = execute_sql("""
            SELECT extraction_id 
            FROM v2_experiment_extractions 
            WHERE experiment_id = ? 
            ORDER BY extraction_id
        """, (experiment_id,), fetch=True)
        
        id_to_index = {ext_id[0]: idx for idx, ext_id in enumerate(all_extractions)}
        
        # Get all existing comparisons
        comparisons = execute_sql("""
            SELECT ec.extraction_id_1, ec.extraction_id_2, ec.winner_id
            FROM v2_experiment_comparisons ec
            WHERE ec.experiment_id = ?
            ORDER BY ec.comparison_date
        """, (experiment_id,), fetch=True)
        
        # Rebuild matrix from comparisons
        valid_comparisons = 0
        for comp in comparisons:
            extraction_id_1, extraction_id_2, winner_id = comp[:3]
            
            idx_1 = id_to_index.get(extraction_id_1)
            idx_2 = id_to_index.get(extraction_id_2) 
            
            if idx_1 is not None and idx_2 is not None:
                # Update matrix based on winner
                if winner_id == extraction_id_1:
                    self.comparison_matrix[idx_1, idx_2] += 1
                    valid_comparisons += 1
                elif winner_id == extraction_id_2:
                    self.comparison_matrix[idx_2, idx_1] += 1
                    valid_comparisons += 1
        
        # Set iteration to total number of valid comparisons
        self.iteration = valid_comparisons
        
        # Calculate initial scores if we have comparisons
        if self.iteration > 0:
            self.current_scores = self.calculate_scores()
        
        logger.info(f"Rebuilt ASAP matrix from {valid_comparisons} valid comparisons out of {len(comparisons)} total")
    
    def _sync_bt_comparisons_with_database(self, experiment_id: int):
        """
        Sync Bradley-Terry comparisons list with actual database comparisons.
        This ensures ASAP state reflects all comparisons that exist in the database.
        
        Args:
            experiment_id: ID of the experiment
        """
        from config import execute_sql
        
        # Get all existing comparisons with extraction IDs
        comparisons = execute_sql("""
            SELECT ec.extraction_id_1, ec.extraction_id_2, ec.winner_id
            FROM v2_experiment_comparisons ec
            WHERE ec.experiment_id = ?
            ORDER BY ec.comparison_date
        """, (experiment_id,), fetch=True)
        
        if not comparisons:
            print(f"DEBUG: No comparisons found in database for experiment {experiment_id}")
            self.bt_comparisons = []
            return
        
        # Get extraction ID to index mapping
        all_extractions = execute_sql("""
            SELECT extraction_id FROM v2_experiment_extractions 
            WHERE experiment_id = ? 
            ORDER BY extraction_id
        """, (experiment_id,), fetch=True)
        
        id_to_index = {ext_id[0]: idx for idx, ext_id in enumerate(all_extractions)}
        
        # Convert database comparisons to choix format (winner_idx, loser_idx)
        bt_comparisons_new = []
        
        for extraction_id_1, extraction_id_2, winner_id in comparisons:
            idx_1 = id_to_index.get(extraction_id_1)
            idx_2 = id_to_index.get(extraction_id_2)
            
            if idx_1 is not None and idx_2 is not None and winner_id is not None:
                if winner_id == extraction_id_1:
                    bt_comparisons_new.append((idx_1, idx_2))  # idx_1 beats idx_2
                elif winner_id == extraction_id_2:
                    bt_comparisons_new.append((idx_2, idx_1))  # idx_2 beats idx_1
                # Skip ties or invalid winners
        
        # Update Bradley-Terry comparisons list
        old_count = len(self.bt_comparisons) if self.bt_comparisons else 0
        self.bt_comparisons = bt_comparisons_new
        new_count = len(self.bt_comparisons)
        
        print(f"DEBUG: BT comparisons sync - Old: {old_count}, New: {new_count}, Database: {len(comparisons)}")
        
        # CRITICAL: Sync iteration count with actual database comparisons
        old_iteration = self.iteration
        self.iteration = len(comparisons)  # Total database comparisons
        
        if old_iteration != self.iteration:
            print(f"DEBUG: Iteration count sync - Old: {old_iteration}, New: {self.iteration}")
        
        # Invalidate BT scores since comparisons changed
        if old_count != new_count:
            self.bt_scores = None
            self.bt_previous_scores = None
            print(f"DEBUG: BT scores invalidated due to comparison count change")
    
    def _force_bt_recalculation(self):
        """
        Force recalculation of Bradley-Terry scores using current comparisons.
        Used after syncing with database to ensure accurate connectivity status.
        """
        if not CHOIX_AVAILABLE or not self.bt_comparisons:
            print(f"DEBUG: Cannot recalculate BT - choix available: {CHOIX_AVAILABLE}, comparisons: {len(self.bt_comparisons) if self.bt_comparisons else 0}")
            return
            
        try:
            print(f"DEBUG: Recalculating BT scores with {len(self.bt_comparisons)} comparisons...")
            
            # Store previous scores if they exist
            if self.bt_scores is not None:
                self.bt_previous_scores = self.bt_scores.copy()
            
            # Calculate new BT scores using choix
            self.bt_scores = choix.ilsr_pairwise(
                n_items=self.n_items,
                data=self.bt_comparisons,
                initial_params=None
            )
            
            print(f"DEBUG: BT recalculation successful - scores calculated for {len(self.bt_scores)} items")
            
            # Update convergence history if we have previous scores
            if self.bt_previous_scores is not None:
                score_change = np.max(np.abs(self.bt_scores - self.bt_previous_scores))
                self.bt_convergence_history.append(score_change)
                print(f"DEBUG: BT score change: {score_change:.6f}")
            
        except Exception as e:
            print(f"DEBUG: BT recalculation failed: {e}")
            self.bt_scores = None
    
    def save_state_to_database(self, experiment_id: int):
        """
        Save current ASAP sampling state to the database including BT data.
        
        Args:
            experiment_id: ID of the experiment
        """
        from config import execute_sql
        
        # Prepare TrueSkill state data with proper type conversion
        matrix_json = json.dumps([[int(val) for val in row] for row in self.comparison_matrix.tolist()])
        
        if self.current_scores is not None:
            scores_safe = [float(score) for score in self.current_scores.tolist()]
            scores_json = json.dumps(scores_safe)
        else:
            scores_json = None
            
        spearman_corr = self.get_spearman_correlation()
        
        # Prepare Bradley-Terry state data with proper type conversion
        if self.bt_scores is not None:
            bt_scores_safe = [float(score) for score in self.bt_scores.tolist()]
            bt_scores_json = json.dumps(bt_scores_safe)
        else:
            bt_scores_json = None
        
        # Convert bt_comparisons tuples to native Python types
        if self.bt_comparisons:
            bt_comparisons_safe = [(int(winner), int(loser)) for winner, loser in self.bt_comparisons]
            bt_comparisons_json = json.dumps(bt_comparisons_safe)
        else:
            bt_comparisons_json = None
            
        # Convert convergence history to native Python types  
        if self.bt_convergence_history:
            bt_convergence_safe = [float(change) for change in self.bt_convergence_history]
            bt_convergence_json = json.dumps(bt_convergence_safe)
        else:
            bt_convergence_json = None
            
        bt_ranking_corr = self.get_bradley_terry_ranking_correlation()
        
        # Convert numpy types to native Python types for PostgreSQL compatibility
        if spearman_corr is not None:
            spearman_corr = float(spearman_corr)
        if bt_ranking_corr is not None:
            bt_ranking_corr = float(bt_ranking_corr)
        
        # Check if we need to add BT columns to the table
        try:
            # Try to insert/update with BT data
            execute_sql("""
                INSERT INTO v2_active_sampling_state 
                (experiment_id, comparison_matrix, current_scores, spearman_correlation, iteration_number, 
                 bt_scores, bt_comparisons, bt_convergence_history, bt_ranking_correlation, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT (experiment_id) DO UPDATE SET
                    comparison_matrix = EXCLUDED.comparison_matrix,
                    current_scores = EXCLUDED.current_scores,
                    spearman_correlation = EXCLUDED.spearman_correlation,
                    iteration_number = EXCLUDED.iteration_number,
                    bt_scores = EXCLUDED.bt_scores,
                    bt_comparisons = EXCLUDED.bt_comparisons,
                    bt_convergence_history = EXCLUDED.bt_convergence_history,
                    bt_ranking_correlation = EXCLUDED.bt_ranking_correlation,
                    last_updated = CURRENT_TIMESTAMP
            """, (experiment_id, matrix_json, scores_json, spearman_corr, self.iteration,
                  bt_scores_json, bt_comparisons_json, bt_convergence_json, bt_ranking_corr))
        except Exception as e:
            # Fallback to original schema if BT columns don't exist
            logger.warning(f"BT columns not available, using legacy schema: {e}")
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
    
    def _update_absorbing_class_persistence(self):
        """
        Update tracking of absorbing class persistence for adaptive sampling.
        Called after each comparison to monitor which cases remain absorbing.
        """
        # Get current absorbing classes
        connectivity_info = self.check_graph_connectivity()
        absorbing_info = connectivity_info.get('absorbing_classes', {})
        
        if not absorbing_info.get('has_absorbing_classes', False):
            # No absorbing classes currently - reset persistence tracking
            self.absorbing_class_persistence = {}
            return
        
        # Get current absorbing class members
        current_absorbing_cases = set()
        
        # If we have detailed absorbing class info, use it
        if 'absorbing_class_members' in absorbing_info:
            for ac_members in absorbing_info['absorbing_class_members']:
                current_absorbing_cases.update(ac_members)
        else:
            # Fallback: identify absorbing cases by checking win/loss ratios
            wins = np.sum(self.comparison_matrix, axis=1)
            losses = np.sum(self.comparison_matrix, axis=0)
            
            for i in range(self.n_items):
                total_comps = wins[i] + losses[i]
                if total_comps >= 3 and wins[i] == 0:  # Has comparisons but never wins
                    current_absorbing_cases.add(i)
        
        # Update persistence tracking
        for case_idx in current_absorbing_cases:
            if case_idx not in self.absorbing_class_persistence:
                # First time seeing this case as absorbing
                self.absorbing_class_persistence[case_idx] = {
                    'first_seen': self.iteration,
                    'comparisons': np.sum(self.comparison_matrix[case_idx, :]) + np.sum(self.comparison_matrix[:, case_idx]),
                    'wins': np.sum(self.comparison_matrix[case_idx, :])
                }
            else:
                # Update existing tracking
                self.absorbing_class_persistence[case_idx]['comparisons'] = (
                    np.sum(self.comparison_matrix[case_idx, :]) + np.sum(self.comparison_matrix[:, case_idx])
                )
                self.absorbing_class_persistence[case_idx]['wins'] = np.sum(self.comparison_matrix[case_idx, :])
        
        # Remove cases that are no longer absorbing
        cases_to_remove = []
        for case_idx in self.absorbing_class_persistence:
            if case_idx not in current_absorbing_cases:
                cases_to_remove.append(case_idx)
        
        for case_idx in cases_to_remove:
            del self.absorbing_class_persistence[case_idx]
        
        # Update absorbing classes history for progress tracking
        self.absorbing_classes_history.append({
            'iteration': self.iteration,
            'count': len(current_absorbing_cases),
            'members': list(current_absorbing_cases)
        })
        
        # Check if we should switch to absorbing-targeted mode
        self._check_mode_switch()
    
    def _check_mode_switch(self):
        """
        Check if we should switch between normal and absorbing-targeted sampling modes.
        """
        # Define persistence thresholds
        MIN_COMPARISONS = 8  # Minimum comparisons to consider persistent
        MIN_ITERATIONS_STABLE = 50  # Minimum iterations a case must remain absorbing
        MIN_TOTAL_COMPARISONS = 200  # Minimum total comparisons before switching modes
        
        # Don't switch modes too frequently
        if self.iteration - self.last_mode_switch_iteration < 30:
            return
        
        # Check if we have enough total comparisons
        total_comparisons = np.sum(self.comparison_matrix)
        if total_comparisons < MIN_TOTAL_COMPARISONS:
            return
        
        # Count persistent absorbing classes
        persistent_absorbing = 0
        for case_idx, info in self.absorbing_class_persistence.items():
            iterations_stable = self.iteration - info['first_seen']
            if (info['comparisons'] >= MIN_COMPARISONS and 
                iterations_stable >= MIN_ITERATIONS_STABLE and 
                info['wins'] == 0):
                persistent_absorbing += 1
        
        # Switch to absorbing-targeted mode if we have persistent absorbing classes
        if persistent_absorbing >= 2 and self.sampling_mode == 'normal':
            self.sampling_mode = 'absorbing_targeted'
            self.last_mode_switch_iteration = self.iteration
            logger.info(f"Switched to absorbing-targeted sampling mode at iteration {self.iteration} "
                       f"due to {persistent_absorbing} persistent absorbing classes")
        
        # Switch back to normal mode if absorbing classes are resolved
        elif persistent_absorbing <= 1 and self.sampling_mode == 'absorbing_targeted':
            self.sampling_mode = 'normal'
            self.last_mode_switch_iteration = self.iteration
            logger.info(f"Switched back to normal sampling mode at iteration {self.iteration} "
                       f"- absorbing classes resolved")
    
    def get_persistent_absorbing_classes(self) -> Dict[int, Dict[str, int]]:
        """
        Get information about persistent absorbing classes for diagnostic purposes.
        
        Returns:
            Dictionary mapping case_idx to persistence info
        """
        MIN_COMPARISONS = 8
        MIN_ITERATIONS_STABLE = 50
        
        persistent = {}
        for case_idx, info in self.absorbing_class_persistence.items():
            iterations_stable = self.iteration - info['first_seen']
            if (info['comparisons'] >= MIN_COMPARISONS and 
                iterations_stable >= MIN_ITERATIONS_STABLE and 
                info['wins'] == 0):
                persistent[case_idx] = {
                    'comparisons': info['comparisons'],
                    'wins': info['wins'],
                    'iterations_stable': iterations_stable,
                    'first_seen': info['first_seen']
                }
        
        return persistent


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