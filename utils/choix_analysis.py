"""
choix-based Bradley-Terry Analysis
Replaces custom Bradley-Terry implementation with professional choix library
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

try:
    import choix
    CHOIX_AVAILABLE = True
except ImportError:
    CHOIX_AVAILABLE = False
    logger.error("choix library not available - Bradley-Terry analysis disabled")

def calculate_choix_bradley_terry_statistics(comparison_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive Bradley-Terry statistics using choix library.
    
    Args:
        comparison_df: DataFrame with comparison results
        
    Returns:
        Dictionary containing all Bradley-Terry statistics and analysis
    """
    if not CHOIX_AVAILABLE:
        return {
            'error': 'choix library not available',
            'extraction_ids': [],
            'bradley_terry_scores': {},
            'rankings': {},
            'confidence_intervals': {},
            'reliability_metrics': {}
        }
    
    try:
        # Get all unique extractions (cases)
        all_extractions = list(set(comparison_df['extraction_id_1'].tolist() + comparison_df['extraction_id_2'].tolist()))
        n_items = len(all_extractions)
        
        # DEBUG: Log initial data counts
        print(f"🔍 CHOIX DEBUG: Raw DataFrame has {len(comparison_df)} comparisons")
        print(f"🔍 CHOIX DEBUG: Found {n_items} unique extractions")
        print(f"🔍 CHOIX DEBUG: Extraction IDs range: {min(all_extractions) if all_extractions else 'N/A'} to {max(all_extractions) if all_extractions else 'N/A'}")
        
        # Create extraction ID to index mapping
        id_to_idx = {ext_id: idx for idx, ext_id in enumerate(all_extractions)}
        
        # Convert comparisons to choix format: list of (winner_idx, loser_idx) tuples
        choix_comparisons = []
        skipped_comparisons = 0
        invalid_winners = 0
        
        for idx, row in comparison_df.iterrows():
            id_1, id_2, winner_id = row['extraction_id_1'], row['extraction_id_2'], row['winner_id']
            
            # DEBUG: Check for missing extraction IDs
            if id_1 not in id_to_idx:
                print(f"🔍 CHOIX DEBUG: Missing id_1 in mapping: {id_1}")
                skipped_comparisons += 1
                continue
            if id_2 not in id_to_idx:
                print(f"🔍 CHOIX DEBUG: Missing id_2 in mapping: {id_2}")
                skipped_comparisons += 1
                continue
            
            if winner_id == id_1:
                winner_idx = id_to_idx[id_1]
                loser_idx = id_to_idx[id_2]
                choix_comparisons.append((winner_idx, loser_idx))
            elif winner_id == id_2:
                winner_idx = id_to_idx[id_2]
                loser_idx = id_to_idx[id_1]
                choix_comparisons.append((winner_idx, loser_idx))
            else:
                skipped_comparisons += 1
                invalid_winners += 1
                if skipped_comparisons <= 5:  # Log first few invalid winners
                    print(f"🔍 CHOIX DEBUG: Skipped comparison - winner_id: {winner_id} (type: {type(winner_id)}), id_1: {id_1}, id_2: {id_2}")
                    # Check if winner_id is None, NaN, or other invalid value
                    import pandas as pd
                    if pd.isna(winner_id):
                        print(f"  → Winner is NaN/NULL")
                    elif winner_id is None:
                        print(f"  → Winner is None")
                    else:
                        print(f"  → Winner doesn't match either extraction ID")
        
        # DEBUG: Log conversion results
        print(f"🔍 CHOIX DEBUG: Valid choix comparisons: {len(choix_comparisons)}")
        print(f"🔍 CHOIX DEBUG: Skipped comparisons: {skipped_comparisons}")
        print(f"🔍 CHOIX DEBUG: Invalid winners: {invalid_winners}")
        
        if not choix_comparisons:
            print(f"🔍 CHOIX DEBUG: ERROR - No valid comparisons after conversion!")
            return {
                'error': 'No valid comparisons found',
                'extraction_ids': all_extractions,
                'bradley_terry_scores': {},
                'rankings': {},
                'confidence_intervals': {},
                'reliability_metrics': {}
            }
        
        # Check if comparison graph is connected before running choix
        connectivity_info = check_graph_connectivity(choix_comparisons, n_items, all_extractions)
        
        # DEBUG: Enhanced connectivity logging
        print(f"🔍 CHOIX DEBUG: Connectivity check results:")
        print(f"  - BT Connected: {connectivity_info['is_connected']}")
        print(f"  - Basic Connected: {connectivity_info.get('is_basic_connected', 'unknown')}")
        print(f"  - Components: {connectivity_info['n_components']}")
        print(f"  - Largest component: {connectivity_info['largest_component_size']}")
        print(f"  - Component sizes: {connectivity_info['component_sizes']}")
        
        # Show absorbing class analysis
        absorbing_info = connectivity_info.get('absorbing_classes', {})
        if 'error' not in absorbing_info:
            print(f"🔍 CHOIX DEBUG: Absorbing class analysis:")
            print(f"  - Has absorbing classes: {absorbing_info.get('has_absorbing_classes', 'unknown')}")
            print(f"  - Number of absorbing classes: {absorbing_info.get('n_absorbing_classes', 0)}")
            print(f"  - Strongly connected components: {absorbing_info.get('n_sccs', 0)}")
            
            if absorbing_info.get('has_absorbing_classes'):
                absorbing_classes = absorbing_info.get('absorbing_classes', [])
                print(f"  - Absorbing class sizes: {[len(ac) for ac in absorbing_classes]}")
        else:
            print(f"🔍 CHOIX DEBUG: Absorbing class analysis failed: {absorbing_info['error']}")
        
        logger.info(f"Connectivity check: {connectivity_info['n_components']} components, connected: {connectivity_info['is_connected']}")
        
        if not connectivity_info['is_connected']:
            print(f"🔍 CHOIX DEBUG: Graph DISCONNECTED - using fallback analysis")
            logger.warning(f"Comparison graph is disconnected: {connectivity_info['n_components']} components")
            # Use fallback analysis for disconnected graphs
            return calculate_fallback_bradley_terry(comparison_df, all_extractions, connectivity_info)
        else:
            print(f"🔍 CHOIX DEBUG: Graph CONNECTED - proceeding with choix analysis")
        
        # Calculate Bradley-Terry parameters using choix
        try:
            # Additional validation before calling choix
            if len(choix_comparisons) == 0:
                logger.warning("No valid comparisons for choix analysis")
                return calculate_fallback_bradley_terry(comparison_df, all_extractions, connectivity_info)
            
            # Check if all items appear in at least one comparison
            items_in_comparisons = set()
            for winner_idx, loser_idx in choix_comparisons:
                items_in_comparisons.add(winner_idx)
                items_in_comparisons.add(loser_idx)
            
            if len(items_in_comparisons) < n_items:
                logger.warning(f"Only {len(items_in_comparisons)}/{n_items} items have comparisons - using fallback")
                return calculate_fallback_bradley_terry(comparison_df, all_extractions, connectivity_info)
            
            bt_params = choix.ilsr_pairwise(
                n_items=n_items,
                data=choix_comparisons,
                initial_params=None
            )
        except Exception as e:
            if ("stationary distribution" in str(e) or 
                "absorbing class" in str(e) or 
                "Markov chain" in str(e) or
                "singular" in str(e).lower()):
                logger.warning(f"choix failed due to graph connectivity issues: {e}")
                # Update connectivity info since choix detected disconnection that we missed
                connectivity_info['is_connected'] = False
                connectivity_info['choix_detected_disconnection'] = True
                return calculate_fallback_bradley_terry(comparison_df, all_extractions, connectivity_info)
            else:
                raise e
        
        # Convert to scores dictionary
        bt_scores = {all_extractions[i]: float(bt_params[i]) for i in range(n_items)}
        
        # Calculate rankings (higher scores = better ranks)
        sorted_items = sorted(all_extractions, key=lambda x: bt_scores[x], reverse=True)
        rankings = {ext_id: rank + 1 for rank, ext_id in enumerate(sorted_items)}
        
        # Calculate statistical measures
        confidence_intervals = calculate_choix_confidence_intervals(
            bt_params, choix_comparisons, all_extractions, n_items
        )
        
        reliability_metrics = calculate_choix_reliability_metrics(
            bt_params, choix_comparisons, all_extractions
        )
        
        # Calculate win statistics using choix for consistency
        win_stats = calculate_win_statistics_with_choix(choix_comparisons, all_extractions, n_items)
        
        return {
            'extraction_ids': all_extractions,
            'bradley_terry_scores': bt_scores,
            'rankings': rankings,
            'win_statistics': win_stats,
            'confidence_intervals': confidence_intervals,
            'reliability_metrics': reliability_metrics,
            'choix_comparisons': choix_comparisons,
            'n_comparisons': len(choix_comparisons),
            'convergence_info': {
                'converged': True,
                'method': 'choix_ilsr',
                'n_items': n_items
            }
        }
        
    except Exception as e:
        logger.error(f"choix analysis failed: {e}")
        return {
            'error': f'choix analysis failed: {str(e)}',
            'extraction_ids': [],
            'bradley_terry_scores': {},
            'rankings': {},
            'confidence_intervals': {},
            'reliability_metrics': {}
        }

def calculate_choix_confidence_intervals(
    bt_params: np.ndarray, 
    comparisons: List[Tuple[int, int]], 
    extraction_ids: List[str],
    n_items: int,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Calculate confidence intervals for Bradley-Terry parameters using choix utilities.
    
    Args:
        bt_params: Bradley-Terry parameters from choix
        comparisons: List of (winner, loser) comparison tuples
        extraction_ids: List of extraction IDs
        n_items: Number of items
        confidence_level: Confidence level for intervals
        
    Returns:
        Dictionary with confidence interval information
    """
    try:
        # Use choix approach for calculating standard errors
        if not CHOIX_AVAILABLE:
            return calculate_fallback_confidence_intervals(bt_params, extraction_ids, confidence_level)
        
        # Create win matrix from comparisons using choix utilities approach
        win_matrix = np.zeros((n_items, n_items), dtype=int)
        for winner_idx, loser_idx in comparisons:
            if 0 <= winner_idx < n_items and 0 <= loser_idx < n_items:
                win_matrix[winner_idx, loser_idx] += 1
        
        # Estimate standard errors based on win matrix and number of comparisons
        # This is a simplified approach - choix doesn't provide direct CI calculation
        standard_errors = {}
        confidence_intervals = {}
        
        from scipy import stats
        alpha = 1 - confidence_level
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        for i, ext_id in enumerate(extraction_ids):
            if i < len(bt_params) and i < n_items:
                # Calculate standard error based on number of comparisons involving this item
                total_comps_for_item = np.sum(win_matrix[i, :]) + np.sum(win_matrix[:, i])
                
                # Use Fisher information approximation
                # SE approximately proportional to 1/sqrt(n_comparisons)
                if total_comps_for_item > 0:
                    se = 1.0 / np.sqrt(total_comps_for_item + 1)  # +1 to avoid zero
                else:
                    se = 1.0  # High uncertainty for uncompared items
                
                standard_errors[ext_id] = float(se)
                
                # Calculate confidence interval
                score = float(bt_params[i])
                margin_error = z_critical * se
                
                confidence_intervals[ext_id] = {
                    'lower': score - margin_error,
                    'upper': score + margin_error,
                    'margin_error': margin_error,
                    'standard_error': se,
                    'comparisons_count': int(total_comps_for_item)
                }
            else:
                # Fallback for items beyond parameter array
                standard_errors[ext_id] = 1.0
                confidence_intervals[ext_id] = {
                    'lower': 0.0,
                    'upper': 0.0,
                    'margin_error': 1.0,
                    'standard_error': 1.0,
                    'comparisons_count': 0
                }
        
        return {
            'standard_errors': standard_errors,
            'confidence_intervals': confidence_intervals,
            'confidence_level': confidence_level,
            'method': 'choix_fisher_approximation'
        }
        
    except Exception as e:
        logger.warning(f"choix confidence interval calculation failed: {e}")
        return calculate_fallback_confidence_intervals(bt_params, extraction_ids, confidence_level)

def calculate_fallback_confidence_intervals(bt_params: np.ndarray, extraction_ids: List[str], confidence_level: float) -> Dict[str, Any]:
    """
    Fallback confidence interval calculation.
    
    Args:
        bt_params: Bradley-Terry parameters
        extraction_ids: List of extraction IDs
        confidence_level: Confidence level
        
    Returns:
        Dictionary with fallback confidence intervals
    """
    try:
        from scipy import stats
        alpha = 1 - confidence_level
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        # Use simple fixed standard error as fallback
        fixed_se = 0.2
        
        standard_errors = {ext_id: fixed_se for ext_id in extraction_ids}
        confidence_intervals = {}
        
        for i, ext_id in enumerate(extraction_ids):
            if i < len(bt_params):
                score = float(bt_params[i])
                margin_error = z_critical * fixed_se
                
                confidence_intervals[ext_id] = {
                    'lower': score - margin_error,
                    'upper': score + margin_error,
                    'margin_error': margin_error,
                    'standard_error': fixed_se
                }
            else:
                confidence_intervals[ext_id] = {
                    'lower': 0.0,
                    'upper': 0.0,
                    'margin_error': margin_error,
                    'standard_error': fixed_se
                }
        
        return {
            'standard_errors': standard_errors,
            'confidence_intervals': confidence_intervals,
            'confidence_level': confidence_level,
            'method': 'fallback_fixed_se'
        }
        
    except Exception as e:
        logger.error(f"Fallback confidence intervals failed: {e}")
        return {
            'standard_errors': {},
            'confidence_intervals': {},
            'confidence_level': confidence_level,
            'error': str(e)
        }

def calculate_choix_reliability_metrics(
    bt_params: np.ndarray,
    comparisons: List[Tuple[int, int]],
    extraction_ids: List[str]
) -> Dict[str, Any]:
    """
    Calculate reliability metrics for choix Bradley-Terry model.
    
    Args:
        bt_params: Bradley-Terry parameters
        comparisons: List of comparisons
        extraction_ids: List of extraction IDs
        
    Returns:
        Dictionary with reliability metrics
    """
    try:
        # Calculate basic reliability measures
        score_variance = np.var(bt_params) if len(bt_params) > 1 else 0
        score_range = np.max(bt_params) - np.min(bt_params) if len(bt_params) > 1 else 0
        
        # Estimate separation (simplified)
        # TODO: Implement proper choix-based reliability measures
        separation_estimate = score_range / 2.0 if score_range > 0 else 0
        
        return {
            'score_variance': float(score_variance),
            'score_range': float(score_range),
            'separation_estimate': float(separation_estimate),
            'n_comparisons': len(comparisons),
            'n_items': len(extraction_ids),
            'comparison_density': len(comparisons) / (len(extraction_ids) * (len(extraction_ids) - 1) / 2) if len(extraction_ids) > 1 else 0,
            'method': 'choix_basic'
        }
        
    except Exception as e:
        logger.warning(f"Reliability calculation failed: {e}")
        return {
            'error': str(e),
            'method': 'choix_basic'
        }

def calculate_win_statistics_with_choix(choix_comparisons: List[Tuple[int, int]], extraction_ids: List[str], n_items: int) -> Dict[str, Any]:
    """
    Calculate win/loss statistics using choix utilities for consistency.
    
    Args:
        choix_comparisons: List of (winner_idx, loser_idx) tuples
        extraction_ids: List of extraction IDs  
        n_items: Number of items
        
    Returns:
        Dictionary with win statistics
    """
    try:
        if not CHOIX_AVAILABLE:
            # Fallback to manual calculation if choix not available
            return calculate_win_statistics_manual(choix_comparisons, extraction_ids, n_items)
        
        # Convert comparisons to win matrix using choix utilities
        import choix.utils as choix_utils
        
        # Create win matrix where win_matrix[i,j] = number of times i beat j
        win_matrix = np.zeros((n_items, n_items), dtype=int)
        
        for winner_idx, loser_idx in choix_comparisons:
            if 0 <= winner_idx < n_items and 0 <= loser_idx < n_items:
                win_matrix[winner_idx, loser_idx] += 1
        
        # Calculate statistics from win matrix
        wins = {}
        losses = {}
        total_comparisons = {}
        win_percentages = {}
        
        for i, ext_id in enumerate(extraction_ids):
            if i < n_items:
                # Wins: sum of row i (times this item won)
                wins[ext_id] = int(np.sum(win_matrix[i, :]))
                
                # Losses: sum of column i (times this item lost)
                losses[ext_id] = int(np.sum(win_matrix[:, i]))
                
                # Total comparisons: wins + losses
                total_comparisons[ext_id] = wins[ext_id] + losses[ext_id]
                
                # Win percentage
                if total_comparisons[ext_id] > 0:
                    win_percentages[ext_id] = wins[ext_id] / total_comparisons[ext_id]
                else:
                    win_percentages[ext_id] = 0.0
            else:
                # Safety fallback for items beyond matrix size
                wins[ext_id] = 0
                losses[ext_id] = 0
                total_comparisons[ext_id] = 0
                win_percentages[ext_id] = 0.0
        
        return {
            'wins': wins,
            'losses': losses,
            'total_comparisons': total_comparisons,
            'win_percentages': win_percentages,
            'win_matrix': win_matrix.tolist(),  # Include matrix for potential future use
            'method': 'choix_matrix'
        }
        
    except Exception as e:
        logger.warning(f"choix win statistics calculation failed: {e}, using manual fallback")
        return calculate_win_statistics_manual(choix_comparisons, extraction_ids, n_items)

def calculate_win_statistics_manual(choix_comparisons: List[Tuple[int, int]], extraction_ids: List[str], n_items: int) -> Dict[str, Any]:
    """
    Manual fallback for win statistics calculation.
    
    Args:
        choix_comparisons: List of (winner_idx, loser_idx) tuples
        extraction_ids: List of extraction IDs
        n_items: Number of items
        
    Returns:
        Dictionary with win statistics
    """
    wins = {ext_id: 0 for ext_id in extraction_ids}
    losses = {ext_id: 0 for ext_id in extraction_ids}
    total_comparisons = {ext_id: 0 for ext_id in extraction_ids}
    
    # Count from choix comparison format
    for winner_idx, loser_idx in choix_comparisons:
        if winner_idx < len(extraction_ids) and loser_idx < len(extraction_ids):
            winner_id = extraction_ids[winner_idx]
            loser_id = extraction_ids[loser_idx]
            
            wins[winner_id] += 1
            losses[loser_id] += 1
            total_comparisons[winner_id] += 1
            total_comparisons[loser_id] += 1
    
    # Calculate win percentages
    win_percentages = {
        ext_id: wins[ext_id] / total_comparisons[ext_id] if total_comparisons[ext_id] > 0 else 0 
        for ext_id in extraction_ids
    }
    
    return {
        'wins': wins,
        'losses': losses,
        'total_comparisons': total_comparisons,
        'win_percentages': win_percentages,
        'method': 'manual_fallback'
    }

def check_graph_connectivity(comparisons: List[Tuple[int, int]], n_items: int, extraction_ids: List[str]) -> Dict[str, Any]:
    """
    Check if the comparison graph is connected and has proper Bradley-Terry structure.
    
    Args:
        comparisons: List of (winner, loser) comparison tuples
        n_items: Number of items
        extraction_ids: List of extraction IDs
        
    Returns:
        Dictionary with connectivity information including absorbing class detection
    """
    try:
        # Build adjacency set for undirected graph (basic connectivity)
        undirected_adjacency = {i: set() for i in range(n_items)}
        
        for winner, loser in comparisons:
            undirected_adjacency[winner].add(loser)
            undirected_adjacency[loser].add(winner)
        
        # Find connected components using DFS
        visited = set()
        components = []
        
        def dfs(node, component):
            if node in visited:
                return
            visited.add(node)
            component.append(node)
            for neighbor in undirected_adjacency[node]:
                dfs(neighbor, component)
        
        for i in range(n_items):
            if i not in visited:
                component = []
                dfs(i, component)
                if component:  # Only add non-empty components
                    components.append(component)
        
        # Handle case where some nodes have no connections at all
        if len(visited) < n_items:
            # Add isolated nodes as single-node components
            for i in range(n_items):
                if i not in visited:
                    components.append([i])
        
        # Basic connectivity check
        is_basic_connected = len(components) == 1 and len(components[0]) == n_items
        
        # CRITICAL: Check for absorbing classes in directed graph (Bradley-Terry requirement)
        absorbing_classes_info = check_absorbing_classes(comparisons, n_items)
        
        # A graph is truly "Bradley-Terry connected" only if it's both:
        # 1. Undirected connected (can reach all nodes)
        # 2. Has no absorbing classes (no Markov chain issues)
        is_bt_connected = is_basic_connected and not absorbing_classes_info['has_absorbing_classes']
        
        return {
            'is_connected': is_bt_connected,  # True BT connectivity (what choix needs)
            'is_basic_connected': is_basic_connected,  # Simple undirected connectivity
            'n_components': len(components),
            'components': components,
            'component_sizes': [len(comp) for comp in components],
            'largest_component_size': max([len(comp) for comp in components]) if components else 0,
            'absorbing_classes': absorbing_classes_info
        }
        
    except Exception as e:
        logger.warning(f"Connectivity check failed: {e}")
        return {
            'is_connected': False,
            'is_basic_connected': False,
            'n_components': n_items,  # Assume all isolated
            'components': [[i] for i in range(n_items)],
            'component_sizes': [1] * n_items,
            'largest_component_size': 1,
            'absorbing_classes': {'has_absorbing_classes': True, 'error': str(e)}
        }

def check_absorbing_classes(comparisons: List[Tuple[int, int]], n_items: int) -> Dict[str, Any]:
    """
    Check for absorbing classes in the directed comparison graph.
    Absorbing classes prevent Bradley-Terry from computing valid scores.
    
    Args:
        comparisons: List of (winner, loser) comparison tuples  
        n_items: Number of items
        
    Returns:
        Dictionary with absorbing class information
    """
    try:
        # Build directed adjacency list
        directed_adj = {i: set() for i in range(n_items)}
        
        for winner, loser in comparisons:
            directed_adj[winner].add(loser)
        
        # Find strongly connected components using Tarjan's algorithm
        sccs = find_strongly_connected_components(directed_adj, n_items)
        
        # Check if any SCC forms an absorbing class
        absorbing_classes = []
        
        for scc in sccs:
            # An SCC is absorbing if no node in it has edges to nodes outside it
            is_absorbing = True
            for node in scc:
                for neighbor in directed_adj[node]:
                    if neighbor not in scc:
                        is_absorbing = False
                        break
                if not is_absorbing:
                    break
            
            if is_absorbing and len(scc) > 0:
                # Check if this SCC actually has internal comparisons
                has_internal_comparisons = any(
                    neighbor in scc for node in scc for neighbor in directed_adj[node]
                )
                
                if has_internal_comparisons or len(scc) == 1:
                    absorbing_classes.append(scc)
        
        # Multiple absorbing classes or isolated absorbing nodes cause BT issues
        has_absorbing_classes = len(absorbing_classes) > 1 or (
            len(absorbing_classes) == 1 and len(absorbing_classes[0]) < n_items
        )
        
        return {
            'has_absorbing_classes': has_absorbing_classes,
            'n_absorbing_classes': len(absorbing_classes),
            'absorbing_classes': absorbing_classes,
            'strongly_connected_components': sccs,
            'n_sccs': len(sccs)
        }
        
    except Exception as e:
        logger.warning(f"Absorbing class check failed: {e}")
        return {
            'has_absorbing_classes': True,  # Assume the worst case
            'error': str(e)
        }

def find_strongly_connected_components(adj: Dict[int, set], n_items: int) -> List[List[int]]:
    """
    Find strongly connected components using Tarjan's algorithm.
    
    Args:
        adj: Directed adjacency list
        n_items: Number of items
        
    Returns:
        List of strongly connected components
    """
    index_counter = [0]
    stack = []
    lowlinks = {}
    index = {}
    on_stack = {}
    components = []
    
    def strongconnect(node):
        index[node] = index_counter[0]
        lowlinks[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack[node] = True
        
        for neighbor in adj.get(node, set()):
            if neighbor not in index:
                strongconnect(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif on_stack.get(neighbor, False):
                lowlinks[node] = min(lowlinks[node], index[neighbor])
        
        if lowlinks[node] == index[node]:
            component = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == node:
                    break
            components.append(component)
    
    for node in range(n_items):
        if node not in index:
            strongconnect(node)
    
    return components

def get_user_friendly_warning(connectivity_info: Dict[str, Any], choix_detected: bool, n_components: int) -> str:
    """
    Generate user-friendly warning message based on connectivity issues.
    
    Args:
        connectivity_info: Connectivity analysis results
        choix_detected: Whether choix detected the disconnection
        n_components: Number of components
        
    Returns:
        User-friendly warning message
    """
    if connectivity_info.get('is_basic_connected', False):
        # Graph is connected but has absorbing classes
        absorbing_info = connectivity_info.get('absorbing_classes', {})
        if absorbing_info.get('has_absorbing_classes'):
            n_absorbing = absorbing_info.get('n_absorbing_classes', 0)
            return f"Graph is connected but has {n_absorbing} absorbing classes. Some cases form isolated ranking groups that prevent meaningful comparison with other cases. More comparisons needed to bridge these groups."
        else:
            return "Graph connectivity issue detected. Comparisons may form patterns that prevent reliable Bradley-Terry scoring."
    else:
        # Graph is not even basically connected
        if n_components > 1:
            return f"Comparison graph has {n_components} disconnected components. Some cases have no comparison path to others."
        else:
            return "Graph connectivity issue detected during analysis."

def calculate_fallback_bradley_terry(comparison_df: pd.DataFrame, extraction_ids: List[str], connectivity_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fallback Bradley-Terry analysis for disconnected graphs using choix utilities.
    
    Args:
        comparison_df: DataFrame with comparison results
        extraction_ids: List of extraction IDs
        connectivity_info: Information about graph connectivity
        
    Returns:
        Dictionary with fallback Bradley-Terry analysis
    """
    try:
        n_components = connectivity_info.get('n_components', 0)
        largest_size = connectivity_info.get('largest_component_size', 0)
        choix_detected = connectivity_info.get('choix_detected_disconnection', False)
        
        if choix_detected:
            logger.info(f"Using fallback Bradley-Terry analysis - choix detected disconnection despite initial connectivity check")
        else:
            logger.info(f"Using fallback Bradley-Terry analysis - {n_components} components, largest: {largest_size} items")
        
        # Convert comparisons to choix format for consistency
        n_items = len(extraction_ids)
        id_to_idx = {ext_id: idx for idx, ext_id in enumerate(extraction_ids)}
        
        choix_comparisons = []
        for _, row in comparison_df.iterrows():
            id_1, id_2, winner_id = row['extraction_id_1'], row['extraction_id_2'], row['winner_id']
            
            if winner_id == id_1 and id_1 in id_to_idx and id_2 in id_to_idx:
                winner_idx = id_to_idx[id_1]
                loser_idx = id_to_idx[id_2]
                choix_comparisons.append((winner_idx, loser_idx))
            elif winner_id == id_2 and id_1 in id_to_idx and id_2 in id_to_idx:
                winner_idx = id_to_idx[id_2]
                loser_idx = id_to_idx[id_1]
                choix_comparisons.append((winner_idx, loser_idx))
        
        # Calculate win statistics using choix utilities
        win_stats = calculate_win_statistics_with_choix(choix_comparisons, extraction_ids, n_items)
        
        # Use win percentages as proxy for Bradley-Terry scores
        bt_scores = {}
        for ext_id in extraction_ids:
            win_rate = win_stats['win_percentages'][ext_id]
            total_comps = win_stats['total_comparisons'][ext_id]
            
            # Adjust score based on number of comparisons (more comparisons = more reliable)
            if total_comps > 0:
                # Use logit transformation of win rate with small adjustment to avoid extremes
                adjusted_win_rate = max(0.01, min(0.99, win_rate))
                bt_scores[ext_id] = float(np.log(adjusted_win_rate / (1 - adjusted_win_rate)))
            else:
                bt_scores[ext_id] = 0.0  # Neutral score for uncompared items
        
        # Calculate rankings
        sorted_items = sorted(extraction_ids, key=lambda x: bt_scores[x], reverse=True)
        rankings = {ext_id: rank + 1 for rank, ext_id in enumerate(sorted_items)}
        
        # Enhanced reliability metrics with absorbing class information
        reliability_metrics = {
            'method': 'fallback_win_ratio',
            'total_items': len(extraction_ids)
        }
        
        # Add appropriate warning based on connectivity issue type
        if connectivity_info.get('is_basic_connected', False):
            # Graph is connected but has absorbing classes
            absorbing_info = connectivity_info.get('absorbing_classes', {})
            if absorbing_info.get('has_absorbing_classes'):
                n_absorbing = absorbing_info.get('n_absorbing_classes', 0)
                reliability_metrics['connectivity_warning'] = f"Graph has {n_absorbing} absorbing classes preventing Bradley-Terry convergence"
                reliability_metrics['absorbing_classes'] = n_absorbing
            else:
                reliability_metrics['connectivity_warning'] = "Graph connectivity issue detected by choix"
        else:
            # Graph is not even basically connected
            reliability_metrics['connectivity_warning'] = f"Graph has {connectivity_info['n_components']} disconnected components"
            reliability_metrics['largest_component_size'] = connectivity_info['largest_component_size']
        
        return {
            'extraction_ids': extraction_ids,
            'bradley_terry_scores': bt_scores,
            'rankings': rankings,
            'win_statistics': win_stats,
            'confidence_intervals': {
                'standard_errors': {ext_id: 0.1 for ext_id in extraction_ids},
                'confidence_intervals': {ext_id: {'lower': bt_scores[ext_id] - 0.1, 'upper': bt_scores[ext_id] + 0.1, 'margin_error': 0.1} for ext_id in extraction_ids},
                'method': 'fallback_approximation'
            },
            'reliability_metrics': reliability_metrics,
            'n_comparisons': len(comparison_df),
            'convergence_info': {
                'converged': False,
                'method': 'fallback_disconnected',
                'n_items': len(extraction_ids),
                'connectivity_warning': True
            },
            'warning': get_user_friendly_warning(connectivity_info, choix_detected, n_components)
        }
        
    except Exception as e:
        logger.error(f"Fallback Bradley-Terry analysis failed: {e}")
        return {
            'error': f'Fallback analysis failed: {str(e)}',
            'extraction_ids': extraction_ids,
            'bradley_terry_scores': {ext_id: 0 for ext_id in extraction_ids},
            'rankings': {ext_id: i+1 for i, ext_id in enumerate(extraction_ids)},
            'confidence_intervals': {},
            'reliability_metrics': {}
        }

# Backward compatibility metrics removed - using native choix metrics only

def calculate_transitivity_score(comparison_df: pd.DataFrame) -> float:
    """
    Calculate transitivity score (higher = more consistent).
    
    Args:
        comparison_df: DataFrame with comparison results
        
    Returns:
        Transitivity score between 0 and 1
    """
    try:
        # Build win relationships
        wins = {}  # wins[A][B] = True if A beats B
        
        for _, row in comparison_df.iterrows():
            id_1, id_2, winner_id = row['extraction_id_1'], row['extraction_id_2'], row['winner_id']
            
            if winner_id == id_1:
                if id_1 not in wins:
                    wins[id_1] = {}
                wins[id_1][id_2] = True
            elif winner_id == id_2:
                if id_2 not in wins:
                    wins[id_2] = {}
                wins[id_2][id_1] = True
        
        # Get all extractions
        all_extractions = list(set(comparison_df['extraction_id_1'].tolist() + comparison_df['extraction_id_2'].tolist()))
        
        if len(all_extractions) < 3:
            return 1.0  # Perfect transitivity with fewer than 3 items
        
        # Check for transitivity violations
        violations = 0
        total_triples = 0
        
        for i, a in enumerate(all_extractions):
            for j, b in enumerate(all_extractions[i+1:], i+1):
                for k, c in enumerate(all_extractions[j+1:], j+1):
                    # Check triple (a, b, c)
                    total_triples += 1
                    
                    # Check if we have a > b > c > a (violation)
                    a_beats_b = a in wins and b in wins[a]
                    b_beats_c = b in wins and c in wins[b]
                    c_beats_a = c in wins and a in wins[c]
                    
                    if a_beats_b and b_beats_c and c_beats_a:
                        violations += 1
        
        # Return transitivity score (1 - violation_rate)
        if total_triples == 0:
            return 1.0
        
        return 1.0 - (violations / total_triples)
        
    except Exception as e:
        logger.warning(f"Transitivity calculation failed: {e}")
        return 0.5  # Neutral score on error

def get_choix_rankings_dataframe(bt_statistics: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Convert choix Bradley-Terry results to a rankings DataFrame.
    
    Args:
        bt_statistics: Results from calculate_choix_bradley_terry_statistics
        
    Returns:
        DataFrame with rankings and statistics
    """
    if 'error' in bt_statistics or not bt_statistics.get('bradley_terry_scores'):
        return None
    
    try:
        # Create DataFrame from results
        data = []
        for ext_id in bt_statistics['extraction_ids']:
            row = {
                'extraction_id': ext_id,
                'bradley_terry_score': bt_statistics['bradley_terry_scores'].get(ext_id, 0),
                'rank': bt_statistics['rankings'].get(ext_id, 0),
                'wins': bt_statistics['win_statistics']['wins'].get(ext_id, 0),
                'losses': bt_statistics['win_statistics']['losses'].get(ext_id, 0),
                'total_comparisons': bt_statistics['win_statistics']['total_comparisons'].get(ext_id, 0),
                'win_percentage': bt_statistics['win_statistics']['win_percentages'].get(ext_id, 0)
            }
            
            # Add confidence interval data if available
            ci_data = bt_statistics.get('confidence_intervals', {}).get('confidence_intervals', {}).get(ext_id, {})
            row['ci_lower'] = ci_data.get('lower', 0)
            row['ci_upper'] = ci_data.get('upper', 0)
            row['margin_error'] = ci_data.get('margin_error', 0)
            row['standard_error'] = ci_data.get('standard_error', 0)
            
            data.append(row)
        
        df = pd.DataFrame(data)
        return df.sort_values('rank')
        
    except Exception as e:
        logger.error(f"Failed to create rankings DataFrame: {e}")
        return None