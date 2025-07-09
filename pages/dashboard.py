"""
Experiment Management Dashboard
Meta-level interface for creating, managing, and comparing experiments
"""

import streamlit as st
from datetime import datetime, timedelta
from config import GEMINI_MODELS, execute_sql, get_database_connection, get_gemini_model
import json
import os
import time

# Lazy imports for performance
def _get_pandas():
    import pandas as pd
    return pd

def _get_numpy():
    import numpy as np
    return np
from utils.case_management import (
    get_database_counts, load_data_from_parquet, clear_database, clear_selected_cases,
    get_case_summary, get_available_cases, filter_cases_by_criteria,
    get_experiment_selected_cases, get_available_cases_for_selection,
    add_cases_to_experiments, remove_cases_from_experiments,
    calculate_bradley_terry_comparisons, generate_bradley_terry_structure,
    get_bradley_terry_structure, get_block_summary, generate_bradley_terry_comparison_pairs
)

# Import experiment execution module
from pages import experiment_execution

def show_bradley_terry_analysis(experiment_id, n_cases, total_tests, total_comparisons):
    """Display comprehensive Bradley-Terry analysis and results"""
    
    # Import analysis libraries
    pd = _get_pandas()
    np = _get_numpy()
    
    try:
        # Get comprehensive comparison data with case information
        comparison_data = execute_sql("""
            SELECT 
                ec.comparison_id,
                ec.extraction_id_1,
                ec.extraction_id_2,
                ec.winner_id,
                ec.comparison_rationale,
                ec.confidence_score,
                ec.comparison_date,
                c1.case_name as case_a_name,
                c1.citation as case_a_citation,
                c1.decision_year as case_a_year,
                c2.case_name as case_b_name,
                c2.citation as case_b_citation,
                c2.decision_year as case_b_year,
                ee1.legal_test_content as test_a_content,
                ee2.legal_test_content as test_b_content,
                ee1.test_novelty as test_a_novelty,
                ee2.test_novelty as test_b_novelty
            FROM v2_experiment_comparisons ec
            JOIN v2_experiment_extractions ee1 ON ec.extraction_id_1 = ee1.extraction_id
            JOIN v2_experiment_extractions ee2 ON ec.extraction_id_2 = ee2.extraction_id
            JOIN v2_cases c1 ON ee1.case_id = c1.case_id
            JOIN v2_cases c2 ON ee2.case_id = c2.case_id
            WHERE ec.experiment_id = ?
            ORDER BY ec.comparison_date
        """, (experiment_id,), fetch=True)
        
        if not comparison_data:
            st.error("No comparison data found for analysis.")
            return
        
        # Convert to DataFrame for analysis
        comparison_df = pd.DataFrame(comparison_data, columns=[
            'comparison_id', 'extraction_id_1', 'extraction_id_2', 'winner_id', 'comparison_rationale',
            'confidence_score', 'comparison_date', 'case_a_name', 'case_a_citation', 'case_a_year',
            'case_b_name', 'case_b_citation', 'case_b_year', 'test_a_content', 'test_b_content',
            'test_a_novelty', 'test_b_novelty'
        ])
        
        # Calculate Bradley-Terry statistics
        bradley_terry_stats = calculate_bradley_terry_statistics(comparison_df)
        
        # Show analysis in organized tabs
        analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4, analysis_tab5 = st.tabs([
            "📊 Overview", "🏆 Rankings", "📈 Temporal Analysis", "🔍 Reliability", "📈 Statistical Reliability"
        ])
        
        with analysis_tab1:
            show_overview_statistics(comparison_df, bradley_terry_stats, n_cases, total_tests, total_comparisons)
        
        with analysis_tab2:
            show_bradley_terry_rankings(comparison_df, bradley_terry_stats)
        
        with analysis_tab3:
            show_temporal_analysis(comparison_df, bradley_terry_stats)
        
        with analysis_tab4:
            show_reliability_analysis(comparison_df, bradley_terry_stats)
        
        with analysis_tab5:
            show_advanced_reliability_metrics(comparison_df, bradley_terry_stats)
            
    except Exception as e:
        st.error(f"Error in Bradley-Terry analysis: {str(e)}")
        st.write("Debug info:", str(e))

def calculate_bradley_terry_statistics(comparison_df):
    """Calculate comprehensive Bradley-Terry statistics"""
    pd = _get_pandas()
    np = _get_numpy()
    
    # Get all unique extractions (cases)
    all_extractions = set(comparison_df['extraction_id_1'].tolist() + comparison_df['extraction_id_2'].tolist())
    
    # Initialize win-loss matrix
    wins = {ext_id: 0 for ext_id in all_extractions}
    losses = {ext_id: 0 for ext_id in all_extractions}
    total_comparisons = {ext_id: 0 for ext_id in all_extractions}
    
    # Count wins and losses
    for _, row in comparison_df.iterrows():
        id_1, id_2, winner_id = row['extraction_id_1'], row['extraction_id_2'], row['winner_id']
        
        total_comparisons[id_1] += 1
        total_comparisons[id_2] += 1
        
        if winner_id == id_1:
            wins[id_1] += 1
            losses[id_2] += 1
        elif winner_id == id_2:
            wins[id_2] += 1
            losses[id_1] += 1
        # If no winner, neither gets a win
    
    # Calculate Bradley-Terry scores using iterative method
    scores, comparisons = calculate_bradley_terry_scores(comparison_df, all_extractions)
    
    # Calculate win percentages
    win_percentages = {ext_id: wins[ext_id] / total_comparisons[ext_id] if total_comparisons[ext_id] > 0 else 0 
                      for ext_id in all_extractions}
    
    # Calculate reliability metrics
    reliability_metrics = calculate_reliability_metrics(comparison_df)
    
    # Calculate Fisher Information Matrix and confidence intervals
    try:
        fisher_matrix = calculate_fisher_information_matrix(scores, comparisons, list(all_extractions))
        ci_results = calculate_confidence_intervals(scores, fisher_matrix, list(all_extractions))
        
        # Calculate pairwise significance tests
        significance_results = calculate_pairwise_significance(
            scores, ci_results.get('covariance_matrix'), list(all_extractions)
        )
        
        # Calculate item separation reliability
        separation_reliability = calculate_item_separation_reliability(
            scores, ci_results.get('standard_errors', {})
        )
        
        # Calculate overdispersion and model diagnostics
        model_diagnostics = calculate_overdispersion_tests(comparison_df, scores, comparisons)
        
    except Exception as e:
        ci_results = {
            'standard_errors': {ext_id: 0 for ext_id in all_extractions},
            'confidence_intervals': {ext_id: {'lower': scores[ext_id], 'upper': scores[ext_id], 'margin_error': 0} for ext_id in all_extractions},
            'covariance_matrix': None,
            'confidence_level': 0.95,
            'error': str(e)
        }
        significance_results = {
            'significance_matrix': {},
            'z_statistics': {},
            'p_values': {},
            'significant_pairs': [],
            'warning': f'Error calculating significance: {str(e)}'
        }
        separation_reliability = {
            'separation_coefficient': 0,
            'reliability': 0,
            'estimated_strata': 1,
            'interpretation': 'Poor separation',
            'error': str(e)
        }
        model_diagnostics = {
            'deviance': 0,
            'pearson_chi2': 0,
            'overdispersion_ratio': 1,
            'degrees_freedom': 0,
            'interpretation': 'Cannot calculate',
            'error': str(e)
        }
        fisher_matrix = None
    
    return {
        'extraction_ids': list(all_extractions),
        'wins': wins,
        'losses': losses,
        'total_comparisons': total_comparisons,
        'win_percentages': win_percentages,
        'bradley_terry_scores': scores,
        'reliability_metrics': reliability_metrics,
        'confidence_intervals': ci_results,
        'significance_tests': significance_results,
        'separation_reliability': separation_reliability,
        'model_diagnostics': model_diagnostics,
        'fisher_matrix': fisher_matrix
    }

def calculate_bradley_terry_scores(comparison_df, all_extractions, max_iterations=1000, tolerance=1e-6):
    """Calculate Bradley-Terry scores using iterative method with enhanced statistical output"""
    pd = _get_pandas()
    np = _get_numpy()
    
    # Initialize scores (start with equal probabilities)
    scores = {ext_id: 1.0 for ext_id in all_extractions}
    
    # Create comparison matrix
    comparisons = {}
    for _, row in comparison_df.iterrows():
        id_1, id_2, winner_id = row['extraction_id_1'], row['extraction_id_2'], row['winner_id']
        
        key = tuple(sorted([id_1, id_2]))
        if key not in comparisons:
            comparisons[key] = {'total': 0, 'wins': {id_1: 0, id_2: 0}}
        
        comparisons[key]['total'] += 1
        if winner_id == id_1:
            comparisons[key]['wins'][id_1] += 1
        elif winner_id == id_2:
            comparisons[key]['wins'][id_2] += 1
    
    # Iterative estimation
    for iteration in range(max_iterations):
        old_scores = scores.copy()
        
        # Update each score
        for ext_id in all_extractions:
            numerator = 0
            denominator = 0
            
            for (id_1, id_2), comp_data in comparisons.items():
                if ext_id in [id_1, id_2]:
                    other_id = id_2 if ext_id == id_1 else id_1
                    
                    wins_against_other = comp_data['wins'][ext_id]
                    total_against_other = comp_data['total']
                    
                    numerator += wins_against_other
                    denominator += total_against_other / (scores[ext_id] + scores[other_id])
            
            if denominator > 0:
                scores[ext_id] = numerator / denominator
        
        # Check for convergence
        max_change = max(abs(scores[ext_id] - old_scores[ext_id]) for ext_id in all_extractions)
        if max_change < tolerance:
            break
    
    # Normalize scores so they sum to number of items
    total_score = sum(scores.values())
    if total_score > 0:
        n_items = len(all_extractions)
        scores = {ext_id: score * n_items / total_score for ext_id, score in scores.items()}
    
    return scores, comparisons

def calculate_fisher_information_matrix(scores, comparisons, all_extractions):
    """Calculate Fisher Information Matrix for Bradley-Terry model"""
    np = _get_numpy()
    
    n_items = len(all_extractions)
    ext_to_idx = {ext_id: i for i, ext_id in enumerate(all_extractions)}
    
    # Initialize Fisher Information Matrix
    fisher_matrix = np.zeros((n_items, n_items))
    
    # Calculate second derivatives of log-likelihood
    for (id_1, id_2), comp_data in comparisons.items():
        if id_1 in scores and id_2 in scores:
            i = ext_to_idx[id_1]
            j = ext_to_idx[id_2]
            
            pi = scores[id_1]
            pj = scores[id_2]
            nij = comp_data['total']
            
            # Second derivative elements for Bradley-Terry log-likelihood
            # For pairs (i,j): d²L/dβi² = -nij * pi * pj / (pi + pj)²
            denominator = (pi + pj) ** 2
            if denominator > 0:
                fisher_val = nij * pi * pj / denominator
                
                # Diagonal elements (negative second derivatives)
                fisher_matrix[i, i] += fisher_val
                fisher_matrix[j, j] += fisher_val
                
                # Off-diagonal elements
                fisher_matrix[i, j] -= fisher_val
                fisher_matrix[j, i] -= fisher_val
    
    return fisher_matrix

def calculate_confidence_intervals(scores, fisher_matrix, all_extractions, confidence_level=0.95):
    """Calculate confidence intervals for Bradley-Terry scores"""
    np = _get_numpy()
    from scipy import stats
    
    try:
        # Calculate covariance matrix (inverse of Fisher Information Matrix)
        covariance_matrix = np.linalg.inv(fisher_matrix)
        
        # Extract standard errors (sqrt of diagonal elements)
        standard_errors = {}
        confidence_intervals = {}
        
        # Get critical value for confidence level
        alpha = 1 - confidence_level
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        for i, ext_id in enumerate(all_extractions):
            if i < len(covariance_matrix):
                # Standard error
                variance = covariance_matrix[i, i]
                if variance > 0:
                    se = np.sqrt(variance)
                    standard_errors[ext_id] = se
                    
                    # Confidence interval
                    score = scores[ext_id]
                    margin_error = z_critical * se
                    confidence_intervals[ext_id] = {
                        'lower': score - margin_error,
                        'upper': score + margin_error,
                        'margin_error': margin_error
                    }
                else:
                    standard_errors[ext_id] = 0
                    confidence_intervals[ext_id] = {
                        'lower': scores[ext_id],
                        'upper': scores[ext_id], 
                        'margin_error': 0
                    }
        
        return {
            'standard_errors': standard_errors,
            'confidence_intervals': confidence_intervals,
            'covariance_matrix': covariance_matrix,
            'confidence_level': confidence_level
        }
        
    except np.linalg.LinAlgError:
        # Fallback if matrix is singular
        standard_errors = {ext_id: 0 for ext_id in all_extractions}
        confidence_intervals = {ext_id: {
            'lower': scores[ext_id], 'upper': scores[ext_id], 'margin_error': 0
        } for ext_id in all_extractions}
        
        return {
            'standard_errors': standard_errors,
            'confidence_intervals': confidence_intervals,
            'covariance_matrix': None,
            'confidence_level': confidence_level,
            'warning': 'Singular matrix - unable to calculate reliable confidence intervals'
        }

def calculate_pairwise_significance(scores, covariance_matrix, all_extractions, alpha=0.05):
    """Calculate statistical significance of pairwise differences using Wald tests"""
    np = _get_numpy()
    from scipy import stats
    
    if covariance_matrix is None:
        return {
            'significance_matrix': {},
            'z_statistics': {},
            'p_values': {},
            'significant_pairs': [],
            'warning': 'No covariance matrix available - cannot calculate significance'
        }
    
    ext_to_idx = {ext_id: i for i, ext_id in enumerate(all_extractions)}
    n_items = len(all_extractions)
    
    # Initialize result dictionaries
    significance_matrix = {}
    z_statistics = {}
    p_values = {}
    significant_pairs = []
    
    # Calculate pairwise differences and their significance
    for i, ext_id_1 in enumerate(all_extractions):
        significance_matrix[ext_id_1] = {}
        z_statistics[ext_id_1] = {}
        p_values[ext_id_1] = {}
        
        for j, ext_id_2 in enumerate(all_extractions):
            if i != j:
                # Difference in scores
                score_diff = scores[ext_id_1] - scores[ext_id_2]
                
                # Standard error of the difference: SE(diff) = sqrt(Var(i) + Var(j) - 2*Cov(i,j))
                var_i = covariance_matrix[i, i] if i < covariance_matrix.shape[0] else 0
                var_j = covariance_matrix[j, j] if j < covariance_matrix.shape[1] else 0
                cov_ij = covariance_matrix[i, j] if i < covariance_matrix.shape[0] and j < covariance_matrix.shape[1] else 0
                
                se_diff = np.sqrt(var_i + var_j - 2 * cov_ij)
                
                if se_diff > 0:
                    # Z-statistic for Wald test
                    z_stat = score_diff / se_diff
                    
                    # P-value (two-tailed test)
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                    
                    # Store results
                    z_statistics[ext_id_1][ext_id_2] = z_stat
                    p_values[ext_id_1][ext_id_2] = p_value
                    significance_matrix[ext_id_1][ext_id_2] = p_value < alpha
                    
                    # Track significant pairs
                    if p_value < alpha and score_diff > 0:  # Only count when first item significantly better
                        significant_pairs.append({
                            'item_1': ext_id_1,
                            'item_2': ext_id_2,
                            'score_diff': score_diff,
                            'z_statistic': z_stat,
                            'p_value': p_value
                        })
                else:
                    z_statistics[ext_id_1][ext_id_2] = 0
                    p_values[ext_id_1][ext_id_2] = 1.0
                    significance_matrix[ext_id_1][ext_id_2] = False
            else:
                # Same item comparison
                z_statistics[ext_id_1][ext_id_2] = 0
                p_values[ext_id_1][ext_id_2] = 1.0
                significance_matrix[ext_id_1][ext_id_2] = False
    
    # Apply Bonferroni correction for multiple comparisons
    n_comparisons = n_items * (n_items - 1)  # Total pairwise comparisons
    bonferroni_alpha = alpha / n_comparisons if n_comparisons > 0 else alpha
    
    bonferroni_significant = []
    for pair in significant_pairs:
        if pair['p_value'] < bonferroni_alpha:
            bonferroni_significant.append(pair)
    
    return {
        'significance_matrix': significance_matrix,
        'z_statistics': z_statistics,
        'p_values': p_values,
        'significant_pairs': significant_pairs,
        'bonferroni_significant': bonferroni_significant,
        'alpha': alpha,
        'bonferroni_alpha': bonferroni_alpha,
        'n_comparisons': n_comparisons
    }

def calculate_item_separation_reliability(scores, standard_errors):
    """Calculate item separation reliability index"""
    np = _get_numpy()
    
    if not scores or not standard_errors:
        return {
            'separation_coefficient': 0,
            'reliability': 0,
            'estimated_strata': 1,
            'interpretation': 'Poor separation',
            'warning': 'Insufficient data for calculation'
        }
    
    # Calculate true score variance (variance of Bradley-Terry scores)
    score_values = list(scores.values())
    true_variance = np.var(score_values, ddof=1) if len(score_values) > 1 else 0
    
    # Calculate error variance (mean of squared standard errors)
    se_values = [se for se in standard_errors.values() if se > 0]
    error_variance = np.mean([se**2 for se in se_values]) if se_values else 0
    
    if error_variance <= 0 or true_variance <= 0:
        return {
            'separation_coefficient': 0,
            'reliability': 0,
            'estimated_strata': 1,
            'interpretation': 'Poor separation',
            'warning': 'Unable to calculate - insufficient variance'
        }
    
    # Calculate separation coefficient: ratio of true SD to error SD
    true_sd = np.sqrt(true_variance)
    error_sd = np.sqrt(error_variance)
    separation_coefficient = true_sd / error_sd
    
    # Calculate reliability: Separation² / (1 + Separation²)
    reliability = (separation_coefficient ** 2) / (1 + separation_coefficient ** 2)
    
    # Estimate number of statistically distinguishable strata
    # Rule of thumb: (4 * Separation + 1) / 3
    estimated_strata = max(1, int((4 * separation_coefficient + 1) / 3))
    
    # Interpretation
    if reliability > 0.9:
        interpretation = 'Excellent separation'
    elif reliability > 0.8:
        interpretation = 'Good separation'
    elif reliability > 0.7:
        interpretation = 'Acceptable separation'
    else:
        interpretation = 'Poor separation'
    
    return {
        'separation_coefficient': separation_coefficient,
        'reliability': reliability,
        'estimated_strata': estimated_strata,
        'interpretation': interpretation,
        'true_variance': true_variance,
        'error_variance': error_variance,
        'n_items': len(scores)
    }

def calculate_overdispersion_tests(comparison_df, scores, comparisons):
    """Calculate overdispersion and model diagnostic tests"""
    np = _get_numpy()
    from scipy import stats
    
    if not scores or not comparisons:
        return {
            'deviance': 0,
            'pearson_chi2': 0,
            'overdispersion_ratio': 1,
            'degrees_freedom': 0,
            'deviance_p_value': 1,
            'pearson_p_value': 1,
            'interpretation': 'Cannot calculate - insufficient data'
        }
    
    # Calculate residuals and fit statistics
    total_deviance = 0
    total_pearson_chi2 = 0
    n_comparisons = 0
    
    for (id_1, id_2), comp_data in comparisons.items():
        if id_1 in scores and id_2 in scores:
            pi = scores[id_1]
            pj = scores[id_2]
            
            # Expected probability that i beats j
            expected_prob = pi / (pi + pj)
            
            # Observed data
            wins_i = comp_data['wins'][id_1]
            total_games = comp_data['total']
            observed_prob = wins_i / total_games if total_games > 0 else 0
            
            # Deviance contribution
            if observed_prob > 0 and observed_prob < 1:
                deviance_contrib = 2 * total_games * (
                    observed_prob * np.log(observed_prob / expected_prob) +
                    (1 - observed_prob) * np.log((1 - observed_prob) / (1 - expected_prob))
                )
                total_deviance += deviance_contrib
            
            # Pearson chi-square contribution
            expected_wins = expected_prob * total_games
            if expected_wins > 0 and expected_wins < total_games:
                pearson_contrib = ((wins_i - expected_wins) ** 2) / (expected_wins * (1 - expected_prob))
                total_pearson_chi2 += pearson_contrib
            
            n_comparisons += 1
    
    # Degrees of freedom = number of comparisons - number of parameters
    # For Bradley-Terry: n_items - 1 parameters (one is fixed as reference)
    n_parameters = len(scores) - 1
    degrees_freedom = max(1, n_comparisons - n_parameters)
    
    # Overdispersion ratio (should be approximately 1 if model fits well)
    overdispersion_ratio = total_deviance / degrees_freedom if degrees_freedom > 0 else 1
    
    # P-values for goodness-of-fit tests
    deviance_p_value = 1 - stats.chi2.cdf(total_deviance, degrees_freedom) if degrees_freedom > 0 else 1
    pearson_p_value = 1 - stats.chi2.cdf(total_pearson_chi2, degrees_freedom) if degrees_freedom > 0 else 1
    
    # Interpretation
    if overdispersion_ratio > 2:
        interpretation = 'Significant overdispersion detected'
    elif overdispersion_ratio > 1.5:
        interpretation = 'Moderate overdispersion'
    elif overdispersion_ratio < 0.5:
        interpretation = 'Possible underdispersion'
    else:
        interpretation = 'Dispersion within expected range'
    
    return {
        'deviance': total_deviance,
        'pearson_chi2': total_pearson_chi2,
        'overdispersion_ratio': overdispersion_ratio,
        'degrees_freedom': degrees_freedom,
        'deviance_p_value': deviance_p_value,
        'pearson_p_value': pearson_p_value,
        'n_comparisons': n_comparisons,
        'n_parameters': n_parameters,
        'interpretation': interpretation
    }

def calculate_reliability_metrics(comparison_df):
    """Calculate reliability and consistency metrics"""
    pd = _get_pandas()
    np = _get_numpy()
    
    # Create a copy to avoid modifying the original
    comparison_df = comparison_df.copy()
    
    total_comparisons = len(comparison_df)
    decisive_comparisons = len(comparison_df[comparison_df['winner_id'].notna()])
    
    # Calculate average confidence score
    avg_confidence = comparison_df['confidence_score'].mean() if 'confidence_score' in comparison_df.columns else 0
    
    # Calculate temporal consistency (only meaningful for extended comparison periods)
    temporal_consistency = None
    temporal_consistency_note = "N/A (batch processing)"
    
    if 'comparison_date' in comparison_df.columns and len(comparison_df) > 1:
        # Convert dates and check time span
        comparison_df['comparison_date'] = pd.to_datetime(comparison_df['comparison_date'])
        time_span = (comparison_df['comparison_date'].max() - comparison_df['comparison_date'].min()).days
        
        # Only calculate if comparisons span multiple days (meaningful temporal variation)
        if time_span > 1:
            df_sorted = comparison_df.sort_values('comparison_date')
            
            # Split comparisons into early and late periods
            midpoint = len(df_sorted) // 2
            early_comparisons = df_sorted.iloc[:midpoint]
            late_comparisons = df_sorted.iloc[midpoint:]
            
            # Calculate win rates for each period
            def get_win_rates(df_subset):
                win_rates = {}
                for _, row in df_subset.iterrows():
                    id_1, id_2, winner_id = row['extraction_id_1'], row['extraction_id_2'], row['winner_id']
                    
                    for ext_id in [id_1, id_2]:
                        if ext_id not in win_rates:
                            win_rates[ext_id] = {'wins': 0, 'total': 0}
                        win_rates[ext_id]['total'] += 1
                        if winner_id == ext_id:
                            win_rates[ext_id]['wins'] += 1
                
                # Convert to percentages
                for ext_id in win_rates:
                    if win_rates[ext_id]['total'] > 0:
                        win_rates[ext_id] = win_rates[ext_id]['wins'] / win_rates[ext_id]['total']
                    else:
                        win_rates[ext_id] = 0
                
                return win_rates
            
            early_rates = get_win_rates(early_comparisons)
            late_rates = get_win_rates(late_comparisons)
            
            # Calculate consistency (how similar the win rates are between periods)
            common_extractions = set(early_rates.keys()) & set(late_rates.keys())
            if common_extractions:
                differences = [abs(early_rates[ext_id] - late_rates[ext_id]) for ext_id in common_extractions]
                avg_difference = sum(differences) / len(differences)
                temporal_consistency = 1.0 - avg_difference  # Higher = more consistent
                temporal_consistency_note = f"Based on {time_span} day span"
            else:
                temporal_consistency = 0.5
                temporal_consistency_note = "Insufficient overlap"
        else:
            temporal_consistency_note = "All comparisons same day"
    
    # Calculate transitivity violations (A beats B, B beats C, C beats A)
    transitivity_score = calculate_transitivity_score(comparison_df)
    
    return {
        'total_comparisons': total_comparisons,
        'decisive_comparisons': decisive_comparisons,
        'decisiveness_rate': decisive_comparisons / total_comparisons if total_comparisons > 0 else 0,
        'average_confidence': avg_confidence,
        'temporal_consistency': temporal_consistency,
        'temporal_consistency_note': temporal_consistency_note,
        'transitivity_score': transitivity_score
    }

def calculate_transitivity_score(comparison_df):
    """Calculate transitivity score (higher = more consistent)"""
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
    all_extractions = set(comparison_df['extraction_id_1'].tolist() + comparison_df['extraction_id_2'].tolist())
    
    # Check transitivity violations
    total_triplets = 0
    violations = 0
    
    for a in all_extractions:
        for b in all_extractions:
            for c in all_extractions:
                if a != b and b != c and a != c:
                    # Check if we have A > B and B > C
                    a_beats_b = a in wins and b in wins[a]
                    b_beats_c = b in wins and c in wins[b]
                    
                    if a_beats_b and b_beats_c:
                        total_triplets += 1
                        # Check if A > C (should be true for transitivity)
                        a_beats_c = a in wins and c in wins[a]
                        c_beats_a = c in wins and a in wins[c]
                        
                        # Violation if C beats A instead of A beating C
                        if c_beats_a and not a_beats_c:
                            violations += 1
    
    # Calculate score (0 to 1, where 1 = perfect transitivity)
    if total_triplets == 0:
        return 1.0  # No triplets to check = perfect by default
    
    return 1.0 - (violations / total_triplets)

def show_overview_statistics(comparison_df, bradley_terry_stats, n_cases, total_tests, total_comparisons):
    """Show overview statistics"""
    pd = _get_pandas()
    
    st.subheader("📊 Experiment Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Cases Analyzed", n_cases)
        st.metric("Tests Extracted", total_tests)
    
    with col2:
        st.metric("Total Comparisons", total_comparisons)
        decisive_rate = bradley_terry_stats['reliability_metrics']['decisiveness_rate']
        st.metric("Decisive Comparisons", f"{decisive_rate:.1%}")
    
    with col3:
        avg_confidence = bradley_terry_stats['reliability_metrics']['average_confidence']
        st.metric("Avg Confidence", f"{avg_confidence:.2f}" if avg_confidence > 0 else "N/A")
    
    with col4:
        transitivity = bradley_terry_stats['reliability_metrics']['transitivity_score']
        st.metric("Transitivity Score", f"{transitivity:.2f}")
    
    # Distribution of test novelty
    st.subheader("📈 Test Novelty Distribution")
    
    # Count novelty categories
    novelty_counts = {}
    for _, row in comparison_df.iterrows():
        for novelty in [row['test_a_novelty'], row['test_b_novelty']]:
            if novelty:
                novelty_counts[novelty] = novelty_counts.get(novelty, 0) + 1
    
    if novelty_counts:
        novelty_df = pd.DataFrame(list(novelty_counts.items()), columns=['Novelty Type', 'Count'])
        st.bar_chart(novelty_df.set_index('Novelty Type'))
    else:
        st.info("No test novelty data available")

def show_bradley_terry_rankings(comparison_df, bradley_terry_stats):
    """Show Bradley-Terry rankings and scores"""
    pd = _get_pandas()
    
    st.subheader("🏆 Bradley-Terry Rankings")
    
    # Get case information for each extraction
    case_info = {}
    for _, row in comparison_df.iterrows():
        case_info[row['extraction_id_1']] = {
            'case_name': row['case_a_name'],
            'citation': row['case_a_citation'],
            'year': row['case_a_year']
        }
        case_info[row['extraction_id_2']] = {
            'case_name': row['case_b_name'],
            'citation': row['case_b_citation'],
            'year': row['case_b_year']
        }
    
    # Create ranking table
    ranking_data = []
    for ext_id in bradley_terry_stats['extraction_ids']:
        if ext_id in case_info:
            ranking_data.append({
                'Case': case_info[ext_id]['case_name'],
                'Citation': case_info[ext_id]['citation'],
                'Year': case_info[ext_id]['year'],
                'Bradley-Terry Score': bradley_terry_stats['bradley_terry_scores'][ext_id],
                'Win Rate': bradley_terry_stats['win_percentages'][ext_id],
                'Wins': bradley_terry_stats['wins'][ext_id],
                'Losses': bradley_terry_stats['losses'][ext_id],
                'Total Comparisons': bradley_terry_stats['total_comparisons'][ext_id]
            })
    
    # Sort by Bradley-Terry score (descending)
    ranking_df = pd.DataFrame(ranking_data)
    ranking_df = ranking_df.sort_values('Bradley-Terry Score', ascending=False)
    ranking_df.index = range(1, len(ranking_df) + 1)
    
    st.dataframe(ranking_df, use_container_width=True)
    
    # Interpretation
    st.info("📊 **Interpretation:** Higher Bradley-Terry scores indicate more 'rule-like' legal tests. Lower scores indicate more 'standard-like' tests.")

def show_temporal_analysis(comparison_df, bradley_terry_stats):
    """Show temporal analysis and regression"""
    pd = _get_pandas()
    
    st.subheader("📈 Temporal Analysis: Rule-Likeness Over Time")
    
    # Get case information with years and scores
    case_data = []
    case_info = {}
    
    # Build case info dictionary
    for _, row in comparison_df.iterrows():
        case_info[row['extraction_id_1']] = {
            'case_name': row['case_a_name'],
            'citation': row['case_a_citation'],
            'year': row['case_a_year']
        }
        case_info[row['extraction_id_2']] = {
            'case_name': row['case_b_name'],
            'citation': row['case_b_citation'],
            'year': row['case_b_year']
        }
    
    # Create temporal dataset
    for ext_id in bradley_terry_stats['extraction_ids']:
        if ext_id in case_info and case_info[ext_id]['year']:
            case_data.append({
                'Case': case_info[ext_id]['case_name'],
                'Year': case_info[ext_id]['year'],
                'Rule_Likeness_Score': bradley_terry_stats['bradley_terry_scores'][ext_id],
                'Win_Rate': bradley_terry_stats['win_percentages'][ext_id]
            })
    
    if not case_data:
        st.warning("No temporal data available for analysis")
        return
    
    temporal_df = pd.DataFrame(case_data)
    
    # Basic statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Time Period:**")
        min_year = temporal_df['Year'].min()
        max_year = temporal_df['Year'].max()
        st.write(f"From {min_year} to {max_year} ({max_year - min_year} years)")
        st.write(f"Cases analyzed: {len(temporal_df)}")
    
    with col2:
        st.write("**Rule-Likeness Statistics:**")
        mean_score = temporal_df['Rule_Likeness_Score'].mean()
        std_score = temporal_df['Rule_Likeness_Score'].std()
        st.write(f"Mean rule-likeness: {mean_score:.3f}")
        st.write(f"Standard deviation: {std_score:.3f}")
    
    # Regression analysis
    st.subheader("📉 Regression Analysis")
    
    try:
        # Simple linear regression
        X = temporal_df['Year'].values
        y = temporal_df['Rule_Likeness_Score'].values
        
        # Calculate regression coefficients
        n = len(X)
        x_mean = X.mean()
        y_mean = y.mean()
        
        # Calculate slope and intercept
        numerator = sum((X - x_mean) * (y - y_mean))
        denominator = sum((X - x_mean) ** 2)
        
        if denominator != 0:
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
            
            # Calculate R-squared
            y_pred = slope * X + intercept
            ss_res = sum((y - y_pred) ** 2)
            ss_tot = sum((y - y_mean) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Calculate standard error and p-value (simplified)
            se_slope = (sum((y - y_pred) ** 2) / (n - 2)) ** 0.5 / (sum((X - x_mean) ** 2) ** 0.5)
            t_stat = slope / se_slope if se_slope != 0 else 0
            
            # Display regression results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Slope (β)", f"{slope:.6f}")
                direction = "Increasing" if slope > 0 else "Decreasing" if slope < 0 else "Stable"
                st.write(f"Trend: {direction}")
            
            with col2:
                st.metric("R-squared", f"{r_squared:.4f}")
                strength = "Strong" if r_squared > 0.7 else "Moderate" if r_squared > 0.3 else "Weak"
                st.write(f"Relationship: {strength}")
            
            with col3:
                st.metric("t-statistic", f"{t_stat:.3f}")
                significance = "Significant" if abs(t_stat) > 2 else "Not significant"
                st.write(f"Statistical: {significance}")
            
            # Interpretation
            st.write("**Interpretation:**")
            if slope > 0:
                st.success(f"↗️ **Positive trend**: Legal tests are becoming more rule-like over time (slope: {slope:.6f})")
            elif slope < 0:
                st.error(f"↘️ **Negative trend**: Legal tests are becoming more standard-like over time (slope: {slope:.6f})")
            else:
                st.info("➡️ **No clear trend**: Rule-likeness remains relatively stable over time")
            
            # Interactive scatter plot with regression line
            st.subheader("📊 Interactive Regression Plot")
            
            # Create chart data with proper year bounds
            chart_data = temporal_df[['Year', 'Rule_Likeness_Score', 'Case']].copy()
            chart_data['Regression_Line'] = slope * chart_data['Year'] + intercept
            
            # Display the combined chart
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Use Altair for better control over the visualization
                try:
                    import altair as alt
                    
                    # Base chart
                    base = alt.Chart(chart_data).add_selection(
                        alt.selection_interval(bind='scales')
                    )
                    
                    # Scatter plot for actual data
                    scatter = base.mark_circle(size=100, color='steelblue').encode(
                        x=alt.X('Year:Q', scale=alt.Scale(domain=[1970, 2024]), title='Year'),
                        y=alt.Y('Rule_Likeness_Score:Q', title='Rule-Likeness Score'),
                        tooltip=['Case:N', 'Year:Q', 'Rule_Likeness_Score:Q']
                    )
                    
                    # Regression line
                    line = base.mark_line(color='red', strokeWidth=3).encode(
                        x=alt.X('Year:Q'),
                        y=alt.Y('Regression_Line:Q')
                    )
                    
                    # Combine charts
                    chart = (scatter + line).resolve_scale(y='shared')
                    
                    st.altair_chart(chart, use_container_width=True)
                    
                except ImportError:
                    # Fallback to simpler chart if Altair not available
                    st.line_chart(
                        data=chart_data.set_index('Year')[['Rule_Likeness_Score', 'Regression_Line']]
                    )
                
                st.caption("🔵 Blue dots: Actual Rule-Likeness Scores | 🔴 Red line: Regression Trend")
            
            with col2:
                st.write("**Regression Equation:**")
                st.code(f"y = {slope:.6f}x + {intercept:.3f}")
                st.write("")
                st.write("**Interpretation:**")
                if slope > 0:
                    st.write("📈 Positive trend")
                    st.write("Tests becoming more rule-like")
                elif slope < 0:
                    st.write("📉 Negative trend") 
                    st.write("Tests becoming more standard-like")
                else:
                    st.write("➡️ No clear trend")
                
                st.write("")
                st.metric("R²", f"{r_squared:.4f}")
                if r_squared > 0.7:
                    st.success("Strong fit")
                elif r_squared > 0.3:
                    st.info("Moderate fit")
                else:
                    st.warning("Weak fit")
            
            # Show detailed data table
            st.subheader("📋 Detailed Data")
            detailed_data = chart_data.copy()
            detailed_data['Residual'] = detailed_data['Rule_Likeness_Score'] - detailed_data['Regression_Line']
            detailed_data = detailed_data.round(4)
            st.dataframe(detailed_data, use_container_width=True)
            
        else:
            st.warning("Cannot perform regression analysis: insufficient variation in years")
    
    except Exception as e:
        st.error(f"Error in regression analysis: {str(e)}")

def show_reliability_analysis(comparison_df, bradley_terry_stats):
    """Show reliability and consistency analysis"""
    pd = _get_pandas()
    
    st.subheader("🔍 Reliability Analysis")
    
    reliability = bradley_terry_stats['reliability_metrics']
    
    # Key reliability metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Comparisons", reliability['total_comparisons'])
        st.metric("Decisive Comparisons", reliability['decisive_comparisons'])
    
    with col2:
        decisiveness = reliability['decisiveness_rate']
        st.metric("Decisiveness Rate", f"{decisiveness:.1%}")
        
        if decisiveness > 0.9:
            st.success("Excellent decisiveness")
        elif decisiveness > 0.7:
            st.info("Good decisiveness")
        else:
            st.warning("Low decisiveness")
    
    with col3:
        if reliability['average_confidence'] > 0:
            st.metric("Average Confidence", f"{reliability['average_confidence']:.2f}")
        else:
            st.metric("Average Confidence", "N/A")
    
    # Detailed reliability assessment
    st.subheader("📊 Reliability Assessment")
    
    # Consistency metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Consistency Metrics:**")
        transitivity = reliability['transitivity_score']
        st.metric("Transitivity Score", f"{transitivity:.2f}")
        
        if transitivity > 0.8:
            st.success("✅ High transitivity - results are logically consistent")
        elif transitivity > 0.6:
            st.info("⚠️ Moderate transitivity - some inconsistencies present")
        else:
            st.warning("❌ Low transitivity - significant inconsistencies detected")
    
    with col2:
        st.write("**Temporal Consistency:**")
        temporal_consistency = reliability['temporal_consistency']
        temporal_note = reliability['temporal_consistency_note']
        
        if temporal_consistency is not None:
            st.metric("Temporal Consistency", f"{temporal_consistency:.2f}")
            
            if temporal_consistency > 0.8:
                st.success("✅ Results are stable over time")
            elif temporal_consistency > 0.6:
                st.info("⚠️ Moderate temporal stability")
            else:
                st.warning("❌ Results vary significantly over time")
        else:
            st.metric("Temporal Consistency", "N/A")
        
        st.caption(f"📝 {temporal_note}")
    
    # Recommendations
    st.subheader("💡 Recommendations")
    
    recommendations = []
    
    if decisiveness < 0.7:
        recommendations.append("• **Low decisiveness**: Consider refining comparison prompts or criteria")
    
    if transitivity < 0.7:
        recommendations.append("• **Low transitivity**: Review comparison consistency and consider additional training")
    
    if reliability['average_confidence'] < 0.7 and reliability['average_confidence'] > 0:
        recommendations.append("• **Low confidence**: Results may need manual validation")
    
    if len(comparison_df) < 100:
        recommendations.append("• **Small sample**: Consider increasing the number of comparisons for more robust results")
    
    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.success("✅ **Excellent reliability**: Your comparison results appear robust and consistent!")
    
    # Show comparison distribution
    st.subheader("📈 Comparison Distribution")
    
    # Year distribution of comparisons
    year_data = []
    for _, row in comparison_df.iterrows():
        year_data.extend([row['case_a_year'], row['case_b_year']])
    
    if year_data:
        year_df = pd.DataFrame({'Year': year_data})
        year_counts = year_df['Year'].value_counts().sort_index()
        
        if len(year_counts) > 1:
            st.bar_chart(year_counts)
        else:
            st.info("All cases are from the same year")
    else:
        st.info("No year data available for distribution analysis")

def show_advanced_reliability_metrics(comparison_df, bradley_terry_stats):
    """Show advanced statistical reliability metrics for Bradley-Terry model"""
    pd = _get_pandas()
    np = _get_numpy()
    
    st.subheader("📈 Advanced Statistical Reliability")
    
    # Check if advanced metrics are available
    if 'confidence_intervals' not in bradley_terry_stats:
        st.error("Advanced reliability metrics not available. Please re-run the analysis.")
        return
    
    ci_data = bradley_terry_stats['confidence_intervals']
    significance_data = bradley_terry_stats.get('significance_tests', {})
    separation_data = bradley_terry_stats.get('separation_reliability', {})
    diagnostics_data = bradley_terry_stats.get('model_diagnostics', {})
    
    # Create tabs for different types of advanced metrics
    reliability_tab1, reliability_tab2, reliability_tab3, reliability_tab4 = st.tabs([
        "📊 Confidence Intervals", "🧪 Statistical Significance", "📏 Item Separation", "🔬 Model Diagnostics"
    ])
    
    with reliability_tab1:
        show_confidence_intervals_analysis(comparison_df, bradley_terry_stats, ci_data)
    
    with reliability_tab2:
        show_significance_analysis(comparison_df, bradley_terry_stats, significance_data)
    
    with reliability_tab3:
        show_separation_reliability_analysis(separation_data)
    
    with reliability_tab4:
        show_model_diagnostics_analysis(diagnostics_data)
    
    # Add export functionality
    st.markdown("---")
    st.subheader("📥 Export Statistical Summary")
    
    if st.button("Generate Export Table"):
        try:
            summary_df, model_stats = create_exportable_summary_table(bradley_terry_stats, comparison_df)
            
            # Display preview
            st.write("**Preview of Export Data:**")
            st.dataframe(summary_df.head(10), use_container_width=True)
            
            # Convert to CSV for download
            csv = summary_df.to_csv(index=False)
            
            # Create model statistics summary
            model_summary = "\n".join([f"{k}: {v}" for k, v in model_stats.items()])
            
            # Combine data and metadata
            full_export = f"# Bradley-Terry Statistical Analysis Summary\n\n## Model Statistics\n{model_summary}\n\n## Individual Item Results\n{csv}"
            
            st.download_button(
                label="📥 Download Statistical Summary (CSV)",
                data=full_export,
                file_name=f"bradley_terry_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            
            st.success("✅ Export table generated successfully!")
            
        except Exception as e:
            st.error(f"Error generating export: {str(e)}")

def show_confidence_intervals_analysis(comparison_df, bradley_terry_stats, ci_data):
    """Show confidence interval analysis"""
    pd = _get_pandas()
    
    st.subheader("📊 Confidence Intervals & Standard Errors")
    
    if 'warning' in ci_data or 'error' in ci_data:
        st.warning(f"Limited confidence interval data: {ci_data.get('warning', ci_data.get('error', ''))}")
    
    # Get case information
    case_info = {}
    for _, row in comparison_df.iterrows():
        case_info[row['extraction_id_1']] = {
            'case_name': row['case_a_name'],
            'citation': row['case_a_citation'],
            'year': row['case_a_year']
        }
        case_info[row['extraction_id_2']] = {
            'case_name': row['case_b_name'],
            'citation': row['case_b_citation'],
            'year': row['case_b_year']
        }
    
    # Create confidence interval table
    ci_table_data = []
    for ext_id in bradley_terry_stats['extraction_ids']:
        if ext_id in case_info and ext_id in ci_data['confidence_intervals']:
            ci_info = ci_data['confidence_intervals'][ext_id]
            se = ci_data['standard_errors'].get(ext_id, 0)
            
            ci_table_data.append({
                'Case': case_info[ext_id]['case_name'],
                'Citation': case_info[ext_id]['citation'],
                'Score': bradley_terry_stats['bradley_terry_scores'][ext_id],
                'Standard Error': se,
                'Lower 95% CI': ci_info['lower'],
                'Upper 95% CI': ci_info['upper'],
                'Margin Error': ci_info['margin_error'],
                'CI Width': ci_info['upper'] - ci_info['lower']
            })
    
    if ci_table_data:
        ci_df = pd.DataFrame(ci_table_data)
        ci_df = ci_df.sort_values('Score', ascending=False)
        
        # Display summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_se = ci_df['Standard Error'].mean()
            st.metric("Average Standard Error", f"{avg_se:.4f}")
        
        with col2:
            avg_width = ci_df['CI Width'].mean()
            st.metric("Average CI Width", f"{avg_width:.4f}")
        
        with col3:
            precision_score = 1 / (1 + avg_width) if avg_width > 0 else 1
            st.metric("Precision Index", f"{precision_score:.3f}")
        
        # Show table
        st.subheader("📋 Detailed Confidence Intervals")
        st.dataframe(ci_df.round(4), use_container_width=True)
        
        # Interpretation
        st.subheader("💡 Interpretation")
        narrow_cis = sum(1 for width in ci_df['CI Width'] if width < avg_width * 0.8)
        total_items = len(ci_df)
        
        if narrow_cis / total_items > 0.7:
            st.success(f"✅ **High precision**: {narrow_cis}/{total_items} items have narrow confidence intervals")
        elif narrow_cis / total_items > 0.4:
            st.info(f"⚠️ **Moderate precision**: {narrow_cis}/{total_items} items have narrow confidence intervals")
        else:
            st.warning(f"❌ **Low precision**: Only {narrow_cis}/{total_items} items have narrow confidence intervals")
    
    else:
        st.warning("No confidence interval data available")

def show_significance_analysis(comparison_df, bradley_terry_stats, significance_data):
    """Show statistical significance analysis"""
    pd = _get_pandas()
    
    st.subheader("🧪 Statistical Significance Testing")
    
    if 'warning' in significance_data:
        st.warning(f"Limited significance data: {significance_data['warning']}")
        return
    
    # Summary statistics
    significant_pairs = significance_data.get('significant_pairs', [])
    bonferroni_significant = significance_data.get('bonferroni_significant', [])
    n_comparisons = significance_data.get('n_comparisons', 0)
    alpha = significance_data.get('alpha', 0.05)
    bonferroni_alpha = significance_data.get('bonferroni_alpha', alpha)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Pairwise Tests", n_comparisons)
        st.metric("Significant (p < 0.05)", len(significant_pairs))
    
    with col2:
        st.metric("Bonferroni α", f"{bonferroni_alpha:.6f}")
        st.metric("Bonferroni Significant", len(bonferroni_significant))
    
    with col3:
        if n_comparisons > 0:
            sig_rate = len(significant_pairs) / n_comparisons
            st.metric("Significance Rate", f"{sig_rate:.1%}")
        
        discovery_rate = len(bonferroni_significant) / len(significant_pairs) if significant_pairs else 0
        st.metric("Corrected Discovery Rate", f"{discovery_rate:.1%}")
    
    # Show significant pairs
    if bonferroni_significant:
        st.subheader("🏆 Statistically Significant Differences (Bonferroni Corrected)")
        
        # Get case names for display
        case_info = {}
        for _, row in comparison_df.iterrows():
            case_info[row['extraction_id_1']] = row['case_a_name']
            case_info[row['extraction_id_2']] = row['case_b_name']
        
        sig_table = []
        for pair in bonferroni_significant[:20]:  # Show top 20
            item1_name = case_info.get(pair['item_1'], f"Item {pair['item_1']}")
            item2_name = case_info.get(pair['item_2'], f"Item {pair['item_2']}")
            
            sig_table.append({
                'Higher Ranked': item1_name,
                'Lower Ranked': item2_name,
                'Score Difference': pair['score_diff'],
                'Z-statistic': pair['z_statistic'],
                'P-value': pair['p_value']
            })
        
        sig_df = pd.DataFrame(sig_table)
        st.dataframe(sig_df.round(6), use_container_width=True)
    
    else:
        st.info("No statistically significant differences found after Bonferroni correction.")
    
    # Interpretation
    st.subheader("💡 Statistical Interpretation")
    
    if len(bonferroni_significant) > 0:
        st.success(f"✅ **Strong evidence**: {len(bonferroni_significant)} pairwise differences are statistically significant even after correcting for multiple comparisons.")
    elif len(significant_pairs) > 0:
        st.warning(f"⚠️ **Weak evidence**: {len(significant_pairs)} differences significant at α=0.05 but none survive multiple comparison correction.")
    else:
        st.error("❌ **No evidence**: No statistically significant differences detected.")

def show_separation_reliability_analysis(separation_data):
    """Show item separation reliability analysis"""
    st.subheader("📏 Item Separation Reliability")
    
    if 'error' in separation_data or 'warning' in separation_data:
        st.warning(f"Limited separation data: {separation_data.get('error', separation_data.get('warning', ''))}")
        return
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        separation_coef = separation_data.get('separation_coefficient', 0)
        st.metric("Separation Coefficient", f"{separation_coef:.3f}")
    
    with col2:
        reliability = separation_data.get('reliability', 0)
        st.metric("Separation Reliability", f"{reliability:.3f}")
    
    with col3:
        strata = separation_data.get('estimated_strata', 1)
        st.metric("Estimated Strata", strata)
    
    # Interpretation
    interpretation = separation_data.get('interpretation', 'Unknown')
    
    if reliability > 0.9:
        st.success(f"✅ **{interpretation}**: The model clearly distinguishes {strata} distinct performance levels.")
    elif reliability > 0.8:
        st.info(f"⚠️ **{interpretation}**: The model reliably separates items into {strata} levels.")
    elif reliability > 0.7:
        st.warning(f"⚠️ **{interpretation}**: The model shows acceptable separation into {strata} levels.")
    else:
        st.error(f"❌ **{interpretation}**: The model has difficulty reliably separating items.")
    
    # Additional details
    st.subheader("📊 Separation Analysis Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Variance Components:**")
        true_var = separation_data.get('true_variance', 0)
        error_var = separation_data.get('error_variance', 0)
        
        st.write(f"• True Score Variance: {true_var:.6f}")
        st.write(f"• Error Variance: {error_var:.6f}")
        
        if true_var > 0 and error_var > 0:
            signal_noise_ratio = true_var / error_var
            st.write(f"• Signal-to-Noise Ratio: {signal_noise_ratio:.3f}")
    
    with col2:
        st.write("**Reliability Benchmarks:**")
        st.write("• > 0.9: Excellent separation (3-4+ strata)")
        st.write("• > 0.8: Good separation (2-3 strata)")
        st.write("• > 0.7: Acceptable separation")
        st.write("• < 0.7: Poor separation")

def show_model_diagnostics_analysis(diagnostics_data):
    """Show model diagnostics and goodness-of-fit analysis"""
    st.subheader("🔬 Model Diagnostics")
    
    if 'error' in diagnostics_data:
        st.warning(f"Limited diagnostic data: {diagnostics_data['error']}")
        return
    
    # Key diagnostic metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        deviance = diagnostics_data.get('deviance', 0)
        st.metric("Deviance", f"{deviance:.2f}")
        
        pearson_chi2 = diagnostics_data.get('pearson_chi2', 0)
        st.metric("Pearson χ²", f"{pearson_chi2:.2f}")
    
    with col2:
        overdispersion = diagnostics_data.get('overdispersion_ratio', 1)
        st.metric("Overdispersion Ratio", f"{overdispersion:.3f}")
        
        df = diagnostics_data.get('degrees_freedom', 0)
        st.metric("Degrees of Freedom", df)
    
    with col3:
        dev_p = diagnostics_data.get('deviance_p_value', 1)
        st.metric("Deviance p-value", f"{dev_p:.4f}")
        
        pearson_p = diagnostics_data.get('pearson_p_value', 1)
        st.metric("Pearson p-value", f"{pearson_p:.4f}")
    
    # Interpretation
    interpretation = diagnostics_data.get('interpretation', 'Unknown')
    
    if overdispersion > 2:
        st.error(f"❌ **{interpretation}**: Model may be inadequate for the data.")
    elif overdispersion > 1.5:
        st.warning(f"⚠️ **{interpretation}**: Some model inadequacy detected.")
    elif overdispersion < 0.5:
        st.info(f"⚠️ **{interpretation}**: Possible model over-fitting.")
    else:
        st.success(f"✅ **{interpretation}**: Model fits the data appropriately.")
    
    # Goodness-of-fit assessment
    st.subheader("📈 Goodness-of-Fit Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Deviance Test:**")
        if dev_p < 0.001:
            st.error("Strong evidence of poor fit (p < 0.001)")
        elif dev_p < 0.05:
            st.warning("Evidence of poor fit (p < 0.05)")
        else:
            st.success("No evidence of poor fit (p ≥ 0.05)")
    
    with col2:
        st.write("**Pearson Test:**")
        if pearson_p < 0.001:
            st.error("Strong evidence of poor fit (p < 0.001)")
        elif pearson_p < 0.05:
            st.warning("Evidence of poor fit (p < 0.05)")
        else:
            st.success("No evidence of poor fit (p ≥ 0.05)")
    
    # Additional model information
    st.subheader("📋 Model Information")
    n_comparisons = diagnostics_data.get('n_comparisons', 0)
    n_parameters = diagnostics_data.get('n_parameters', 0)
    
    st.write(f"• Number of pairwise comparisons: {n_comparisons}")
    st.write(f"• Number of model parameters: {n_parameters}")
    st.write(f"• Degrees of freedom: {df}")
    
    if n_comparisons > 0 and n_parameters > 0:
        data_param_ratio = n_comparisons / n_parameters
        st.write(f"• Data-to-parameter ratio: {data_param_ratio:.1f}")
        
        if data_param_ratio < 5:
            st.warning("⚠️ Low data-to-parameter ratio may affect reliability")
        else:
            st.success("✅ Adequate data-to-parameter ratio")

def create_exportable_summary_table(bradley_terry_stats, comparison_df):
    """Create exportable statistical summary table for academic use"""
    pd = _get_pandas()
    
    # Get case information
    case_info = {}
    for _, row in comparison_df.iterrows():
        case_info[row['extraction_id_1']] = {
            'case_name': row['case_a_name'],
            'citation': row['case_a_citation'],
            'year': row['case_a_year']
        }
        case_info[row['extraction_id_2']] = {
            'case_name': row['case_b_name'],
            'citation': row['case_b_citation'],
            'year': row['case_b_year']
        }
    
    # Build comprehensive summary table
    summary_data = []
    
    ci_data = bradley_terry_stats.get('confidence_intervals', {})
    sep_data = bradley_terry_stats.get('separation_reliability', {})
    diag_data = bradley_terry_stats.get('model_diagnostics', {})
    
    for ext_id in bradley_terry_stats['extraction_ids']:
        if ext_id in case_info:
            ci_info = ci_data.get('confidence_intervals', {}).get(ext_id, {})
            se = ci_data.get('standard_errors', {}).get(ext_id, 0)
            
            summary_data.append({
                'Case_Name': case_info[ext_id]['case_name'],
                'Citation': case_info[ext_id]['citation'],
                'Decision_Year': case_info[ext_id]['year'],
                'Bradley_Terry_Score': bradley_terry_stats['bradley_terry_scores'][ext_id],
                'Standard_Error': se,
                'Lower_95_CI': ci_info.get('lower', 0),
                'Upper_95_CI': ci_info.get('upper', 0),
                'Win_Rate': bradley_terry_stats['win_percentages'][ext_id],
                'Total_Comparisons': bradley_terry_stats['total_comparisons'][ext_id]
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Add model-level statistics as metadata
    model_stats = {
        'Separation_Reliability': sep_data.get('reliability', 0),
        'Separation_Coefficient': sep_data.get('separation_coefficient', 0),
        'Estimated_Strata': sep_data.get('estimated_strata', 1),
        'Overdispersion_Ratio': diag_data.get('overdispersion_ratio', 1),
        'Model_Deviance': diag_data.get('deviance', 0),
        'Degrees_Freedom': diag_data.get('degrees_freedom', 0),
        'Total_Items': len(bradley_terry_stats['extraction_ids']),
        'Total_Comparisons': bradley_terry_stats['reliability_metrics']['total_comparisons']
    }
    
    return summary_df, model_stats

def run_extraction_for_experiment(experiment_id):
    """Execute legal test extraction for an experiment"""
    try:
        # Get experiment configuration
        exp = execute_sql("SELECT * FROM v2_experiments WHERE experiment_id = ?", (experiment_id,), fetch=True)
        if not exp:
            st.error("Experiment not found")
            return
            
        exp = exp[0]
        exp_dict = dict(zip(['experiment_id', 'name', 'description', 'researcher_name', 'status', 'ai_model', 
                           'temperature', 'top_p', 'top_k', 'max_output_tokens', 
                           'extraction_strategy', 'extraction_prompt', 'comparison_prompt',
                           'system_instruction', 'cost_limit_usd', 'created_date',
                           'modified_date', 'created_by'], exp))
        
        # Get API key from session state
        if 'api_key' not in st.session_state or not st.session_state.api_key:
            st.error("API key not configured. Please set your API key in the sidebar.")
            return
            
        api_key = st.session_state.api_key
            
        # Get cases that need extraction (using global selected cases pool for now)
        cases_to_extract = execute_sql("""
            SELECT c.case_id, c.case_name, c.citation, c.case_text, c.case_length
            FROM v2_cases c
            JOIN v2_experiment_selected_cases esc ON c.case_id = esc.case_id
            WHERE c.case_id NOT IN (
                SELECT case_id FROM v2_experiment_extractions 
                WHERE experiment_id = ?
            )
        """, (experiment_id,), fetch=True)
        
        if not cases_to_extract:
            st.info("No cases need extraction for this experiment. Make sure cases are selected for experiments in the Cases section.")
            return
            
        # Load extraction prompt from experiment configuration or file
        if exp_dict.get('extraction_prompt') and exp_dict['extraction_prompt'].strip():
            extraction_prompt = exp_dict['extraction_prompt']
        else:
            # Fall back to file if experiment doesn't have custom prompt
            prompt_file = 'prompts/extractor_prompt.txt'
            if os.path.exists(prompt_file):
                with open(prompt_file, 'r') as f:
                    extraction_prompt = f.read()
            else:
                extraction_prompt = "Extract the main legal test from this case."
            
        # Configure Gemini model with structured output
        import google.generativeai as genai
        
        # Define the structured schema for extraction
        extraction_schema = {
            "type": "object",
            "properties": {
                "legal_test": {
                    "type": "string",
                    "description": "The legal test extracted from the case"
                },
                "passages": {
                    "type": "string", 
                    "description": "The paragraphs (e.g., paras. x, y-z) or pages (e.g., pages x, y-z) from the decision where the test is found"
                },
                "test_novelty": {
                    "type": "string",
                    "enum": ["new test", "major change in existing test", "minor change in existing test", "application of existing test", "no substantive discussion"],
                    "description": "Classification of the test novelty"
                }
            },
            "required": ["legal_test", "passages", "test_novelty"]
        }
        
        # Configure model with structured output and system instruction
        genai.configure(api_key=api_key)
        system_instruction = exp_dict.get('system_instruction', '').strip()
        if not system_instruction:
            system_instruction = "You are a helpful assistant that helps legal researchers analyze legal texts."
        
        model = genai.GenerativeModel(
            model_name=exp_dict['ai_model'],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=extraction_schema,
                temperature=exp_dict['temperature'],
                top_p=exp_dict.get('top_p', 1.0),
                top_k=exp_dict.get('top_k', 40),
                max_output_tokens=exp_dict.get('max_output_tokens', 8192)
            ),
            system_instruction=system_instruction
        )
        
        # Create progress placeholder
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        total_cases = len(cases_to_extract)
        total_cost = 0.0
        
        status_placeholder.info(f"Starting extraction for {total_cases} cases...")
        
        # Process each case
        for i, case in enumerate(cases_to_extract):
            case_id, case_name, citation, case_text, case_length = case
            
            try:
                # Update progress
                progress_placeholder.progress((i + 1) / total_cases, text=f"Processing case {i + 1}/{total_cases}: {case_name}")
                
                # Prepare the prompt
                full_prompt = f"{extraction_prompt}\n\nCase Text:\n{case_text}"
                
                # Call Gemini API
                response = model.generate_content(full_prompt)
                
                # Parse structured JSON response
                try:
                    structured_response = json.loads(response.text)
                    legal_test = structured_response.get('legal_test', '')
                    passages = structured_response.get('passages', '')
                    test_novelty = structured_response.get('test_novelty', '')
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    legal_test = response.text
                    passages = "Not available"
                    test_novelty = "no substantive discussion"
                
                # Calculate cost (simplified)
                input_tokens = len(full_prompt.split()) * 1.3  # Rough token estimate
                output_tokens = len(response.text.split()) * 1.3
                model_pricing = GEMINI_MODELS.get(exp_dict['ai_model'], {'input': 0.30, 'output': 2.50})
                case_cost = (input_tokens / 1_000_000) * model_pricing['input'] + (output_tokens / 1_000_000) * model_pricing['output']
                total_cost += case_cost
                
                # Store in database with structured fields
                execute_sql("""
                    INSERT INTO v2_experiment_extractions 
                    (experiment_id, case_id, legal_test_name, legal_test_content, 
                     extraction_rationale, test_passages, test_novelty,
                     rule_like_score, confidence_score, validation_status, api_cost_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (experiment_id, case_id, f"Legal Test from {case_name}", legal_test,
                      "AI extracted legal test using structured output", passages, test_novelty,
                      0.5, 0.8, 'pending', case_cost))
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                st.error(f"Error processing case {case_name}: {str(e)}")
                continue
        
        # Update experiment status and cost
        execute_sql("UPDATE v2_experiments SET modified_date = CURRENT_TIMESTAMP WHERE experiment_id = ?", (experiment_id,))
        
        # Check if all extractions are complete
        remaining_cases = execute_sql("""
            SELECT COUNT(*) FROM v2_cases c
            JOIN v2_experiment_selected_cases esc ON c.case_id = esc.case_id
            WHERE c.case_id NOT IN (
                SELECT case_id FROM v2_experiment_extractions 
                WHERE experiment_id = ?
            )
        """, (experiment_id,), fetch=True)
        
        if remaining_cases and remaining_cases[0][0] == 0:
            # Check if comparisons are also complete
            total_comparisons = execute_sql("""
                SELECT COUNT(*) FROM v2_experiment_comparisons WHERE experiment_id = ?
            """, (experiment_id,), fetch=True)
            
            required_comparisons = calculate_bradley_terry_comparisons(len(cases_to_extract))
            
            if total_comparisons and total_comparisons[0][0] >= required_comparisons:
                execute_sql("UPDATE v2_experiments SET status = 'complete', modified_date = CURRENT_TIMESTAMP WHERE experiment_id = ?", (experiment_id,))
        
        progress_placeholder.empty()
        status_placeholder.success(f"✅ Extraction complete! Processed {total_cases} cases. Total cost: ${total_cost:.2f}")
        
    except Exception as e:
        st.error(f"Error during extraction: {str(e)}")

def run_comparisons_for_experiment(experiment_id):
    """Execute pairwise comparisons for an experiment"""
    try:
        # Get experiment configuration
        exp = execute_sql("SELECT * FROM v2_experiments WHERE experiment_id = ?", (experiment_id,), fetch=True)
        if not exp:
            st.error("Experiment not found")
            return
            
        exp = exp[0]
        exp_dict = dict(zip(['experiment_id', 'name', 'description', 'researcher_name', 'status', 'ai_model', 
                           'temperature', 'top_p', 'top_k', 'max_output_tokens', 
                           'extraction_strategy', 'extraction_prompt', 'comparison_prompt',
                           'system_instruction', 'cost_limit_usd', 'created_date',
                           'modified_date', 'created_by'], exp))
        
        # Get API key from session state
        if 'api_key' not in st.session_state or not st.session_state.api_key:
            st.error("API key not configured. Please set your API key in the sidebar.")
            return
            
        api_key = st.session_state.api_key
            
        # Get comparison pairs that need processing
        comparison_pairs, pair_block_info = generate_bradley_terry_comparison_pairs()
        
        if not comparison_pairs:
            st.info("No comparison pairs found. Please ensure Bradley-Terry structure is generated.")
            return
            
        # Get already completed comparisons
        completed_comparisons = execute_sql("""
            SELECT extraction_id_1, extraction_id_2 FROM v2_experiment_comparisons 
            WHERE experiment_id = ?
        """, (experiment_id,), fetch=True)
        
        completed_pairs = set((comp[0], comp[1]) for comp in completed_comparisons)
        
        # Get extractions for this experiment
        extractions = execute_sql("""
            SELECT extraction_id, case_id, legal_test_content 
            FROM v2_experiment_extractions 
            WHERE experiment_id = ?
        """, (experiment_id,), fetch=True)
        
        if not extractions:
            st.error("No extractions found for this experiment. Please run extractions first.")
            return
            
        extraction_map = {ext[1]: ext for ext in extractions}  # case_id -> extraction
        
        # Load comparison prompt from experiment configuration or file
        if exp_dict.get('comparison_prompt') and exp_dict['comparison_prompt'].strip():
            comparison_prompt = exp_dict['comparison_prompt']
        else:
            # Fall back to file if experiment doesn't have custom prompt
            prompt_file = 'prompts/comparator_prompt.txt'
            if os.path.exists(prompt_file):
                with open(prompt_file, 'r') as f:
                    comparison_prompt = f.read()
            else:
                comparison_prompt = "Compare these two legal tests and determine which is more rule-like."
            
        # Configure Gemini model with structured output for comparisons
        import google.generativeai as genai
        
        # Define the structured schema for comparison
        comparison_schema = {
            "type": "object",
            "properties": {
                "more_rule_like_test": {
                    "type": "string",
                    "enum": ["Test A", "Test B"],
                    "description": "Which test is more rule-like"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Clear reasoning for why the chosen test is more rule-like, referring to cases as Test A and Test B only"
                }
            },
            "required": ["more_rule_like_test", "reasoning"]
        }
        
        # Configure model with structured output and system instruction
        genai.configure(api_key=api_key)
        system_instruction = exp_dict.get('system_instruction', '').strip()
        if not system_instruction:
            system_instruction = "You are a helpful assistant that helps legal researchers analyze legal texts."
        
        model = genai.GenerativeModel(
            model_name=exp_dict['ai_model'],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=comparison_schema,
                temperature=exp_dict['temperature'],
                top_p=exp_dict.get('top_p', 1.0),
                top_k=exp_dict.get('top_k', 40),
                max_output_tokens=exp_dict.get('max_output_tokens', 8192)
            ),
            system_instruction=system_instruction
        )
        
        # Create progress placeholder
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        pairs_to_process = []
        for case_id_1, case_id_2 in comparison_pairs:
            if case_id_1 in extraction_map and case_id_2 in extraction_map:
                ext_1 = extraction_map[case_id_1]
                ext_2 = extraction_map[case_id_2]
                if (ext_1[0], ext_2[0]) not in completed_pairs and (ext_2[0], ext_1[0]) not in completed_pairs:
                    pairs_to_process.append((ext_1, ext_2))
        
        if not pairs_to_process:
            st.info("No comparison pairs need processing.")
            return
            
        total_pairs = len(pairs_to_process)
        total_cost = 0.0
        
        status_placeholder.info(f"Starting comparisons for {total_pairs} pairs...")
        
        # Process each pair
        for i, (ext_1, ext_2) in enumerate(pairs_to_process):
            try:
                # Update progress
                progress_placeholder.progress((i + 1) / total_pairs, text=f"Comparing pair {i + 1}/{total_pairs}")
                
                # Prepare the prompt
                full_prompt = f"{comparison_prompt}\n\nTest A: {ext_1[2]}\n\nTest B: {ext_2[2]}"
                
                # Call Gemini API
                response = model.generate_content(full_prompt)
                
                # Parse structured JSON response
                try:
                    structured_response = json.loads(response.text)
                    more_rule_like_test = structured_response.get('more_rule_like_test', 'Test A')
                    reasoning = structured_response.get('reasoning', '')
                    
                    # Determine winner based on structured response
                    winner_id = ext_1[0] if more_rule_like_test == "Test A" else ext_2[0]
                    
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    response_text = response.text
                    winner_id = ext_1[0] if "Test A" in response_text else ext_2[0]
                    reasoning = response_text
                
                # Calculate cost (simplified)
                input_tokens = len(full_prompt.split()) * 1.3
                output_tokens = len(response.text.split()) * 1.3
                model_pricing = GEMINI_MODELS.get(exp_dict['ai_model'], {'input': 0.30, 'output': 2.50})
                pair_cost = (input_tokens / 1_000_000) * model_pricing['input'] + (output_tokens / 1_000_000) * model_pricing['output']
                total_cost += pair_cost
                
                # Store in database with structured fields
                execute_sql("""
                    INSERT INTO v2_experiment_comparisons 
                    (experiment_id, extraction_id_1, extraction_id_2, winner_id, 
                     comparison_rationale, confidence_score, api_cost_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (experiment_id, ext_1[0], ext_2[0], winner_id, reasoning, 0.8, pair_cost))
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                st.error(f"Error processing comparison pair {i + 1}: {str(e)}")
                continue
        
        # Update experiment status
        execute_sql("UPDATE v2_experiments SET modified_date = CURRENT_TIMESTAMP WHERE experiment_id = ?", (experiment_id,))
        
        # Check if all comparisons are complete
        total_comparisons = execute_sql("""
            SELECT COUNT(*) FROM v2_experiment_comparisons WHERE experiment_id = ?
        """, (experiment_id,), fetch=True)
        
        required_comparisons = calculate_bradley_terry_comparisons(len(extractions))
        
        if total_comparisons and total_comparisons[0][0] >= required_comparisons:
            execute_sql("UPDATE v2_experiments SET status = 'complete', modified_date = CURRENT_TIMESTAMP WHERE experiment_id = ?", (experiment_id,))
        
        progress_placeholder.empty()
        status_placeholder.success(f"✅ Comparisons complete! Processed {total_pairs} pairs. Total cost: ${total_cost:.2f}")
        
    except Exception as e:
        st.error(f"Error during comparisons: {str(e)}")

@st.cache_resource
def initialize_experiment_tables():
    """Initialize database tables for experiment management (cached - runs only once)"""
    
    # Cases table (v2) - Just store the cases and metadata
    from config import DB_TYPE
    if DB_TYPE == 'postgresql':
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_cases (
                case_id SERIAL PRIMARY KEY,
                case_name TEXT NOT NULL,
                citation TEXT UNIQUE NOT NULL,
                decision_year INTEGER,
                area_of_law TEXT,
                subject TEXT,
                decision_url TEXT,
                case_text TEXT,
                case_length INTEGER,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')
    else:
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_cases (
                case_id INTEGER PRIMARY KEY,
                case_name TEXT NOT NULL,
                citation TEXT UNIQUE NOT NULL,
                decision_year INTEGER,
                area_of_law TEXT,
                subject TEXT,
                decision_url TEXT,
                case_text TEXT,
                case_length INTEGER,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')
    
    # Experiments table (v2)
    if DB_TYPE == 'postgresql':
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_experiments (
                experiment_id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                researcher_name TEXT DEFAULT '',
                status TEXT DEFAULT 'draft',
                ai_model TEXT DEFAULT 'gemini-2.5-pro',
                temperature REAL DEFAULT 0.0,
                top_p REAL DEFAULT 1.0,
                top_k INTEGER DEFAULT 40,
                max_output_tokens INTEGER DEFAULT 8192,
                extraction_strategy TEXT DEFAULT 'single_test',
                extraction_prompt TEXT,
                comparison_prompt TEXT,
                system_instruction TEXT,
                cost_limit_usd REAL DEFAULT 100.0,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                modified_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT DEFAULT 'researcher'
            );
        ''')
    else:
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_experiments (
                experiment_id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                researcher_name TEXT DEFAULT '',
                status TEXT DEFAULT 'draft',
                ai_model TEXT DEFAULT 'gemini-2.5-pro',
                temperature REAL DEFAULT 0.0,
                top_p REAL DEFAULT 1.0,
                top_k INTEGER DEFAULT 40,
                max_output_tokens INTEGER DEFAULT 8192,
                extraction_strategy TEXT DEFAULT 'single_test',
                extraction_prompt TEXT,
                comparison_prompt TEXT,
                system_instruction TEXT,
                cost_limit_usd REAL DEFAULT 100.0,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                modified_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT DEFAULT 'researcher'
            );
        ''')
    
    # Experiment runs table (v2)
    if DB_TYPE == 'postgresql':
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_experiment_runs (
                run_id SERIAL PRIMARY KEY,
                experiment_id INTEGER,
                run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending',
                cases_processed INTEGER DEFAULT 0,
                tests_extracted INTEGER DEFAULT 0,
                comparisons_completed INTEGER DEFAULT 0,
                total_cost_usd REAL DEFAULT 0.0,
                execution_time_minutes REAL DEFAULT 0.0,
                error_message TEXT,
                FOREIGN KEY (experiment_id) REFERENCES v2_experiments (experiment_id)
            );
        ''')
    else:
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_experiment_runs (
                run_id INTEGER PRIMARY KEY,
                experiment_id INTEGER,
                run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending',
                cases_processed INTEGER DEFAULT 0,
                tests_extracted INTEGER DEFAULT 0,
                comparisons_completed INTEGER DEFAULT 0,
                total_cost_usd REAL DEFAULT 0.0,
                execution_time_minutes REAL DEFAULT 0.0,
                error_message TEXT,
                FOREIGN KEY (experiment_id) REFERENCES v2_experiments (experiment_id)
            );
        ''')
    
    # Experiment extractions table (v2) - Store extractions per experiment
    if DB_TYPE == 'postgresql':
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_experiment_extractions (
                extraction_id SERIAL PRIMARY KEY,
                experiment_id INTEGER,
                case_id INTEGER,
                legal_test_name TEXT,
                legal_test_content TEXT,
                extraction_rationale TEXT,
                rule_like_score REAL,
                confidence_score REAL,
                validation_status TEXT DEFAULT 'pending',
                validator_notes TEXT,
                api_cost_usd REAL DEFAULT 0.0,
                extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES v2_experiments (experiment_id),
                FOREIGN KEY (case_id) REFERENCES v2_cases (case_id),
                UNIQUE(experiment_id, case_id)
            );
        ''')
    else:
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_experiment_extractions (
                extraction_id INTEGER PRIMARY KEY,
                experiment_id INTEGER,
                case_id INTEGER,
                legal_test_name TEXT,
                legal_test_content TEXT,
                extraction_rationale TEXT,
                rule_like_score REAL,
                confidence_score REAL,
                validation_status TEXT DEFAULT 'pending',
                validator_notes TEXT,
                api_cost_usd REAL DEFAULT 0.0,
                extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES v2_experiments (experiment_id),
                FOREIGN KEY (case_id) REFERENCES v2_cases (case_id),
                UNIQUE(experiment_id, case_id)
            );
        ''')
    
    # Experiment comparisons table (v2) - Store pairwise comparisons per experiment
    if DB_TYPE == 'postgresql':
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_experiment_comparisons (
                comparison_id SERIAL PRIMARY KEY,
                experiment_id INTEGER,
                extraction_id_1 INTEGER,
                extraction_id_2 INTEGER,
                winner_id INTEGER,
                comparison_rationale TEXT,
                confidence_score REAL,
                human_validated BOOLEAN DEFAULT FALSE,
                api_cost_usd REAL DEFAULT 0.0,
                comparison_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES v2_experiments (experiment_id),
                FOREIGN KEY (extraction_id_1) REFERENCES v2_experiment_extractions (extraction_id),
                FOREIGN KEY (extraction_id_2) REFERENCES v2_experiment_extractions (extraction_id),
                FOREIGN KEY (winner_id) REFERENCES v2_experiment_extractions (extraction_id),
                UNIQUE(experiment_id, extraction_id_1, extraction_id_2)
            );
        ''')
    else:
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_experiment_comparisons (
                comparison_id INTEGER PRIMARY KEY,
                experiment_id INTEGER,
                extraction_id_1 INTEGER,
                extraction_id_2 INTEGER,
                winner_id INTEGER,
                comparison_rationale TEXT,
                confidence_score REAL,
                human_validated BOOLEAN DEFAULT FALSE,
                api_cost_usd REAL DEFAULT 0.0,
                comparison_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES v2_experiments (experiment_id),
                FOREIGN KEY (extraction_id_1) REFERENCES v2_experiment_extractions (extraction_id),
                FOREIGN KEY (extraction_id_2) REFERENCES v2_experiment_extractions (extraction_id),
                FOREIGN KEY (winner_id) REFERENCES v2_experiment_extractions (extraction_id),
                UNIQUE(experiment_id, extraction_id_1, extraction_id_2)
            );
        ''')
    
    # Experiment results summary table (v2)
    if DB_TYPE == 'postgresql':
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_experiment_results (
                result_id SERIAL PRIMARY KEY,
                experiment_id INTEGER,
                metric_type TEXT,
                metric_value REAL,
                bt_statistics_json TEXT,
                regression_results_json TEXT,
                confidence_scores_json TEXT,
                calculated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES v2_experiments (experiment_id)
            );
        ''')
    else:
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_experiment_results (
                result_id INTEGER PRIMARY KEY,
                experiment_id INTEGER,
                metric_type TEXT,
                metric_value REAL,
                bt_statistics_json TEXT,
                regression_results_json TEXT,
                confidence_scores_json TEXT,
                calculated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES v2_experiments (experiment_id)
            );
        ''')
    
    # Experiment selected cases table (v2) - Cases chosen for experiments
    if DB_TYPE == 'postgresql':
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_experiment_selected_cases (
                selection_id SERIAL PRIMARY KEY,
                case_id INTEGER,
                selected_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                selected_by TEXT DEFAULT 'researcher',
                FOREIGN KEY (case_id) REFERENCES v2_cases (case_id),
                UNIQUE(case_id)
            );
        ''')
    else:
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_experiment_selected_cases (
                selection_id INTEGER PRIMARY KEY,
                case_id INTEGER,
                selected_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                selected_by TEXT DEFAULT 'researcher',
                FOREIGN KEY (case_id) REFERENCES v2_cases (case_id),
                UNIQUE(case_id)
            );
        ''')
    
    # Bradley-Terry structure table (v2) - Centralized block assignments for consistent methodology
    if DB_TYPE == 'postgresql':
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_bradley_terry_structure (
                structure_id SERIAL PRIMARY KEY,
                case_id INTEGER,
                block_number INTEGER NOT NULL,
                case_role TEXT CHECK (case_role IN ('core', 'bridge')) NOT NULL,
                importance_score REAL DEFAULT 0.0,
                assigned_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                assigned_by TEXT DEFAULT 'system',
                FOREIGN KEY (case_id) REFERENCES v2_cases (case_id),
                UNIQUE(case_id)
            );
        ''')
    else:
        execute_sql('''
            CREATE TABLE IF NOT EXISTS v2_bradley_terry_structure (
                structure_id INTEGER PRIMARY KEY,
                case_id INTEGER,
                block_number INTEGER NOT NULL,
                case_role TEXT CHECK (case_role IN ('core', 'bridge')) NOT NULL,
                importance_score REAL DEFAULT 0.0,
                assigned_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                assigned_by TEXT DEFAULT 'system',
                FOREIGN KEY (case_id) REFERENCES v2_cases (case_id),
                UNIQUE(case_id)
            );
        ''')
    
    # Add new columns for structured extraction data (if they don't exist)
    try:
        execute_sql('ALTER TABLE v2_experiment_extractions ADD COLUMN test_passages TEXT;')
    except:
        pass  # Column already exists
    
    try:
        execute_sql('ALTER TABLE v2_experiment_extractions ADD COLUMN test_novelty TEXT;')
    except:
        pass  # Column already exists
    
    # Add indexes for performance (v2)
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_cases_citation ON v2_cases (citation);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_cases_year ON v2_cases (decision_year);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_experiments_status ON v2_experiments (status);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_experiment_runs_experiment_id ON v2_experiment_runs (experiment_id);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_experiment_extractions_experiment_id ON v2_experiment_extractions (experiment_id);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_experiment_extractions_case_id ON v2_experiment_extractions (case_id);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_experiment_comparisons_experiment_id ON v2_experiment_comparisons (experiment_id);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_experiment_results_experiment_id ON v2_experiment_results (experiment_id);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_experiment_selected_cases_case_id ON v2_experiment_selected_cases (case_id);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_bradley_terry_structure_case_id ON v2_bradley_terry_structure (case_id);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_bradley_terry_structure_block ON v2_bradley_terry_structure (block_number);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_bradley_terry_structure_role ON v2_bradley_terry_structure (case_role);')
    
    # Additional performance indexes
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_experiments_modified_date ON v2_experiments (modified_date DESC);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_experiment_runs_date ON v2_experiment_runs (run_date);')
    execute_sql('CREATE INDEX IF NOT EXISTS idx_v2_cases_case_length ON v2_cases (case_length) WHERE case_length IS NOT NULL;')

# Cache experiment overview data for 60 seconds
@st.cache_data(ttl=60)
def get_experiments_overview_data():
    """Get optimized experiment data for overview with minimal columns"""
    experiments_data = execute_sql("""
        SELECT 
            e.experiment_id, e.name, e.description, e.status, e.ai_model, 
            e.extraction_strategy, e.modified_date,
            COALESCE(
                (SELECT SUM(api_cost_usd) FROM v2_experiment_extractions WHERE experiment_id = e.experiment_id) + 
                (SELECT SUM(api_cost_usd) FROM v2_experiment_comparisons WHERE experiment_id = e.experiment_id), 
                0
            ) as total_cost,
            COALESCE((SELECT COUNT(*) FROM v2_experiment_extractions WHERE experiment_id = e.experiment_id), 0) as total_tests,
            COALESCE((SELECT COUNT(*) FROM v2_experiment_comparisons WHERE experiment_id = e.experiment_id), 0) as total_comparisons
        FROM v2_experiments e
        ORDER BY e.modified_date DESC
        LIMIT 50
    """, fetch=True)
    return experiments_data

# Cache case statistics for 120 seconds (changes less frequently)
@st.cache_data(ttl=120)
def get_case_statistics():
    """Get case count and length statistics"""
    stats = {}
    
    # Get counts
    selected_cases_count = execute_sql("SELECT COUNT(*) FROM v2_experiment_selected_cases", fetch=True)[0][0]
    total_cases_count = execute_sql("SELECT COUNT(*) FROM v2_cases", fetch=True)[0][0]
    
    # Get case length statistics with single query
    if selected_cases_count > 0:
        case_stats = execute_sql("""
            SELECT 
                AVG(CASE WHEN s.case_id IS NOT NULL THEN c.case_length END) as avg_selected_length,
                AVG(c.case_length) as avg_all_length
            FROM v2_cases c 
            LEFT JOIN v2_experiment_selected_cases s ON c.case_id = s.case_id
            WHERE c.case_length IS NOT NULL
        """, fetch=True)
        
        if case_stats and case_stats[0]:
            stats['avg_selected_case_length'] = float(case_stats[0][0]) if case_stats[0][0] else 52646.0
            stats['avg_all_case_length'] = float(case_stats[0][1]) if case_stats[0][1] else 52646.0
        else:
            stats['avg_selected_case_length'] = 52646.0
            stats['avg_all_case_length'] = 52646.0
    else:
        # If no selected cases, get overall average
        all_avg = execute_sql("SELECT AVG(case_length) FROM v2_cases WHERE case_length IS NOT NULL", fetch=True)
        avg_length = float(all_avg[0][0]) if all_avg and all_avg[0][0] else 52646.0
        stats['avg_selected_case_length'] = avg_length  
        stats['avg_all_case_length'] = avg_length
    
    stats['selected_cases_count'] = selected_cases_count
    stats['total_cases_count'] = total_cases_count
    
    return stats

# Cache CSS injection for startup performance
@st.cache_data(ttl=3600)  # 1 hour cache for CSS
def inject_sidebar_css():
    """Inject cached CSS for sidebar styling"""
    st.markdown("""
    <style>
    /* Compact sidebar styling */
    .stSidebar .stButton > button {
        padding: 0.5rem 1rem !important;
        margin: 0.2rem 0 !important;
        border-radius: 6px !important;
        font-size: 0.9rem !important;
        height: auto !important;
        min-height: 2.5rem !important;
    }
    
    /* Selected navigation buttons - blue theme */
    div[data-testid="stSidebar"] button[kind="secondary"] {
        background-color: #0066cc !important;
        color: white !important;
        border: 1px solid #004499 !important;
        box-shadow: 0 2px 4px rgba(0,102,204,0.2) !important;
    }
    div[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #0056b3 !important;
        color: white !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0,102,204,0.3) !important;
    }
    
    /* Action buttons - red theme */
    div[data-testid="stSidebar"] button[kind="primary"] {
        background-color: #ff4b4b !important;
        color: white !important;
        border: 1px solid #cc0000 !important;
        box-shadow: 0 2px 4px rgba(255,75,75,0.2) !important;
    }
    div[data-testid="stSidebar"] button[kind="primary"]:hover {
        background-color: #e60000 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(255,75,75,0.3) !important;
    }
    
    /* Default buttons - clean styling */
    div[data-testid="stSidebar"] button:not([kind]) {
        background-color: #f8f9fa !important;
        color: #333 !important;
        border: 1px solid #dee2e6 !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stSidebar"] button:not([kind]):hover {
        background-color: #e9ecef !important;
        color: #000 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    /* Experiment list styling */
    .stSidebar .stExpander {
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Compact spacing */
    .stSidebar .stMarkdown {
        margin: 0.3rem 0 !important;
    }
    
    /* Settings button special styling */
    div[data-testid="stSidebar"] button[aria-label*="Settings"] {
        padding: 0.3rem 0.5rem !important;
        font-size: 1.1rem !important;
        min-height: 2rem !important;
        border-radius: 50% !important;
        width: 2.5rem !important;
        height: 2.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Cache cost calculation parameters for 300 seconds (5 min - very stable)
@st.cache_data(ttl=300)
def get_cost_calculation_params(n_cases, avg_selected_case_length, extraction_strategy='single_test'):
    """Pre-calculate shared cost parameters for all experiment cards"""
    required_comparisons = calculate_bradley_terry_comparisons(n_cases)
    
    # Calculate shared cost parameters
    avg_tokens_selected = avg_selected_case_length / 4
    system_prompt_tokens = 100
    extraction_prompt_tokens = 200
    comparison_prompt_tokens = 150
    extracted_test_tokens = 325  # ~250 words * 1.3 tokens/word
    
    # Base cost per extraction for any model (includes prompt tokens)
    extraction_input_tokens = avg_tokens_selected + system_prompt_tokens + extraction_prompt_tokens
    
    # Calculate comparison input tokens based on extraction strategy
    if extraction_strategy == 'full_text_comparison':
        # For full text comparison, we compare entire case texts
        comparison_input_tokens = (avg_tokens_selected * 2) + system_prompt_tokens + comparison_prompt_tokens
    else:
        # For single_test/multi_test, we compare extracted tests
        comparison_input_tokens = (extracted_test_tokens * 2) + system_prompt_tokens + comparison_prompt_tokens
    
    comparison_output_tokens = 100  # Estimated tokens for comparison result
    
    # Bradley-Terry parameters for display
    block_size = 15  # 12 core + 3 bridge cases per block
    core_cases_per_block = 12
    comparisons_per_block = 105
    
    return {
        'required_comparisons': required_comparisons,
        'extraction_input_tokens': extraction_input_tokens,
        'extracted_test_tokens': extracted_test_tokens,
        'comparison_input_tokens': comparison_input_tokens,
        'comparison_output_tokens': comparison_output_tokens,
        'block_size': block_size,
        'core_cases_per_block': core_cases_per_block,
        'comparisons_per_block': comparisons_per_block
    }

def show_experiment_overview():
    """Display overview of all experiments"""
    st.header("📊 Experiment Overview")
    
    # Get cached experiment data
    experiments_data = get_experiments_overview_data()
    
    if not experiments_data:
        st.info("No experiments found. Create your first experiment below!")
        return
    
    # Convert to DataFrame with optimized columns
    columns = ['experiment_id', 'name', 'description', 'status', 'ai_model', 
               'extraction_strategy', 'modified_date', 'total_cost', 'total_tests', 'total_comparisons']
    
    pd = _get_pandas()
    df = pd.DataFrame(experiments_data, columns=columns)
    
    # Get cached case statistics
    try:
        stats = get_case_statistics()
        selected_cases_count = stats['selected_cases_count']
        total_cases_count = stats['total_cases_count']
        avg_selected_case_length = stats['avg_selected_case_length']
        avg_all_case_length = stats['avg_all_case_length']
        
    except Exception as e:
        selected_cases_count = 0
        total_cases_count = 0
        avg_selected_case_length = 52646.0
        avg_all_case_length = 52646.0
    
    # Calculate shared parameters once using cached function
    n_cases = selected_cases_count
    # Use default strategy for overview - individual cards will recalculate with specific strategy
    cost_params = get_cost_calculation_params(n_cases, avg_selected_case_length, 'single_test')
    
    required_comparisons = cost_params['required_comparisons']
    block_size = cost_params['block_size']
    core_cases_per_block = cost_params['core_cases_per_block']
    comparisons_per_block = cost_params['comparisons_per_block']
    
    # Skip if no cases selected
    if n_cases == 0:
        st.info("No cases selected for experiments yet. Select cases first to see experiment details.")
        return
    
    # Display experiment cards in responsive grid
    st.write(f"**{len(df)} experiments found**")
    st.write("")
    
    # Create responsive columns for card layout (3 cards per row on wide screens)
    num_cols = 3
    rows = [df.iloc[i:i+num_cols] for i in range(0, len(df), num_cols)]
    
    for row in rows:
        cols = st.columns(num_cols)
        
        for idx, (_, exp) in enumerate(row.iterrows()):
            if idx < len(cols):  # Safety check
                with cols[idx]:
                    show_experiment_card(exp, n_cases, required_comparisons, 
                                        avg_selected_case_length, avg_all_case_length, 
                                        total_cases_count, cost_params)

def show_experiment_card(exp, n_cases, required_comparisons, avg_selected_case_length, 
                        avg_all_case_length, total_cases_count, cost_params):
    """Display a single experiment card"""
            
    # Get model pricing (prices are per million tokens)
    model_pricing = GEMINI_MODELS.get(exp['ai_model'], {'input': 0.30, 'output': 2.50})
    
    # Recalculate cost parameters for this specific experiment's strategy
    exp_cost_params = get_cost_calculation_params(n_cases, avg_selected_case_length, exp['extraction_strategy'])
    
    extraction_input_tokens = exp_cost_params['extraction_input_tokens']
    extracted_test_tokens = exp_cost_params['extracted_test_tokens']
    comparison_input_tokens = exp_cost_params['comparison_input_tokens']
    comparison_output_tokens = exp_cost_params['comparison_output_tokens']
    
    # Calculate per-case costs
    extraction_cost_per_case = (extraction_input_tokens / 1_000_000) * model_pricing['input'] + (extracted_test_tokens / 1_000_000) * model_pricing['output']
    comparison_cost_per_pair = (comparison_input_tokens / 1_000_000) * model_pricing['input'] + (comparison_output_tokens / 1_000_000) * model_pricing['output']
    
    # Calculate estimates
    remaining_extractions = max(0, n_cases - int(exp['total_tests'] or 0))
    remaining_comparisons = max(0, required_comparisons - int(exp['total_comparisons'] or 0))
    
    extraction_cost_estimate = remaining_extractions * extraction_cost_per_case
    
    # Calculate total sample cost based on strategy
    if exp['extraction_strategy'] == 'full_text_comparison':
        # No extraction cost for full text comparison
        sample_total_cost = required_comparisons * comparison_cost_per_pair
    else:
        # Extraction + comparison costs
        sample_total_cost = (n_cases * extraction_cost_per_case) + (required_comparisons * comparison_cost_per_pair)
    
    # Status and progress
    status_colors = {
        'draft': '🟡 Draft',
        'active': '🟢 Active', 
        'completed': '🔵 Completed',
        'archived': '⚫ Archived'
    }
    
    # Use Streamlit's built-in container with visual separation
    with st.container(border=True):
        # Card content with smaller title
        st.markdown(f"### 🧪 #{exp['experiment_id']} {exp['name']}")
        st.markdown(f"**{status_colors.get(exp['status'], exp['status'])}**")
        
        # Key metrics in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Tests", f"{int(exp['total_tests'] or 0)}/{n_cases}")
            st.metric("Comparisons", f"{int(exp['total_comparisons'] or 0)}/{required_comparisons}")
        
        with col2:
            st.metric("Spent", f"${exp['total_cost'] or 0:.2f}")
            st.metric("Sample Est.", f"${sample_total_cost:.2f}")
        
        # Description (truncated)
        description = exp['description'] or 'No description'
        if len(description) > 60:
            description = description[:60] + "..."
        st.caption(f"**Model:** {exp['ai_model']} | **Strategy:** {exp['extraction_strategy']}")
        st.caption(f"**Description:** {description}")
        
        # Actions - only Details and Configure buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Details", key=f"details_{exp['experiment_id']}", use_container_width=True):
                st.session_state.selected_page = "Experiment Detail"
                st.session_state.selected_experiment = exp['experiment_id']
                st.rerun()
        
        with col2:
            if st.button("⚙️ Configure", key=f"config_{exp['experiment_id']}", use_container_width=True):
                st.session_state.editing_experiment = exp['experiment_id']
                st.rerun()

def show_experiment_configuration():
    """Show experiment configuration interface with two-step form"""
    st.header("⚙️ Experiment Configuration")
    
    # Initialize session state for multi-step form
    if 'config_step' not in st.session_state:
        st.session_state.config_step = 1
    
    # Check if we're editing an existing experiment
    editing_id = st.session_state.get('editing_experiment')
    
    if editing_id:
        # Load existing experiment
        exp_data = execute_sql(
            "SELECT * FROM v2_experiments WHERE experiment_id = ?", 
            (editing_id,), 
            fetch=True
        )
        if exp_data:
            exp = dict(zip(['experiment_id', 'name', 'description', 'researcher_name', 'status', 'ai_model', 
                           'temperature', 'top_p', 'top_k', 'max_output_tokens', 
                           'extraction_strategy', 'extraction_prompt', 'comparison_prompt',
                           'system_instruction', 'cost_limit_usd', 'created_date',
                           'modified_date', 'created_by'], exp_data[0]))
            
            # Pre-populate session state with existing experiment data
            if 'experiment_config' not in st.session_state:
                st.session_state.experiment_config = exp
        else:
            st.error("Experiment not found!")
            return
    else:
        # Default values for new experiment
        exp = {
            'name': '',
            'description': '',
            'researcher_name': '',
            'status': 'draft',
            'ai_model': 'gemini-2.5-pro',
            'temperature': 0.0,
            'top_p': 1.0,
            'top_k': 40,
            'max_output_tokens': 8192,
            'extraction_strategy': 'single_test',
            'extraction_prompt': '',
            'comparison_prompt': '',
            'system_instruction': 'You are a helpful assistant that helps legal researchers analyze legal texts.',
            'cost_limit_usd': 100.0
        }
        
        # Initialize session state with defaults
        if 'experiment_config' not in st.session_state:
            st.session_state.experiment_config = exp
    
    # Show appropriate step
    if st.session_state.config_step == 1:
        show_basic_info_form(exp, editing_id)
    elif st.session_state.config_step == 2:
        show_prompts_config_form(exp, editing_id)
    
def show_basic_info_form(exp, editing_id):
    """Show Step 1: Basic Information and AI Configuration"""
    # Progress indicator
    st.progress(0.5, text="Step 1 of 2: Basic Configuration")
    
    with st.form("basic_info_form"):
        # Basic Information
        st.subheader("📝 Basic Information")
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Experiment Name", value=exp['name'])
            researcher_name = st.text_input("Researcher's Name", value=exp['researcher_name'])
        
        with col2:
            description = st.text_area("Description", value=exp['description'])
            cost_limit = st.number_input("Cost Limit (USD)", min_value=0.0, value=float(exp['cost_limit_usd']))
        
        # AI Model Configuration
        st.subheader("🤖 AI Model Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            ai_model = st.selectbox("AI Model", list(GEMINI_MODELS.keys()), 
                                  index=list(GEMINI_MODELS.keys()).index(exp['ai_model']))
            temperature = st.slider(
                "Temperature", 
                0.0, 2.0, 
                float(exp['temperature']), 
                step=0.1,
                help="Controls response randomness. 0.0 = deterministic/consistent, 0.5-0.7 = balanced, 1.0+ = creative. Recommended: 0.0-0.3 for legal analysis."
            )
            top_p = st.slider(
                "Top P", 
                0.0, 1.0, 
                float(exp['top_p']), 
                step=0.1,
                help="Nucleus sampling threshold. 0.1 = very focused, 0.9 = diverse vocabulary, 1.0 = no filtering. Recommended: 0.8-1.0 for comprehensive analysis."
            )
        
        with col2:
            top_k = st.slider(
                "Top K", 
                1, 100, 
                int(exp['top_k']), 
                step=1,
                help="Number of top tokens to consider. 1 = deterministic, 20-40 = balanced, 80+ = creative. Recommended: 20-60 for legal text."
            )
            max_tokens = st.number_input(
                "Max Output Tokens", 
                min_value=1, 
                max_value=16384, 
                value=int(exp['max_output_tokens']),
                help="Maximum response length. 1000-2000 = summaries, 4000-8192 = detailed analysis, 8192 = comprehensive extraction."
            )
        
        # Extraction Strategy Selection
        st.subheader("📋 Extraction Strategy")
        extraction_strategy = st.selectbox(
            "Extraction Strategy", 
            ['single_test', 'multi_test', 'full_text_comparison'],
            index=['single_test', 'multi_test', 'full_text_comparison'].index(exp['extraction_strategy']),
            help="This determines the prompts configuration in the next step"
        )
        
        # Strategy preview/explanation
        st.info("💡 **Next Step**: You'll configure the prompts.")
        
        # Form submission
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.form_submit_button("➡️ Continue to Prompts Configuration", type="primary"):
                # Save basic info to session state
                st.session_state.experiment_config.update({
                    'name': name,
                    'description': description,
                    'researcher_name': researcher_name,
                    'cost_limit_usd': cost_limit,
                    'ai_model': ai_model,
                    'temperature': temperature,
                    'top_p': top_p,
                    'top_k': top_k,
                    'max_output_tokens': max_tokens,
                    'extraction_strategy': extraction_strategy
                })
                st.session_state.config_step = 2
                st.rerun()
        
        with col2:
            if st.form_submit_button("❌ Cancel"):
                st.session_state.editing_experiment = None
                st.session_state.config_step = 1
                st.rerun()

def show_prompts_config_form(exp, editing_id):
    """Show Step 2: Strategy-Specific Prompts Configuration"""
    # Progress indicator
    st.progress(1.0, text="Step 2 of 2: Prompts Configuration")
    
    config = st.session_state.experiment_config
    strategy = config['extraction_strategy']
    
    with st.form("prompts_config_form"):
        st.subheader(f"📝 {strategy.replace('_', ' ').title()} Configuration")
        
        # Strategy explanation
        if strategy == 'full_text_comparison':
            st.info("💡 **Full Text Comparison Strategy**: This strategy compares entire case texts directly without extraction. The extraction prompt will not be used.")
        elif strategy == 'multi_test':
            st.info("💡 **Multi-Test Strategy**: This strategy can extract multiple legal tests from a single case. The output will be structured as an array of test objects.")
        else:
            st.info("💡 **Single Test Strategy**: This strategy extracts one primary legal test per case.")
        
        # System instruction (always shown)
        system_instruction = st.text_area("System Instruction", value=config.get('system_instruction', exp['system_instruction']), height=100)
        
        # Strategy-specific prompts
        col1, col2 = st.columns(2)
        
        with col1:
            if strategy != 'full_text_comparison':
                extraction_prompt = st.text_area("Extraction Prompt", value=config.get('extraction_prompt', exp['extraction_prompt']), height=200,
                                               help="Custom prompt for legal test extraction (leave empty to use default)")
                
                # Show extraction schema
                with st.expander("📄 Extraction Structured Output Schema"):
                    st.markdown("**Gemini will be instructed to return JSON with these fields:**")
                    
                    if strategy == 'multi_test':
                        # Multi-test schema with arrays
                        st.code('''{"legal_tests": [
    {
        "legal_test": "string - The legal test extracted from the case",
        "passages": "string - The paragraphs (e.g., paras. x, y-z) or pages where the test is found",
        "test_novelty": "enum - One of: new test, major change in existing test, minor change in existing test, application of existing test, no substantive discussion"
    }
    // Additional test objects if multiple tests found
  ]}
''', language="json")
                    else:
                        # Single test schema
                        st.code('''{"legal_test": "string - The legal test extracted from the case",
 "passages": "string - The paragraphs (e.g., paras. x, y-z) or pages where the test is found",
 "test_novelty": "enum - One of: new test, major change in existing test, minor change in existing test, application of existing test, no substantive discussion"}
''', language="json")
            else:
                # Show info for full_text_comparison
                st.info("💡 Full text comparison strategy compares entire case texts directly without extraction.")
                extraction_prompt = ""  # No extraction prompt needed
        
        with col2:
            comparison_prompt = st.text_area("Comparison Prompt", value=config.get('comparison_prompt', exp['comparison_prompt']), height=200,
                                           help="Custom prompt for test comparison (leave empty to use default)")
            
            # Show comparison schema
            with st.expander("⚖️ Comparison Structured Output Schema"):
                st.markdown("**Gemini will be instructed to return JSON with these fields:**")
                st.code('''{"more_rule_like_test": "enum - Either 'Test A' or 'Test B'",
 "reasoning": "string - Clear reasoning for why the chosen test is more rule-like, referring to cases as Test A and Test B only"}
''', language="json")
        
        # Form submission
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.form_submit_button("⬅️ Back to Basic Info"):
                st.session_state.config_step = 1
                st.rerun()
        
        with col2:
            if st.form_submit_button("💾 Save Experiment", type="primary"):
                # Combine all config and save
                final_config = {
                    **config,
                    'system_instruction': system_instruction,
                    'extraction_prompt': extraction_prompt if strategy != 'full_text_comparison' else '',
                    'comparison_prompt': comparison_prompt,
                    'status': 'draft'
                }
                
                # Validate required fields
                if not final_config['name'] or not final_config['name'].strip():
                    st.error("Experiment name is required!")
                else:
                    saved_experiment_id = save_experiment(editing_id, final_config['name'], final_config['description'], 
                                     final_config['researcher_name'], final_config['status'], final_config['ai_model'], 
                                     final_config['temperature'], final_config['top_p'], final_config['top_k'], 
                                     final_config['max_output_tokens'], final_config['extraction_strategy'], 
                                     final_config['extraction_prompt'], final_config['comparison_prompt'], 
                                     final_config['system_instruction'], final_config['cost_limit_usd'])
                    
                    if saved_experiment_id:
                        st.success("Experiment saved successfully!")
                        # Clear session state and navigate to the experiment detail page
                        st.session_state.editing_experiment = None
                        st.session_state.config_step = 1
                        if 'experiment_config' in st.session_state:
                            del st.session_state.experiment_config
                        # Navigate to the experiment detail page
                        st.session_state.page_navigation = "Experiment Detail"
                        st.session_state.selected_page = "Experiment Detail"
                        st.session_state.selected_experiment = saved_experiment_id
                        st.rerun()
        
        with col3:
            if editing_id and st.form_submit_button("📋 Clone Experiment"):
                # Create a copy of the experiment
                base_name = config['name'] if config['name'] and config['name'].strip() else "Unnamed_Experiment"
                new_name = f"{base_name}_copy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                final_config = {
                    **config,
                    'system_instruction': system_instruction,
                    'extraction_prompt': extraction_prompt if strategy != 'full_text_comparison' else '',
                    'comparison_prompt': comparison_prompt,
                    'status': 'draft'
                }
                
                cloned_experiment_id = save_experiment(None, new_name, final_config['description'], 
                                 final_config['researcher_name'], final_config['status'], final_config['ai_model'], 
                                 final_config['temperature'], final_config['top_p'], final_config['top_k'], 
                                 final_config['max_output_tokens'], final_config['extraction_strategy'], 
                                 final_config['extraction_prompt'], final_config['comparison_prompt'], 
                                 final_config['system_instruction'], final_config['cost_limit_usd'])
                
                if cloned_experiment_id:
                    st.success(f"Experiment cloned as '{new_name}'!")
                    # Clear session state and navigate to the cloned experiment's detail page
                    st.session_state.editing_experiment = None
                    st.session_state.config_step = 1
                    if 'experiment_config' in st.session_state:
                        del st.session_state.experiment_config
                    # Navigate to the cloned experiment's detail page
                    st.session_state.page_navigation = "Experiment Detail"
                    st.session_state.selected_page = "Experiment Detail"
                    st.session_state.selected_experiment = cloned_experiment_id
                    st.rerun()

def save_experiment(experiment_id, name, description, researcher_name, status, ai_model, temperature, top_p, 
                   top_k, max_tokens, extraction_strategy, extraction_prompt, 
                   comparison_prompt, system_instruction, cost_limit):
    """Save experiment configuration to database"""
    try:
        if experiment_id:
            # Update existing experiment
            execute_sql("""
                UPDATE v2_experiments SET 
                    name = ?, description = ?, researcher_name = ?, ai_model = ?, temperature = ?,
                    top_p = ?, top_k = ?, max_output_tokens = ?, extraction_strategy = ?,
                    extraction_prompt = ?, comparison_prompt = ?, system_instruction = ?,
                    cost_limit_usd = ?, modified_date = CURRENT_TIMESTAMP
                WHERE experiment_id = ?
            """, (name, description, researcher_name, ai_model, temperature, top_p, top_k, max_tokens,
                  extraction_strategy, extraction_prompt, comparison_prompt, system_instruction,
                  cost_limit, experiment_id))
            return experiment_id
        else:
            # Create new experiment
            execute_sql("""
                INSERT INTO v2_experiments (name, description, researcher_name, status, ai_model, temperature, top_p,
                                       top_k, max_output_tokens, extraction_strategy, extraction_prompt,
                                       comparison_prompt, system_instruction, cost_limit_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, description, researcher_name, 'draft', ai_model, temperature, top_p, top_k, max_tokens,
                  extraction_strategy, extraction_prompt, comparison_prompt, system_instruction, cost_limit))
            
            # Get the ID of the newly created experiment
            new_experiment_id = execute_sql("""
                SELECT experiment_id FROM v2_experiments 
                WHERE name = ? AND researcher_name = ? AND created_date = (
                    SELECT MAX(created_date) FROM v2_experiments WHERE name = ? AND researcher_name = ?
                )
            """, (name, researcher_name, name, researcher_name), fetch=True)
            
            if new_experiment_id:
                experiment_id = new_experiment_id[0][0]
            else:
                return None
        
        # Clear caches after modifying experiments
        get_experiments_list.clear()
        _get_experiment_detail.clear()
        
        return experiment_id
    except Exception as e:
        st.error(f"Error saving experiment: {e}")
        return None

def show_case_management():
    """Show experiment case selection interface"""
    st.header("📚 Experiment Case Selection")
    st.markdown("*Select specific cases from the database to include in all experiments*")
    
    # Get database counts
    total_cases, selected_cases, tests_count, comparisons_count, validated_count = get_database_counts()
    
    # 1. Data Management (moved to top)
    with st.expander("1️⃣ **Data Management**", expanded=False):
        # Admin password protection
        admin_password = st.text_input("🔐 Admin Password (required for data operations)", 
                                     type="password", key="admin_password_dashboard")
        is_admin = admin_password == "scc2024admin"
        
        if not is_admin:
            st.warning("⚠️ Admin password required to access data loading and clearing functions.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📤 Upload Data")
                uploaded_file = st.file_uploader("Choose a Parquet file", type="parquet", 
                                               disabled=not is_admin)
                if uploaded_file is not None and is_admin:
                    # Batch skip option
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        if st.button("🚀 Load Data", type="primary"):
                            with st.spinner("Loading data..."):
                                load_data_from_parquet(uploaded_file)
                                st.rerun()  # Refresh to update metrics
                    
                    with col_b:
                        start_batch = st.number_input("Start Batch", min_value=1, value=1, 
                                                    step=1, help="Skip to batch number (100 cases per batch)")
                        if st.button("⏭️ Load from Batch", type="secondary"):
                            with st.spinner(f"Loading from batch {start_batch}..."):
                                load_data_from_parquet(uploaded_file, start_batch=start_batch)
                                st.rerun()  # Refresh to update metrics
            
            with col2:
                st.subheader("🗑️ Database Management")
                
                # Clear all data button
                if st.button("🗑️ Clear All Data", disabled=not is_admin, type="secondary"):
                    if st.checkbox("I understand this will delete ALL data", key="confirm_clear_all"):
                        if clear_database():
                            st.rerun()
                
                st.divider()
                
                # Clear selected cases button
                if st.button("🎯 Clear Selected Cases Only", disabled=not is_admin, type="secondary"):
                    if st.checkbox("I understand this will clear experiment case selection", key="confirm_clear_selected"):
                        if clear_selected_cases():
                            st.rerun()
                
                st.caption("Clear Selected Cases removes experiment selection & Bradley-Terry structure while preserving main case database")
    
    # 2. Database Overview
    with st.expander("2️⃣ **Database Overview**", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Cases Available", f"{total_cases:,}")
        with col2:
            st.metric("Cases Selected for Experiments", f"{selected_cases:,}")
        with col3:
            st.metric("Extracted Tests", f"{tests_count:,}")
        with col4:
            st.metric("Comparisons Made", f"{comparisons_count:,}")
        
        if selected_cases > 0:
            st.info(f"✅ **Experiment Dataset:** {selected_cases} cases selected. All experiments will run on these cases.")
        else:
            st.warning("⚠️ **No cases selected yet.** Select cases below to create your experiment dataset.")
    
    # 3. Case Selection for Experiments
    with st.expander("3️⃣ **Case Selection for Experiments**", expanded=True):
        # Show currently selected cases
        selected_cases_df = get_experiment_selected_cases()
        if not selected_cases_df.empty:
            with st.expander(f"📋 Currently Selected Cases ({len(selected_cases_df)})"):
                st.dataframe(
                    selected_cases_df[['case_name', 'citation', 'decision_year', 'area_of_law', 'selected_date']],
                    use_container_width=True
                )
        
        # Add new cases interface
        if total_cases > selected_cases:
            st.subheader("➕ Add Cases to Experiments")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Filters:**")
                
                # Get available cases for filter options
                sample_cases = get_available_cases_for_selection(limit=1000)
                
                if not sample_cases.empty:
                    # Year range filter
                    min_year = int(sample_cases['decision_year'].min())
                    max_year = int(sample_cases['decision_year'].max())
                    year_filter = st.slider("Year Range", min_year, max_year, (min_year, max_year), key="experiment_year_filter")
                    
                    # Area of law filter
                    unique_areas = sample_cases['area_of_law'].dropna().unique()
                    area_filter = st.multiselect("Areas of Law (optional)", unique_areas)
                    
                    # Number to select
                    available_count = len(get_available_cases_for_selection(
                        year_range=year_filter if year_filter != (min_year, max_year) else None,
                        areas=area_filter if area_filter else None
                    ))
                    
                    # Calculate suggested values divisible by 15
                    max_selectable = min(available_count, 100)
                    suggested_max = (max_selectable // 15) * 15
                    suggested_default = min(15, suggested_max) if suggested_max > 0 else 15
                    
                    num_to_select = st.number_input(
                        f"Number of cases to randomly select (must be divisible by 15)",
                        min_value=15, 
                        max_value=max_selectable, 
                        value=suggested_default,
                        step=15,
                        help=f"{available_count} cases available with current filters. Bradley-Terry analysis requires blocks of 15 cases (12 core + 3 bridge)."
                    )
                    
                    # Validation for multiples of 15
                    if num_to_select % 15 != 0:
                        st.error(f"⚠️ Number of cases must be divisible by 15 for Bradley-Terry block structure. Current: {num_to_select}, remainder: {num_to_select % 15}")
                        st.info(f"💡 Suggested values: {', '.join(str(i) for i in range(15, max_selectable + 1, 15) if i <= max_selectable)[:50]}...")
                        selection_valid = False
                    else:
                        blocks_needed = num_to_select // 15
                        st.success(f"✅ Valid selection: {num_to_select} cases = {blocks_needed} block{'s' if blocks_needed != 1 else ''} of 15 cases each")
                        selection_valid = True
            
            with col2:
                st.write("**Preview:**")
                
                if not sample_cases.empty:
                    # Get preview of available cases
                    preview_cases = get_available_cases_for_selection(
                        year_range=year_filter if year_filter != (min_year, max_year) else None,
                        areas=area_filter if area_filter else None,
                        limit=5
                    )
                    
                    if not preview_cases.empty:
                        st.dataframe(
                            preview_cases[['case_name', 'citation', 'decision_year', 'area_of_law']],
                            use_container_width=True
                        )
                        st.caption(f"Preview of {len(preview_cases)} cases (total available: {available_count})")
                    else:
                        st.warning("No cases available with current filters")
            
            # Add cases button (only enabled if selection is valid)
            button_disabled = not selection_valid if 'selection_valid' in locals() else num_to_select % 15 != 0
            button_label = "🎲 Randomly Select and Add Cases" if not button_disabled else "❌ Invalid Selection (Must be divisible by 15)"
            
            if st.button(button_label, type="primary", disabled=button_disabled):
                if num_to_select > 0 and num_to_select % 15 == 0:
                    with st.spinner(f"Randomly selecting {num_to_select} cases..."):
                        # Get random cases
                        new_cases = get_available_cases_for_selection(
                            year_range=year_filter if year_filter != (min_year, max_year) else None,
                            areas=area_filter if area_filter else None,
                            limit=num_to_select
                        )
                        
                        if not new_cases.empty:
                            # Add to experiments
                            success_count, duplicate_count = add_cases_to_experiments(new_cases['case_id'].tolist())
                            
                            if success_count > 0:
                                st.success(f"✅ Successfully added {success_count} cases to experiments!")
                                
                                # Check if we should generate/update Bradley-Terry structure
                                total_selected = len(get_experiment_selected_cases())
                                if total_selected % 15 == 0:
                                    with st.spinner("Generating Bradley-Terry block structure..."):
                                        success, message = generate_bradley_terry_structure(force_regenerate=True)
                                        if success:
                                            st.success(f"🎯 {message}")
                                            
                                            # Show block summary
                                            block_summary = get_block_summary()
                                            if block_summary:
                                                st.info(f"📊 Structure: {block_summary['total_blocks']} blocks, {block_summary['total_core_cases']} core cases, {block_summary['total_bridge_cases']} bridge cases")
                                        else:
                                            st.warning(f"⚠️ Bradley-Terry structure generation failed: {message}")
                                
                                # Show added cases
                                with st.expander("📋 Cases Added"):
                                    st.dataframe(
                                        new_cases[['case_name', 'citation', 'decision_year', 'area_of_law']],
                                        use_container_width=True
                                    )
                                
                                st.rerun()
                            else:
                                st.error("Failed to add cases to experiments")
                        else:
                            st.error("No cases found matching the criteria")
                else:
                    st.error("Please select at least 1 case")
        else:
            st.info("All available cases have been selected for experiments")

def show_experiment_comparison():
    """Show cross-experiment comparison interface"""
    st.header("📈 Experiment Comparison")
    st.markdown("*Compare methodology effectiveness across different experimental configurations*")
    
    # Get experiments with results
    experiments_with_results = execute_sql("""
        SELECT 
            e.experiment_id, 
            e.name, 
            e.ai_model, 
            e.temperature, 
            e.extraction_strategy,
            e.modified_date,
            COUNT(er.run_id) as run_count,
            SUM(er.tests_extracted) as total_tests,
            SUM(er.comparisons_completed) as total_comparisons,
            AVG(er.total_cost_usd) as avg_cost,
            MAX(er.run_date) as last_run
        FROM v2_experiments e
        LEFT JOIN v2_experiment_runs er ON e.experiment_id = er.experiment_id
        WHERE e.status IN ('active', 'completed') 
        GROUP BY e.experiment_id, e.name, e.ai_model, e.temperature, e.extraction_strategy, e.modified_date
        HAVING COUNT(er.run_id) > 0
        ORDER BY e.modified_date DESC
    """, fetch=True)
    
    if not experiments_with_results:
        st.info("No experiments with results available for comparison. Run some experiments first!")
        return
    
    # Convert to DataFrame for easier handling
    columns = ['experiment_id', 'name', 'ai_model', 'temperature', 'extraction_strategy', 
               'modified_date', 'run_count', 'total_tests', 'total_comparisons', 'avg_cost', 'last_run']
    pd = _get_pandas()
    exp_df = pd.DataFrame(experiments_with_results, columns=columns)
    
    # Experiment Selection
    st.subheader("🎯 Select Experiments to Compare")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Multi-select with experiment details
        exp_options = {}
        for _, exp in exp_df.iterrows():
            label = f"{exp['name']} ({exp['ai_model']}, temp: {exp['temperature']}, {exp['extraction_strategy']})"
            exp_options[label] = exp['experiment_id']
        
        selected_experiments = st.multiselect(
            "Choose experiments to compare:",
            exp_options.keys(),
            help="Select 2 or more experiments to compare their performance"
        )
    
    with col2:
        if len(selected_experiments) >= 2:
            st.success(f"✅ {len(selected_experiments)} experiments selected")
            show_comparison = st.button("📊 Generate Comparison", type="primary")
        else:
            st.warning("Select at least 2 experiments")
            show_comparison = False
    
    if show_comparison and len(selected_experiments) >= 2:
        # Get selected experiment IDs
        selected_ids = [exp_options[exp] for exp in selected_experiments]
        
        # Performance Comparison
        st.subheader("⚡ Performance Comparison")
        
        # Create comparison metrics
        comparison_data = []
        for exp_id in selected_ids:
            exp_info = exp_df[exp_df['experiment_id'] == exp_id].iloc[0]
            
            # Get detailed statistics (placeholder for now)
            comparison_data.append({
                'Experiment': exp_info['name'],
                'Model': exp_info['ai_model'],
                'Temperature': exp_info['temperature'],
                'Strategy': exp_info['extraction_strategy'],
                'Total Tests': exp_info['total_tests'],
                'Total Comparisons': exp_info['total_comparisons'],
                'Avg Cost ($)': f"${exp_info['avg_cost']:.2f}" if exp_info['avg_cost'] else "N/A",
                'Run Count': exp_info['run_count'],
                'Tests per Run': exp_info['total_tests'] / exp_info['run_count'] if exp_info['run_count'] > 0 else 0
            })
        
        pd = _get_pandas()
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visual comparisons
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Tests Extracted")
            chart_data = comparison_df.set_index('Experiment')['Total Tests']
            st.bar_chart(chart_data)
        
        with col2:
            st.subheader("💰 Cost Comparison")
            # Convert cost back to numeric for charting
            cost_data = comparison_df.copy()
            cost_data['Cost_Numeric'] = cost_data['Avg Cost ($)'].str.replace('$', '').str.replace('N/A', '0').astype(float)
            chart_data = cost_data.set_index('Experiment')['Cost_Numeric']
            st.bar_chart(chart_data)
        
        # Model Performance Analysis
        st.subheader("🧠 Model Performance Analysis")
        
        # Group by model type
        model_performance = comparison_df.groupby('Model').agg({
            'Total Tests': 'sum',
            'Total Comparisons': 'sum',
            'Tests per Run': 'mean'
        }).round(2)
        
        st.write("**Performance by AI Model:**")
        st.dataframe(model_performance, use_container_width=True)
        
        # Temperature Analysis
        st.subheader("🌡️ Temperature Impact Analysis")
        
        temp_analysis = comparison_df.groupby('Temperature').agg({
            'Total Tests': 'mean',
            'Tests per Run': 'mean'
        }).round(2)
        
        st.write("**Performance by Temperature Setting:**")
        st.dataframe(temp_analysis, use_container_width=True)
        
        # Statistical Significance Testing (Placeholder)
        st.subheader("📈 Statistical Analysis")
        
        st.info("**Coming Soon:** Statistical significance testing between experiments")
        st.write("Future features will include:")
        st.write("- Bradley-Terry score comparisons")
        st.write("- Confidence intervals for performance metrics")
        st.write("- ANOVA testing for model differences")
        st.write("- Cost-effectiveness ratios")
        st.write("- Reproducibility metrics")
        
        # Export Comparison Results
        st.subheader("📥 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export comparison table
            csv_data = comparison_df.to_csv(index=False)
            st.download_button(
                label="📊 Export Comparison Table",
                data=csv_data,
                file_name=f"experiment_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export model performance summary
            model_csv = model_performance.to_csv()
            st.download_button(
                label="🧠 Export Model Analysis",
                data=model_csv,
                file_name=f"model_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Experiment History and Trends
    st.subheader("📅 Experiment History")
    
    if not exp_df.empty:
        # Timeline visualization
        st.write("**Recent Experiment Activity:**")
        
        # Convert last_run to datetime for plotting
        pd = _get_pandas()
        exp_df['last_run_date'] = pd.to_datetime(exp_df['last_run'])
        recent_activity = exp_df.sort_values('last_run_date', ascending=False).head(10)
        
        # Display recent activity
        for _, exp in recent_activity.iterrows():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.write(f"**{exp['name']}**")
                st.caption(f"{exp['ai_model']} | {exp['extraction_strategy']}")
            
            with col2:
                st.metric("Tests", int(exp['total_tests']))
            
            with col3:
                st.metric("Runs", int(exp['run_count']))
            
            with col4:
                last_run_str = exp['last_run'][:10] if exp['last_run'] else "N/A"
                st.write(f"Last: {last_run_str}")
            
            st.divider()
    
    else:
        st.info("No experiment history available.")

# Cache experiments list for 5 minutes (longer cache for startup performance)
@st.cache_data(ttl=300)
def get_experiments_list():
    """Get list of experiments for navigation (reuses overview data for efficiency)"""
    try:
        # Reuse the cached overview data to avoid duplicate queries
        experiments_data = get_experiments_overview_data()
        
        if experiments_data:
            # Extract only the fields needed for navigation
            return [{'id': exp[0], 'name': exp[1], 'status': exp[3]} for exp in experiments_data]
        return []
    except Exception as e:
        st.error(f"Error loading experiments: {e}")
        return []

def show_sidebar_navigation():
    """Display hierarchical sidebar navigation"""
    # Initialize navigation state
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "Library Overview"  # Default to Library Overview
    if 'selected_experiment' not in st.session_state:
        st.session_state.selected_experiment = None
    
    # Inject cached CSS for compact application-like sidebar
    inject_sidebar_css()
    
    # Navigation section header
    st.sidebar.markdown("**📋 Navigation**")
    
    # 1. Create New Experiment (At the top)
    if st.sidebar.button("➕ Create New Experiment", use_container_width=True, type="primary"):
        st.session_state.selected_page = "Create Experiment"
        st.session_state.selected_experiment = None
        st.session_state.editing_experiment = None  # Clear edit state
        st.rerun()
    
    # Small separator
    st.sidebar.markdown("")
    
    # 2. Experiment Library (Expandable)
    with st.sidebar.expander("🧪 Experiment Library", expanded=True):
        # Overview option
        is_active = st.session_state.selected_page == "Library Overview"
        button_kwargs = {"use_container_width": True}
        if is_active:
            button_kwargs["type"] = "secondary"
        
        if st.button("📊 Library Overview", **button_kwargs):
            st.session_state.selected_page = "Library Overview"
            st.session_state.selected_experiment = None
            st.rerun()
            st.session_state.selected_page = "Library Overview"
            st.session_state.selected_experiment = None
            st.rerun()
        
        # Individual experiments - show by default
        experiments = get_experiments_list()
        if experiments:
            st.markdown("*Experiments:*")
            for exp in experiments:
                status_emoji = {
                    'draft': '🟡',
                    'in_progress': '🟠', 
                    'complete': '🟢',
                    'archived': '⚫'
                }.get(exp['status'], '⚪')
                
                # Add experiment number and truncate long names for better UI
                display_name = exp['name'][:20] + "..." if len(exp['name']) > 20 else exp['name']
                button_label = f"{status_emoji} #{exp['id']} {display_name}"
                is_selected = (st.session_state.selected_page == "Experiment Detail" and 
                             st.session_state.selected_experiment == exp['id'])
                
                button_kwargs = {"use_container_width": True, "key": f"exp_{exp['id']}"}
                if is_selected:
                    button_kwargs["type"] = "secondary"
                
                if st.button(button_label, **button_kwargs):
                    st.session_state.selected_page = "Experiment Detail"
                    st.session_state.selected_experiment = exp['id']
                    st.rerun()
        else:
            st.caption("No experiments found")
    
    # Small separator
    st.sidebar.markdown("")
    
    # 3. Experiment Comparison
    is_active = st.session_state.selected_page == "Comparison"
    button_kwargs = {"use_container_width": True}
    if is_active:
        button_kwargs["type"] = "secondary"
    
    if st.sidebar.button("📈 Experiment Comparison", **button_kwargs):
        st.session_state.selected_page = "Comparison"
        st.session_state.selected_experiment = None
        st.rerun()
    
    # Small separator
    st.sidebar.markdown("")
    
    # 4. Cases (Blue button)
    is_active = st.session_state.selected_page == "Cases"
    button_kwargs = {"use_container_width": True}
    if is_active:
        button_kwargs["type"] = "secondary"
    
    if st.sidebar.button("📚 View/Add Cases to Experiments", **button_kwargs):
        st.session_state.selected_page = "Cases"
        st.session_state.selected_experiment = None
        st.rerun()
    
    # Small separator
    st.sidebar.markdown("")
    
    # 5. API Key Management (Collapsible)
    # Check if API key is required (from execution buttons)
    api_key_required = st.session_state.get('api_key_required', False)
    
    # Force expand if API key is required
    with st.sidebar.expander("🔑 API Key", expanded=api_key_required):
        # Check current API key status
        api_key_status = "Not set"
        if 'api_key' in st.session_state and st.session_state.api_key:
            api_key_status = f"Set (ends with ...{st.session_state.api_key[-4:]})"
        
        # Show highlighted status if API key is required
        if api_key_required:
            # Add pulsing animation CSS
            st.markdown("""
            <style>
            @keyframes pulse {
                0% { border-color: #ffeaa7; }
                50% { border-color: #ff6b6b; }
                100% { border-color: #ffeaa7; }
            }
            .api-key-required {
                animation: pulse 2s infinite;
                background-color: #fff3cd;
                padding: 10px;
                border-radius: 5px;
                border: 2px solid #ffeaa7;
            }
            </style>
            <div class="api-key-required">
                <strong>⚠️ API Key Required!</strong><br>
                Please enter your API key to execute experiments.
            </div>
            """, unsafe_allow_html=True)
            st.write("")
        
        st.write(f"**Status:** {api_key_status}")
        
        # API Key input
        api_key_input = st.text_input("Enter API Key:", type="password", key="api_key_input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save", use_container_width=True, key="save_api_key"):
                if api_key_input.strip():
                    st.session_state.api_key = api_key_input.strip()
                    # Clear the required flag once saved
                    if 'api_key_required' in st.session_state:
                        del st.session_state.api_key_required
                    st.success("API key saved!")
                    st.rerun()
                else:
                    st.error("Please enter an API key")
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True, key="clear_api_key"):
                if 'api_key' in st.session_state:
                    del st.session_state.api_key
                st.success("API key cleared!")
                st.rerun()
        
        st.caption("API key is stored only for this session")

# Cache experiment details for 60 seconds
@st.cache_data(ttl=60)
def _get_experiment_detail(experiment_id):
    """Get experiment details from database"""
    exp_data = execute_sql(
        "SELECT experiment_id, name, description, researcher_name, status, ai_model, temperature, top_p, top_k, max_output_tokens, extraction_strategy, extraction_prompt, comparison_prompt, system_instruction, cost_limit_usd, created_date, modified_date, created_by FROM v2_experiments WHERE experiment_id = ?", 
        (experiment_id,), 
        fetch=True
    )
    
    if not exp_data:
        return None
    
    # Convert row to dictionary, handling any data type conversion needed
    row = exp_data[0]
    return {
        'experiment_id': row[0],
        'name': row[1], 
        'description': row[2],
        'researcher_name': row[3],
        'status': row[4],
        'ai_model': row[5],
        'temperature': float(row[6]) if row[6] is not None else 0.0,
        'top_p': float(row[7]) if row[7] is not None else 1.0,
        'top_k': int(row[8]) if row[8] is not None else 40,
        'max_output_tokens': int(row[9]) if row[9] is not None else 8192,
        'extraction_strategy': row[10],
        'extraction_prompt': row[11],
        'comparison_prompt': row[12], 
        'system_instruction': row[13],
        'cost_limit_usd': float(row[14]) if row[14] is not None else 100.0,
        'created_date': row[15],
        'modified_date': row[16],
        'created_by': row[17]
    }

def show_experiment_detail(experiment_id):
    """Show details for a specific experiment with comprehensive cost analysis"""
    try:
        exp = _get_experiment_detail(experiment_id)
        
        if not exp:
            st.error("Experiment not found!")
            return
        
        st.header(f"Experiment #{exp['experiment_id']}: {exp['name']}")
        
        # Status with color
        status_colors = {
            'draft': '🟡 Draft',
            'in_progress': '🟠 In Progress',
            'complete': '🟢 Complete',
            'archived': '⚫ Archived'
        }
        st.markdown(f"**Status:** {status_colors.get(exp['status'], exp['status'])}")
        
        # Get shared data for cost calculations using cached functions
        try:
            stats = get_case_statistics()
            selected_cases_count = stats['selected_cases_count']
            total_cases_count = stats['total_cases_count']
            avg_selected_case_length = stats['avg_selected_case_length']
            avg_all_case_length = stats['avg_all_case_length']
        except Exception as e:
            st.warning(f"Could not load case statistics: {e}")
            selected_cases_count = 0
            total_cases_count = 0
            avg_selected_case_length = 52646.0
            avg_all_case_length = 52646.0
        
        # Calculate comprehensive cost data
        n_cases = selected_cases_count
        required_comparisons = calculate_bradley_terry_comparisons(n_cases)
        
        # Bradley-Terry parameters
        block_size = 15
        core_cases_per_block = 12
        comparisons_per_block = 105
        
        # Get run statistics with safe handling
        try:
            runs_data = execute_sql("""
                SELECT 
                    1 as run_count,
                    (SELECT COUNT(*) FROM v2_experiment_extractions WHERE experiment_id = ?) as total_tests,
                    (SELECT COUNT(*) FROM v2_experiment_comparisons WHERE experiment_id = ?) as total_comparisons,
                    (SELECT COALESCE(SUM(api_cost_usd), 0) FROM v2_experiment_extractions WHERE experiment_id = ?) + 
                    (SELECT COALESCE(SUM(api_cost_usd), 0) FROM v2_experiment_comparisons WHERE experiment_id = ?) as total_cost
            """, (experiment_id, experiment_id, experiment_id, experiment_id), fetch=True)
            
            if runs_data and runs_data[0]:
                row = runs_data[0]
                # Handle both tuple and row object access
                if hasattr(row, '__getitem__'):
                    total_tests = int(row[1]) if row[1] is not None else 0
                    total_comparisons = int(row[2]) if row[2] is not None else 0
                    total_cost = float(row[3]) if row[3] is not None else 0.0
                else:
                    total_tests = 0
                    total_comparisons = 0 
                    total_cost = 0.0
            else:
                total_tests = 0
                total_comparisons = 0
                total_cost = 0.0
        except Exception as e:
            st.warning(f"Could not load run statistics: {e}")
            total_tests = 0
            total_comparisons = 0
            total_cost = 0.0
        
        # Layout in tabs for organization
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📋 Configuration & Stats", "💰 Cost Estimates", "🚀 Execution", "📄 Extractions", "⚖️ Comparisons", "📊 Results", "⚙️ Settings"])
        
        with tab1:
            # Configuration and Stats combined
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Basic Information")
                st.write(f"**Description:** {exp['description'] or 'No description'}")
                st.write(f"**Researcher:** {exp.get('researcher_name', 'Not specified')}")
                # Handle datetime objects properly
                created_date = exp['created_date']
                modified_date = exp['modified_date']
                
                if isinstance(created_date, str):
                    created_str = created_date[:10] if created_date else 'Unknown'
                else:
                    created_str = created_date.strftime('%Y-%m-%d') if created_date else 'Unknown'
                
                if isinstance(modified_date, str):
                    modified_str = modified_date[:10] if modified_date else 'Unknown'
                else:
                    modified_str = modified_date.strftime('%Y-%m-%d') if modified_date else 'Unknown'
                
                st.write(f"**Created:** {created_str}")
                st.write(f"**Modified:** {modified_str}")
                
                st.subheader("Current Progress")
                st.metric("Tests Extracted", f"{total_tests}/{n_cases}")
                st.metric("Comparisons Made", f"{total_comparisons}/{required_comparisons}")
                st.metric("Total Spent", f"${total_cost:.2f}")
                
                if n_cases > 0:
                    extraction_progress = total_tests / n_cases
                    st.progress(extraction_progress, text=f"Extraction Progress: {extraction_progress:.1%}")
                
                if required_comparisons > 0:
                    comparison_progress = total_comparisons / required_comparisons
                    st.progress(comparison_progress, text=f"Comparison Progress: {comparison_progress:.1%}")
            
            with col2:
                st.subheader("AI Configuration")
                st.write(f"**Model:** {exp['ai_model']}")
                st.write(f"**Temperature:** {exp['temperature']}")
                st.write(f"**Top P:** {exp.get('top_p', 1.0)}")
                st.write(f"**Top K:** {exp.get('top_k', 40)}")
                st.write(f"**Max Tokens:** {exp.get('max_output_tokens', 8192)}")
                st.write(f"**Strategy:** {exp['extraction_strategy']}")
                st.write(f"**Cost Limit:** ${exp['cost_limit_usd']}")
                
                # Show comparison strategy
                if n_cases <= block_size:
                    comparison_strategy = "Full pairwise"
                else:
                    blocks_needed = (n_cases + core_cases_per_block - 1) // core_cases_per_block
                    comparison_strategy = f"Bradley-Terry ({blocks_needed} blocks)"
                
                st.subheader("Comparison Strategy")
                st.write(f"**Method:** {comparison_strategy}")
                st.write(f"**Required Comparisons:** {required_comparisons:,}")
            
            # Add full prompts display
            st.subheader("🤖 AI Prompts & Configuration")
            
            # System instruction
            with st.expander("📋 System Instruction"):
                system_instruction = exp.get('system_instruction', '').strip()
                if not system_instruction:
                    system_instruction = "You are a helpful assistant that helps legal researchers analyze legal texts."
                st.code(system_instruction, language="text")
            
            # Extraction prompt
            with st.expander("📄 Full Extraction Prompt"):
                extraction_prompt = exp.get('extraction_prompt', '').strip()
                if not extraction_prompt:
                    # Load default from file
                    prompt_file = 'prompts/extractor_prompt.txt'
                    if os.path.exists(prompt_file):
                        with open(prompt_file, 'r') as f:
                            extraction_prompt = f.read()
                    else:
                        extraction_prompt = "Extract the main legal test from this case."
                
                st.markdown("**Custom Prompt:**")
                st.code(extraction_prompt, language="text")
                
                st.markdown("**Structured Output Schema:**")
                st.code('''{"legal_test": "string - The legal test extracted from the case",
 "passages": "string - The paragraphs (e.g., paras. x, y-z) or pages where the test is found",
 "test_novelty": "enum - One of: new test, major change in existing test, minor change in existing test, application of existing test, no substantive discussion"}
''', language="json")
            
            # Comparison prompt
            with st.expander("⚖️ Full Comparison Prompt"):
                comparison_prompt = exp.get('comparison_prompt', '').strip()
                if not comparison_prompt:
                    # Load default from file
                    prompt_file = 'prompts/comparator_prompt.txt'
                    if os.path.exists(prompt_file):
                        with open(prompt_file, 'r') as f:
                            comparison_prompt = f.read()
                    else:
                        comparison_prompt = "Compare these two legal tests and determine which is more rule-like."
                
                st.markdown("**Custom Prompt:**")
                st.code(comparison_prompt, language="text")
                
                st.markdown("**Structured Output Schema:**")
                st.code('''{"more_rule_like_test": "enum - Either 'Test A' or 'Test B'",
 "reasoning": "string - Clear reasoning for why the chosen test is more rule-like, referring to cases as Test A and Test B only"}
''', language="json")
        
        with tab2:
            st.subheader("💰 Comprehensive Cost Analysis")
            
            # Get model pricing
            model_pricing = GEMINI_MODELS.get(exp['ai_model'], {'input': 0.30, 'output': 2.50})
            
            # Calculate detailed cost parameters for this experiment's strategy
            cost_params = get_cost_calculation_params(n_cases, avg_selected_case_length, exp['extraction_strategy'])
            extraction_input_tokens = cost_params['extraction_input_tokens']
            extracted_test_tokens = cost_params['extracted_test_tokens']
            comparison_input_tokens = cost_params['comparison_input_tokens']
            comparison_output_tokens = cost_params['comparison_output_tokens']
            
            # Per-case extraction cost
            extraction_cost_per_case = (extraction_input_tokens / 1_000_000) * model_pricing['input'] + (extracted_test_tokens / 1_000_000) * model_pricing['output']
            
            # Per-pair comparison cost based on strategy
            comparison_cost_per_pair = (comparison_input_tokens / 1_000_000) * model_pricing['input'] + (comparison_output_tokens / 1_000_000) * model_pricing['output']
            
            # Pre-calculate all values based on strategy
            if exp['extraction_strategy'] == 'full_text_comparison':
                # No extraction cost for full text comparison
                sample_extraction_cost = 0
                sample_comparison_cost = required_comparisons * comparison_cost_per_pair
                sample_total_cost = sample_comparison_cost
            else:
                # Extraction + comparison costs
                sample_extraction_cost = n_cases * extraction_cost_per_case
                sample_comparison_cost = required_comparisons * comparison_cost_per_pair
                sample_total_cost = sample_extraction_cost + sample_comparison_cost
            
            remaining_extractions = max(0, n_cases - total_tests)
            remaining_comparisons = max(0, required_comparisons - total_comparisons)
            remaining_extraction_cost = remaining_extractions * extraction_cost_per_case
            remaining_comparison_cost = remaining_comparisons * comparison_cost_per_pair
            remaining_total_cost = remaining_extraction_cost + remaining_comparison_cost
            
            # For full DB estimates, use same strategy as experiment for fair comparison
            full_db_comparisons = calculate_bradley_terry_comparisons(total_cases_count)
            full_db_cost_params = get_cost_calculation_params(total_cases_count, avg_selected_case_length, exp['extraction_strategy'])
            full_db_comparison_cost_per_pair = (full_db_cost_params['comparison_input_tokens'] / 1_000_000) * model_pricing['input'] + (full_db_cost_params['comparison_output_tokens'] / 1_000_000) * model_pricing['output']
            
            if exp['extraction_strategy'] == 'full_text_comparison':
                # No extraction cost for full text comparison
                full_extraction_cost = 0
                full_comparison_cost = full_db_comparisons * full_db_comparison_cost_per_pair
                full_total_cost = full_comparison_cost
            else:
                # Extraction + comparison costs
                full_extraction_cost = total_cases_count * extraction_cost_per_case
                full_comparison_cost = full_db_comparisons * full_db_comparison_cost_per_pair
                full_total_cost = full_extraction_cost + full_comparison_cost
            efficiency_ratio = (sample_total_cost / full_total_cost) * 100 if full_total_cost > 0 else 0
            
            # 2x2 Grid Layout for perfect alignment
            # Top row
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Sample Cost Breakdown")
                st.write(f"**Selected Cases:** {n_cases:,}")
                st.write(f"**Required Comparisons:** {required_comparisons:,}")
                st.metric("Sample Extraction Cost", f"${sample_extraction_cost:.2f}")
                st.metric("Sample Comparison Cost", f"${sample_comparison_cost:.2f}")
                st.metric("Sample Total Cost", f"${sample_total_cost:.2f}")
            
            with col2:
                st.subheader("🌍 Full Database Estimates")
                st.write(f"**Total Cases in Database:** {total_cases_count:,}")
                st.write(f"**Required Comparisons:** {full_db_comparisons:,}")
                st.metric("Full DB Extraction Cost", f"${full_extraction_cost:.2f}")
                st.metric("Full DB Comparison Cost", f"${full_comparison_cost:.2f}")
                st.metric("Full DB Total Cost", f"${full_total_cost:.2f}")
            
            # Divider
            st.write("---")
            
            # Bottom row - aligned sections
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("🎯 Remaining Work")
                st.metric("Remaining Extraction Cost", f"${remaining_extraction_cost:.2f}")
                st.metric("Remaining Comparison Cost", f"${remaining_comparison_cost:.2f}")
                st.metric("Remaining Total Cost", f"${remaining_total_cost:.2f}")
            
            with col4:
                st.subheader("📊 Cost Efficiency")
                st.metric("Sample vs Full DB", f"{efficiency_ratio:.1f}%")
                
                # Bradley-Terry efficiency
                if total_cases_count > 15:
                    full_pairwise = (total_cases_count * (total_cases_count - 1)) // 2
                    bt_efficiency = ((full_pairwise - full_db_comparisons) / full_pairwise) * 100
                    st.metric("Bradley-Terry Savings", f"{bt_efficiency:.1f}%")
                else:
                    st.metric("Bradley-Terry Savings", "N/A")
            
            # Detailed cost calculation breakdown
            st.write("---")
            with st.expander("🔍 Detailed Cost Calculation"):
                st.write("**Token Calculations:**")
                st.write(f"- Average case length: {avg_selected_case_length:,.0f} characters")
                st.write(f"- Estimated tokens per case: {avg_selected_case_length / 4:,.0f} tokens")
                st.write(f"- System prompt tokens: {100}")
                st.write(f"- Extraction prompt tokens: {200}")
                st.write(f"- Comparison prompt tokens: {150}")
                st.write(f"- Total input tokens per case: {extraction_input_tokens:,.0f}")
                st.write(f"- Expected output tokens per case: {extracted_test_tokens}")
                
                st.write("**Model Pricing:**")
                st.write(f"- Model: {exp['ai_model']}")
                st.write(f"- Input cost: ${model_pricing['input']:.2f} per million tokens")
                st.write(f"- Output cost: ${model_pricing['output']:.2f} per million tokens")
                
                if exp['extraction_strategy'] == 'full_text_comparison':
                    st.write("**Per-Case Extraction Cost:**")
                    st.write("- **No extraction needed** (comparing full case texts directly)")
                    st.write("- **Total per case: $0.00**")
                else:
                    st.write("**Per-Case Extraction Cost:**")
                    input_cost = (extraction_input_tokens / 1_000_000) * model_pricing['input']
                    output_cost = (extracted_test_tokens / 1_000_000) * model_pricing['output']
                    st.write(f"- Input cost: ${input_cost:.4f}")
                    st.write(f"- Output cost: ${output_cost:.4f}")
                    st.write(f"- **Total per case: ${extraction_cost_per_case:.4f}**")
                
                st.write("**Per-Pair Comparison Cost:**")
                if exp['extraction_strategy'] == 'full_text_comparison':
                    st.write(f"- Strategy: Full text comparison (comparing entire case texts)")
                    st.write(f"- Input tokens per comparison: {comparison_input_tokens:,.0f} (2 × {avg_selected_case_length / 4:,.0f} + prompts)")
                else:
                    st.write(f"- Strategy: {exp['extraction_strategy']} (comparing extracted tests)")
                    st.write(f"- Input tokens per comparison: {comparison_input_tokens:,.0f} (2 × {extracted_test_tokens} + prompts)")
                comparison_input_cost = (comparison_input_tokens / 1_000_000) * model_pricing['input']
                comparison_output_cost = (comparison_output_tokens / 1_000_000) * model_pricing['output']
                st.write(f"- Input cost: ${comparison_input_cost:.4f}")
                st.write(f"- Output cost: ${comparison_output_cost:.4f}")
                st.write(f"- **Total per pair: ${comparison_cost_per_pair:.4f}**")
        
        with tab3:
            # Execution interface
            st.subheader("🚀 Run Experiment")
            
            # Status check
            if exp['status'] == 'archived':
                st.error("❌ This experiment is archived and cannot be executed.")
                st.stop()
            
            # Check if API key is required and show prominent notification
            if st.session_state.get('api_key_required', False):
                st.error("⚠️ **API Key Required!** Please set your API key in the sidebar to execute experiments.")
                st.markdown("""
                <div style="background-color: #f8d7da; padding: 15px; border-radius: 5px; border: 1px solid #f5c6cb; margin: 10px 0;">
                    <strong>🔑 How to set your API key:</strong><br>
                    1. Look at the sidebar on the left<br>
                    2. Click on "🔑 API Key" to expand it<br>
                    3. Enter your Gemini API key<br>
                    4. Click "💾 Save"
                </div>
                """, unsafe_allow_html=True)
            
            # Execution overview
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Extraction Phase**")
                if total_tests == 0:
                    st.write("🟡 Ready to extract legal tests from cases")
                    st.write(f"(0/{n_cases} complete)")
                elif total_tests < n_cases:
                    st.write(f"🟠 In progress: {total_tests}/{n_cases} cases processed")
                    st.write(f"({total_tests}/{n_cases} complete)")
                else:
                    st.write("🟢 All cases extracted")
                    st.write(f"({total_tests}/{n_cases} complete)")
                
                if st.button("▶️ Run Extraction", 
                           type="primary" if total_tests < n_cases else "secondary",
                           use_container_width=True,
                           disabled=(exp['status'] == 'archived')):
                    # Check if API key is set
                    if 'api_key' not in st.session_state or not st.session_state.api_key:
                        # Force expand sidebar and highlight API key section
                        st.session_state.api_key_required = True
                        st.error("⚠️ API Key must be entered to execute extractions. Please set your API key in the sidebar.")
                        st.rerun()
                    else:
                        # Update experiment status to in_progress
                        execute_sql("UPDATE v2_experiments SET status = 'in_progress', modified_date = CURRENT_TIMESTAMP WHERE experiment_id = ?", (experiment_id,))
                        
                        # Execute extraction
                        run_extraction_for_experiment(experiment_id)
                        st.rerun()
                    
            with col2:
                st.write("**Comparison Phase**")
                if total_comparisons == 0:
                    if total_tests == n_cases:
                        st.write("🟡 Ready to run pairwise comparisons")
                        st.write(f"(0/{required_comparisons} complete)")
                    else:
                        st.write("⏳ Waiting for extractions to complete")
                        st.write(f"(0/{required_comparisons} complete)")
                elif total_comparisons < required_comparisons:
                    st.write(f"🟠 In progress: {total_comparisons}/{required_comparisons} comparisons")
                    st.write(f"({total_comparisons}/{required_comparisons} complete)")
                else:
                    st.write("🟢 All comparisons completed")
                    st.write(f"({total_comparisons}/{required_comparisons} complete)")
                
                comparison_disabled = (exp['status'] == 'archived') or (total_tests < n_cases)
                if st.button("▶️ Run Comparisons", 
                           type="primary" if total_comparisons < required_comparisons and not comparison_disabled else "secondary",
                           use_container_width=True,
                           disabled=comparison_disabled):
                    # Check if API key is set
                    if 'api_key' not in st.session_state or not st.session_state.api_key:
                        # Force expand sidebar and highlight API key section
                        st.session_state.api_key_required = True
                        st.error("⚠️ API Key must be entered to execute comparisons. Please set your API key in the sidebar.")
                        st.rerun()
                    else:
                        # Update experiment status to in_progress
                        execute_sql("UPDATE v2_experiments SET status = 'in_progress', modified_date = CURRENT_TIMESTAMP WHERE experiment_id = ?", (experiment_id,))
                        
                        # Execute comparisons
                        run_comparisons_for_experiment(experiment_id)
                        st.rerun()
                
            # Quick status summary
            if n_cases > 0:
                st.write("---")
                st.write("**Quick Status Summary**")
                overall_progress = ((total_tests / n_cases) * 0.5 + (total_comparisons / required_comparisons) * 0.5) if required_comparisons > 0 else (total_tests / n_cases)
                st.progress(overall_progress, text=f"Overall Progress: {overall_progress:.1%}")
                
                if total_tests == n_cases and total_comparisons == required_comparisons:
                    st.success("🎉 Experiment completed! All extractions and comparisons are done.")
                    st.info("💡 Next steps: View results in the Extractions, Comparisons, and Results tabs.")
                elif total_cost >= exp['cost_limit_usd'] * 0.9:
                    st.warning(f"⚠️ Approaching cost limit: ${total_cost:.2f} / ${exp['cost_limit_usd']}")
                    st.info("💡 Consider increasing the cost limit in Settings or analyzing current results.")
        
        with tab4:
            # Extractions tab
            st.subheader("📄 Extracted Legal Tests")
            
            # Query extractions for this experiment
            extractions = execute_sql("""
                SELECT 
                    ee.extraction_id,
                    c.case_name,
                    c.citation,
                    ee.legal_test_name,
                    ee.legal_test_content,
                    ee.extraction_rationale,
                    ee.test_passages,
                    ee.test_novelty,
                    ee.rule_like_score,
                    ee.confidence_score,
                    ee.validation_status,
                    c.decision_url
                FROM v2_experiment_extractions ee
                JOIN v2_cases c ON ee.case_id = c.case_id
                WHERE ee.experiment_id = ?
                ORDER BY c.case_name
            """, (experiment_id,), fetch=True)
            
            if not extractions:
                st.info("No extractions found for this experiment yet. Run extractions first.")
            else:
                # Convert to DataFrame for easy display
                pd = _get_pandas()
                df = pd.DataFrame(extractions, columns=[
                    'extraction_id', 'case_name', 'citation', 'legal_test_name', 
                    'legal_test_content', 'extraction_rationale', 'test_passages', 
                    'test_novelty', 'rule_like_score', 'confidence_score', 
                    'validation_status', 'decision_url'
                ])
                
                # Add search, filter, and sort controls
                st.markdown("### 🔍 Search & Filter Controls")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    search_term = st.text_input(
                        "🔍 Search by case name",
                        placeholder="Type case name...",
                        key="extraction_search"
                    )
                
                with col2:
                    # Get unique validation statuses
                    unique_statuses = df['validation_status'].dropna().unique().tolist()
                    status_filter = st.selectbox(
                        "📋 Filter by status",
                        ["All"] + unique_statuses,
                        key="extraction_status_filter"
                    )
                
                with col3:
                    # Get unique test novelty types
                    unique_novelties = df['test_novelty'].dropna().unique().tolist()
                    novelty_filter = st.selectbox(
                        "🆕 Filter by novelty",
                        ["All"] + unique_novelties,
                        key="extraction_novelty_filter"
                    )
                
                # Sort controls
                col4, col5 = st.columns(2)
                
                with col4:
                    sort_by = st.selectbox(
                        "📊 Sort by",
                        ["Case Name", "Rule-like Score", "Confidence Score", "Status"],
                        key="extraction_sort_by"
                    )
                
                with col5:
                    sort_order = st.selectbox(
                        "🔄 Order",
                        ["Ascending", "Descending"],
                        key="extraction_sort_order"
                    )
                
                # Apply filters
                filtered_df = df.copy()
                
                # Search filter
                if search_term:
                    filtered_df = filtered_df[filtered_df['case_name'].str.contains(search_term, case=False, na=False)]
                
                # Status filter
                if status_filter != "All":
                    filtered_df = filtered_df[filtered_df['validation_status'] == status_filter]
                
                # Novelty filter
                if novelty_filter != "All":
                    filtered_df = filtered_df[filtered_df['test_novelty'] == novelty_filter]
                
                # Apply sorting
                sort_column_map = {
                    "Case Name": "case_name",
                    "Rule-like Score": "rule_like_score", 
                    "Confidence Score": "confidence_score",
                    "Status": "validation_status"
                }
                sort_column = sort_column_map[sort_by]
                ascending = sort_order == "Ascending"
                
                # Handle NaN values in sorting
                if sort_column in ["rule_like_score", "confidence_score"]:
                    filtered_df = filtered_df.sort_values(sort_column, ascending=ascending, na_position='last')
                else:
                    filtered_df = filtered_df.sort_values(sort_column, ascending=ascending)
                
                # Display results summary
                st.markdown("---")
                col_summary1, col_summary2 = st.columns(2)
                with col_summary1:
                    st.write(f"**Showing {len(filtered_df)} of {len(df)} extractions**")
                with col_summary2:
                    if len(filtered_df) > 0:
                        avg_rule_score = filtered_df['rule_like_score'].mean()
                        st.write(f"**Average Rule-like Score:** {avg_rule_score:.2f}" if not pd.isna(avg_rule_score) else "**Average Rule-like Score:** N/A")
                
                # Create display with filtered results
                for idx, row in filtered_df.iterrows():
                    # Create enhanced expander title with key info
                    rule_score_text = f" | Rule-like: {row['rule_like_score']:.2f}" if row['rule_like_score'] else ""
                    status_text = f" | {row['validation_status']}" if row['validation_status'] else ""
                    expander_title = f"**{row['case_name']}** ({row['citation']}){rule_score_text}{status_text}"
                    
                    with st.expander(expander_title, expanded=False):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write("**Legal Test:**")
                            st.write(row['legal_test_content'])
                            
                            if row['extraction_rationale']:
                                st.write("**AI Rationale:**")
                                st.write(row['extraction_rationale'])
                            
                            st.write("**Test Location:**")
                            st.write(row['test_passages'] if row['test_passages'] else "Not available")
                            
                            st.write("**Test Novelty:**")
                            st.write(row['test_novelty'] if row['test_novelty'] else "Not available")
                            
                        with col2:
                            st.metric("Rule-like Score", f"{row['rule_like_score']:.2f}" if row['rule_like_score'] else "N/A")
                            st.metric("Confidence", f"{row['confidence_score']:.2f}" if row['confidence_score'] else "N/A")
                            
                            # Status with color coding
                            status = row['validation_status']
                            if status == 'accurate':
                                st.success(f"✅ {status}")
                            elif status == 'inaccurate':
                                st.error(f"❌ {status}")
                            elif status == 'pending_review':
                                st.warning(f"⏳ {status}")
                            else:
                                st.write(f"**Status:** {status}")
                            
                            if row['decision_url']:
                                st.link_button("📖 View Case", row['decision_url'])
                            else:
                                st.write("**Case Link:** Not available")
        
        with tab5:
            # Comparisons tab
            st.subheader("⚖️ Pairwise Comparisons")
            
            # Query comparisons for this experiment
            comparisons = execute_sql("""
                SELECT 
                    ec.comparison_id,
                    c1.case_name as case_a_name,
                    c2.case_name as case_b_name,
                    c1.citation as case_a_citation,
                    c2.citation as case_b_citation,
                    ee1.legal_test_content as test_a,
                    ee2.legal_test_content as test_b,
                    winner_case.case_name as winner_case_name,
                    ec.comparison_rationale,
                    ec.confidence_score,
                    ec.human_validated,
                    ec.comparison_date,
                    ec.winner_id,
                    ee1.extraction_id as extraction_id_1,
                    ee2.extraction_id as extraction_id_2
                FROM v2_experiment_comparisons ec
                JOIN v2_experiment_extractions ee1 ON ec.extraction_id_1 = ee1.extraction_id
                JOIN v2_experiment_extractions ee2 ON ec.extraction_id_2 = ee2.extraction_id
                JOIN v2_cases c1 ON ee1.case_id = c1.case_id
                JOIN v2_cases c2 ON ee2.case_id = c2.case_id
                LEFT JOIN v2_experiment_extractions winner_ext ON ec.winner_id = winner_ext.extraction_id
                LEFT JOIN v2_cases winner_case ON winner_ext.case_id = winner_case.case_id
                WHERE ec.experiment_id = ?
                ORDER BY ec.comparison_date DESC
            """, (experiment_id,), fetch=True)
            
            if not comparisons:
                st.info("No comparisons found for this experiment yet. Run comparisons first.")
            else:
                # Process comparisons data for filtering and stats
                processed_comparisons = []
                test_a_wins = 0
                test_b_wins = 0
                no_winner = 0
                
                for idx, comp in enumerate(comparisons):
                    comparison_id, case_a_name, case_b_name, case_a_citation, case_b_citation, test_a, test_b, winner_case_name, comparison_rationale, confidence_score, human_validated, comparison_date, winner_id, extraction_id_1, extraction_id_2 = comp
                    
                    # Format the date
                    if comparison_date:
                        if isinstance(comparison_date, str):
                            formatted_date = comparison_date[:10]
                        else:
                            formatted_date = comparison_date.strftime('%Y-%m-%d')
                    else:
                        formatted_date = "Unknown"
                    
                    # Determine winner display
                    if winner_id == extraction_id_1:
                        winner_text = case_a_name
                        winner_label = "Test A"
                        test_a_wins += 1
                    elif winner_id == extraction_id_2:
                        winner_text = case_b_name
                        winner_label = "Test B"
                        test_b_wins += 1
                    else:
                        winner_text = "No winner determined"
                        winner_label = "N/A"
                        no_winner += 1
                    
                    processed_comparisons.append({
                        'idx': idx,
                        'comparison_id': comparison_id,
                        'case_a_name': case_a_name,
                        'case_b_name': case_b_name,
                        'case_a_citation': case_a_citation,
                        'case_b_citation': case_b_citation,
                        'test_a': test_a,
                        'test_b': test_b,
                        'winner_text': winner_text,
                        'winner_label': winner_label,
                        'comparison_rationale': comparison_rationale,
                        'confidence_score': confidence_score,
                        'human_validated': human_validated,
                        'formatted_date': formatted_date,
                        'comparison_date': comparison_date
                    })
                
                # Header with stats and controls
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**Total Comparisons:** {len(comparisons)}")
                    st.write(f"**Quick Stats:** Test A wins: {test_a_wins}, Test B wins: {test_b_wins}, No winner: {no_winner}")
                
                with col2:
                    # Search functionality
                    search_term = st.text_input("🔍 Search by case name", placeholder="Enter case name...")
                
                with col3:
                    # Filter by winner
                    winner_filter = st.selectbox(
                        "Filter by winner",
                        ["All", "Test A", "Test B", "No winner"]
                    )
                
                # Sort options
                sort_option = st.selectbox(
                    "Sort by",
                    ["Date (newest first)", "Date (oldest first)", "Case A name", "Case B name"]
                )
                
                # Apply filters and sorting
                filtered_comparisons = processed_comparisons.copy()
                
                # Apply search filter
                if search_term:
                    filtered_comparisons = [
                        comp for comp in filtered_comparisons 
                        if search_term.lower() in comp['case_a_name'].lower() or 
                           search_term.lower() in comp['case_b_name'].lower()
                    ]
                
                # Apply winner filter
                if winner_filter != "All":
                    if winner_filter == "No winner":
                        filtered_comparisons = [comp for comp in filtered_comparisons if comp['winner_label'] == "N/A"]
                    else:
                        filtered_comparisons = [comp for comp in filtered_comparisons if comp['winner_label'] == winner_filter]
                
                # Apply sorting
                if sort_option == "Date (newest first)":
                    filtered_comparisons.sort(key=lambda x: x['comparison_date'] or "", reverse=True)
                elif sort_option == "Date (oldest first)":
                    filtered_comparisons.sort(key=lambda x: x['comparison_date'] or "")
                elif sort_option == "Case A name":
                    filtered_comparisons.sort(key=lambda x: x['case_a_name'])
                elif sort_option == "Case B name":
                    filtered_comparisons.sort(key=lambda x: x['case_b_name'])
                
                # Show filtered count
                if len(filtered_comparisons) != len(comparisons):
                    st.info(f"Showing {len(filtered_comparisons)} of {len(comparisons)} comparisons")
                
                st.write("---")
                
                # Display filtered comparisons in collapsible format
                for comp in filtered_comparisons:
                    # Create informative expander title
                    winner_info = f"→ Winner: {comp['winner_text']} ({comp['winner_label']})" if comp['winner_label'] != "N/A" else "→ No winner"
                    expander_title = f"**Comparison #{comp['idx'] + 1}:** {comp['case_a_name']} vs {comp['case_b_name']} {winner_info}"
                    
                    with st.expander(expander_title, expanded=False):
                        # Header with AI badge and date
                        col_header1, col_header2 = st.columns([4, 1])
                        with col_header1:
                            st.markdown(f"**Comparison #{comp['idx'] + 1} - AI_Generated**")
                        with col_header2:
                            st.markdown(f"**{comp['formatted_date']}**")
                            st.markdown("🤖 AI")
                        
                        # Main comparison layout
                        col1, col2, col3 = st.columns([2, 2, 1])
                        
                        with col1:
                            st.markdown("📋 **Test A:**")
                            st.markdown(f"*{comp['case_a_name']}* ({comp['case_a_citation']})")
                            st.markdown("**Test:** " + comp['test_a'])
                            
                        with col2:
                            st.markdown("📋 **Test B:**")
                            st.markdown(f"*{comp['case_b_name']}* ({comp['case_b_citation']})")
                            st.markdown("**Test:** " + comp['test_b'])
                            
                        with col3:
                            st.markdown("🏆 **Winner:**")
                            if comp['winner_label'] != "N/A":
                                st.markdown(f"<div style='background-color: #d4edda; padding: 10px; border-radius: 5px; text-align: center;'><strong style='color: #155724;'>{comp['winner_text']}</strong><br><small>More Rule-Like</small></div>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center;'><strong>{comp['winner_text']}</strong></div>", unsafe_allow_html=True)
                        
                        # Reasoning section
                        st.markdown("🧠 **Reasoning:**")
                        st.markdown(comp['comparison_rationale'] if comp['comparison_rationale'] else "No reasoning provided")
        
        with tab6:
            # Results tab
            st.subheader("📊 Analysis Results")
            
            if total_tests == 0 or total_comparisons == 0:
                st.info("Complete extractions and comparisons to view detailed analysis results.")
                st.write("**Current Progress:**")
                st.write(f"- Tests Extracted: {total_tests}/{n_cases}")
                st.write(f"- Comparisons Made: {total_comparisons}/{required_comparisons}")
            else:
                # Comprehensive Bradley-Terry Analysis
                show_bradley_terry_analysis(experiment_id, n_cases, total_tests, total_comparisons)
        
        with tab7:
            st.subheader("⚙️ Experiment Settings")
            
            # Configuration Management
            st.write("**Configuration Management**")
            
            # Check if experiment has started execution
            has_extractions = execute_sql("SELECT COUNT(*) FROM v2_experiment_extractions WHERE experiment_id = ?", (experiment_id,), fetch=True)
            has_extractions = has_extractions[0][0] > 0 if has_extractions else False
            
            col1, col2 = st.columns(2)
            
            with col1:
                if has_extractions:
                    st.button("✏️ Edit Configuration", type="secondary", use_container_width=True, disabled=True, 
                             help="Cannot edit configuration after execution has started")
                else:
                    if st.button("✏️ Edit Configuration", type="secondary", use_container_width=True):
                        st.session_state.editing_experiment = experiment_id
                        st.session_state.selected_page = "Create Experiment"
                        st.rerun()
            
            with col2:
                if st.button("📋 Clone Experiment", type="secondary", use_container_width=True):
                    # Set up cloning by copying the experiment data
                    st.session_state.editing_experiment = None
                    st.session_state.clone_from_experiment = experiment_id
                    st.session_state.selected_page = "Create Experiment"
                    st.rerun()
            
            if has_extractions:
                st.info("⚠️ Configuration editing is disabled once execution has started to maintain experiment integrity.")
            
            # Navigation
            st.write("")
            st.write("**Navigation**")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Back to Overview", use_container_width=True):
                    st.session_state.selected_page = "Library Overview"
                    st.rerun()
            
            with col2:
                if st.button("📈 View in Comparison", use_container_width=True):
                    st.session_state.selected_page = "Comparison"
                    st.rerun()
            
            # Experiment Status Management
            st.write("")
            st.write("**Status Management**")
            
            current_status = exp['status']
            status_options = ['draft', 'in_progress', 'complete', 'archived']
            status_descriptions = {
                'draft': 'Draft - Experiment is being configured',
                'in_progress': 'In Progress - Experiment is actively running',
                'complete': 'Complete - Experiment has finished successfully',
                'archived': 'Archived - Experiment is stored but not active'
            }
            
            st.write(f"**Current Status:** {status_descriptions[current_status]}")
            
            # Status change buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if current_status == 'draft':
                    st.info("Experiment will automatically transition to 'In Progress' when execution starts.")
                elif current_status == 'in_progress':
                    st.info("Experiment is actively running.")
                elif current_status == 'complete':
                    if st.button("🔄 Revert to In Progress", type="secondary", use_container_width=True):
                        execute_sql("UPDATE v2_experiments SET status = 'in_progress', modified_date = CURRENT_TIMESTAMP WHERE experiment_id = ?", (experiment_id,))
                        st.success("Experiment reverted to in progress!")
                        st.rerun()
                        
            with col2:
                if current_status != 'complete':
                    if st.button("🏁 Mark Complete", type="secondary", use_container_width=True):
                        execute_sql("UPDATE v2_experiments SET status = 'complete', modified_date = CURRENT_TIMESTAMP WHERE experiment_id = ?", (experiment_id,))
                        st.success("Experiment marked as complete!")
                        st.rerun()
                        
            with col3:
                if current_status != 'archived':
                    if st.button("📦 Archive", type="secondary", use_container_width=True):
                        execute_sql("UPDATE v2_experiments SET status = 'archived', modified_date = CURRENT_TIMESTAMP WHERE experiment_id = ?", (experiment_id,))
                        st.success("Experiment archived!")
                        st.rerun()
                        
            # Danger zone
            st.write("")
            st.write("**⚠️ Danger Zone**")
            with st.expander("Advanced Actions", expanded=False):
                st.warning("These actions cannot be undone. Use with caution.")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Delete Extractions**")
                    confirm_delete_extractions = st.checkbox(
                        "I confirm I want to permanently delete all extractions for this experiment",
                        key=f"confirm_extractions_{experiment_id}"
                    )
                    if st.button(
                        "🗑️ Delete Extractions", 
                        type="primary" if confirm_delete_extractions else "secondary",
                        disabled=not confirm_delete_extractions,
                        use_container_width=True
                    ):
                        execute_sql("DELETE FROM v2_experiment_extractions WHERE experiment_id = ?", (experiment_id,))
                        
                        # Check if both extractions and comparisons are now empty
                        remaining_extractions = execute_sql("SELECT COUNT(*) FROM v2_experiment_extractions WHERE experiment_id = ?", (experiment_id,), fetch=True)[0][0]
                        remaining_comparisons = execute_sql("SELECT COUNT(*) FROM v2_experiment_comparisons WHERE experiment_id = ?", (experiment_id,), fetch=True)[0][0]
                        
                        if remaining_extractions == 0 and remaining_comparisons == 0:
                            # Reset experiment status to draft if all work is deleted
                            execute_sql("UPDATE v2_experiments SET status = 'draft' WHERE experiment_id = ?", (experiment_id,))
                            st.success("All extractions deleted! Experiment status reset to draft.")
                        else:
                            st.success("All extractions deleted!")
                        
                        st.cache_data.clear()
                        st.rerun()
                
                with col2:
                    st.write("**Delete Comparisons**")
                    confirm_delete_comparisons = st.checkbox(
                        "I confirm I want to permanently delete all comparisons for this experiment",
                        key=f"confirm_comparisons_{experiment_id}"
                    )
                    if st.button(
                        "🗑️ Delete Comparisons", 
                        type="primary" if confirm_delete_comparisons else "secondary",
                        disabled=not confirm_delete_comparisons,
                        use_container_width=True
                    ):
                        execute_sql("DELETE FROM v2_experiment_comparisons WHERE experiment_id = ?", (experiment_id,))
                        
                        # Check if both extractions and comparisons are now empty
                        remaining_extractions = execute_sql("SELECT COUNT(*) FROM v2_experiment_extractions WHERE experiment_id = ?", (experiment_id,), fetch=True)[0][0]
                        remaining_comparisons = execute_sql("SELECT COUNT(*) FROM v2_experiment_comparisons WHERE experiment_id = ?", (experiment_id,), fetch=True)[0][0]
                        
                        if remaining_extractions == 0 and remaining_comparisons == 0:
                            # Reset experiment status to draft if all work is deleted
                            execute_sql("UPDATE v2_experiments SET status = 'draft' WHERE experiment_id = ?", (experiment_id,))
                            st.success("All comparisons deleted! Experiment status reset to draft.")
                        else:
                            st.success("All comparisons deleted!")
                        
                        st.cache_data.clear()
                        st.rerun()
                
    except Exception as e:
        st.error(f"Error loading experiment details: {e}")
        st.write("Debug info:")
        st.write(f"Experiment ID: {experiment_id}")
        st.write(f"Error: {str(e)}")

def show():
    """Main dashboard interface"""
    # Initialize database tables
    initialize_experiment_tables()
    
    # Show sidebar navigation
    show_sidebar_navigation()
    
    # Main content area - no redundant title since it's in sidebar
    # Get current page from session state - check both selected_page and page_navigation
    current_page = st.session_state.get('page_navigation') or st.session_state.get('selected_page', 'Cases')
    
    # Render content based on selected page
    if current_page == "Cases":
        show_case_management()
        
    elif current_page == "Library Overview":
        show_experiment_overview()
        
    elif current_page == "Experiment Detail":
        experiment_id = st.session_state.get('selected_experiment')
        if experiment_id:
            show_experiment_detail(experiment_id)
        else:
            st.error("No experiment selected")
            
    elif current_page == "Comparison":
        show_experiment_comparison()
        
    elif current_page == "Create Experiment":
        show_experiment_configuration()
        
    elif current_page == "⚗️ Experiment Execution":
        # Show experiment execution interface
        active_experiment = st.session_state.get('active_experiment')
        if active_experiment:
            experiment_execution.show()
        else:
            st.error("No active experiment selected")
            st.session_state.page_navigation = None  # Reset navigation
            st.session_state.selected_page = "Library Overview"
            st.rerun()