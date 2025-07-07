
import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import scipy.stats
import scipy.optimize
from scipy.stats import chi2
import time
from config import DB_NAME, get_gemini_model, GEMINI_MODELS, list_available_models, save_api_key, load_api_key, delete_api_key, DEFAULT_MODEL
from schemas import ExtractedLegalTest, LegalTestComparison
import google.generativeai as genai
import statsmodels.api as sm
import matplotlib.pyplot as plt
import choix

def read_prompt(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def _compute_bradley_terry_statistics(n_items, comparison_data, params):
    """Compute comprehensive Bradley-Terry statistical metrics"""
    
    # Initialize results dictionary
    stats = {
        'standard_errors': None,
        'confidence_intervals_95': None,
        'confidence_intervals_99': None,
        'p_values': None,
        'log_likelihood': None,
        'aic': None,
        'bic': None,
        'deviance': None,
        'null_deviance': None,
        'likelihood_ratio_stat': None,
        'likelihood_ratio_p': None,
        'bootstrap_results': None
    }
    
    try:
        # Compute log-likelihood for current model
        log_likelihood = _bradley_terry_log_likelihood(params, comparison_data)
        stats['log_likelihood'] = log_likelihood
        
        # Compute AIC and BIC
        k = len(params) - 1  # Number of free parameters (last parameter fixed to 0)
        stats['aic'] = 2 * k - 2 * log_likelihood
        stats['bic'] = np.log(len(comparison_data)) * k - 2 * log_likelihood
        
        # Compute null model statistics (all parameters equal)
        null_params = np.zeros(n_items)
        null_log_likelihood = _bradley_terry_log_likelihood(null_params, comparison_data)
        stats['null_deviance'] = -2 * null_log_likelihood
        stats['deviance'] = -2 * log_likelihood
        
        # Likelihood ratio test
        stats['likelihood_ratio_stat'] = 2 * (log_likelihood - null_log_likelihood)
        df = k  # degrees of freedom
        stats['likelihood_ratio_p'] = 1 - chi2.cdf(stats['likelihood_ratio_stat'], df)
        
        # Bootstrap for standard errors and confidence intervals
        try:
            bootstrap_results = _bootstrap_bradley_terry(n_items, comparison_data, n_bootstrap=1000)
            stats['bootstrap_results'] = bootstrap_results
            
            # Standard errors from bootstrap
            stats['standard_errors'] = np.std(bootstrap_results, axis=0)
            
            # Bootstrap confidence intervals
            stats['confidence_intervals_95'] = np.percentile(bootstrap_results, [2.5, 97.5], axis=0)
            stats['confidence_intervals_99'] = np.percentile(bootstrap_results, [0.5, 99.5], axis=0)
            
            # P-values using bootstrap distribution
            # Test H0: parameter = 0
            stats['p_values'] = []
            for i in range(len(params)):
                # Two-tailed test
                bootstrap_param = bootstrap_results[:, i]
                p_value = 2 * min(
                    np.mean(bootstrap_param >= 0),
                    np.mean(bootstrap_param <= 0)
                )
                stats['p_values'].append(min(p_value, 1.0))
            stats['p_values'] = np.array(stats['p_values'])
            
        except Exception as bootstrap_error:
            st.warning(f"Bootstrap analysis failed: {bootstrap_error}. Using approximate methods.")
            
            # Fallback to approximate standard errors using numerical Hessian
            try:
                hessian = _numerical_hessian(params, comparison_data)
                fisher_info = -hessian
                cov_matrix = np.linalg.inv(fisher_info)
                stats['standard_errors'] = np.sqrt(np.diag(cov_matrix))
                
                # Approximate confidence intervals using normal distribution
                z_95 = 1.96
                z_99 = 2.576
                se = stats['standard_errors']
                
                stats['confidence_intervals_95'] = np.array([
                    params - z_95 * se,
                    params + z_95 * se
                ])
                
                stats['confidence_intervals_99'] = np.array([
                    params - z_99 * se,
                    params + z_99 * se
                ])
                
                # P-values using normal approximation
                t_stats = params / se
                stats['p_values'] = 2 * (1 - scipy.stats.norm.cdf(np.abs(t_stats)))
                
            except Exception as numerical_error:
                st.warning(f"Numerical methods also failed: {numerical_error}. Statistical inference not available.")
    
    except Exception as e:
        st.warning(f"Statistical computation failed: {e}")
    
    return stats

def _bradley_terry_log_likelihood(params, comparison_data):
    """Compute log-likelihood for Bradley-Terry model"""
    log_likelihood = 0.0
    for winner, loser in comparison_data:
        prob = np.exp(params[winner]) / (np.exp(params[winner]) + np.exp(params[loser]))
        log_likelihood += np.log(max(prob, 1e-15))  # Avoid log(0)
    return log_likelihood

def _numerical_hessian(params, comparison_data, h=1e-5):
    """Compute numerical Hessian matrix for Bradley-Terry likelihood"""
    n = len(params)
    hessian = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Second partial derivative
            params_pp = params.copy()
            params_pm = params.copy()
            params_mp = params.copy()
            params_mm = params.copy()
            
            params_pp[i] += h
            params_pp[j] += h
            
            params_pm[i] += h
            params_pm[j] -= h
            
            params_mp[i] -= h
            params_mp[j] += h
            
            params_mm[i] -= h
            params_mm[j] -= h
            
            hessian[i, j] = (
                _bradley_terry_log_likelihood(params_pp, comparison_data) -
                _bradley_terry_log_likelihood(params_pm, comparison_data) -
                _bradley_terry_log_likelihood(params_mp, comparison_data) +
                _bradley_terry_log_likelihood(params_mm, comparison_data)
            ) / (4 * h * h)
    
    return hessian

def _bootstrap_bradley_terry(n_items, comparison_data, n_bootstrap=1000):
    """Bootstrap Bradley-Terry parameter estimates"""
    bootstrap_params = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample of comparisons
        bootstrap_indices = np.random.choice(len(comparison_data), len(comparison_data), replace=True)
        bootstrap_comparisons = [comparison_data[i] for i in bootstrap_indices]
        
        try:
            # Fit Bradley-Terry model to bootstrap sample
            bootstrap_result = choix.ilsr_pairwise(n_items, bootstrap_comparisons, alpha=0.01, max_iter=5000)
            bootstrap_params.append(bootstrap_result)
        except:
            # If fitting fails, use original parameters
            continue
    
    return np.array(bootstrap_params)

def _simple_k_fold_split(data, n_folds=5, random_seed=42):
    """Simple K-fold split implementation without sklearn"""
    np.random.seed(random_seed)
    n_samples = len(data)
    indices = np.random.permutation(n_samples)
    
    fold_size = n_samples // n_folds
    folds = []
    
    for i in range(n_folds):
        start_idx = i * fold_size
        if i == n_folds - 1:  # Last fold gets remaining samples
            end_idx = n_samples
        else:
            end_idx = (i + 1) * fold_size
        
        test_indices = indices[start_idx:end_idx]
        train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
        folds.append((train_indices, test_indices))
    
    return folds

def _cross_validate_bradley_terry(n_items, comparison_data, n_folds=5):
    """Perform cross-validation for Bradley-Terry model"""
    if len(comparison_data) < n_folds:
        return None
    
    folds = _simple_k_fold_split(comparison_data, n_folds=n_folds)
    fold_accuracies = []
    
    for train_idx, test_idx in folds:
        train_data = [comparison_data[i] for i in train_idx]
        test_data = [comparison_data[i] for i in test_idx]
        
        try:
            # Train model on training fold
            train_params = choix.ilsr_pairwise(n_items, train_data, alpha=0.01, max_iter=5000)
            
            # Test on validation fold
            correct_predictions = 0
            for winner, loser in test_data:
                predicted_prob = np.exp(train_params[winner]) / (
                    np.exp(train_params[winner]) + np.exp(train_params[loser])
                )
                if predicted_prob > 0.5:  # Model predicts winner correctly
                    correct_predictions += 1
            
            accuracy = correct_predictions / len(test_data)
            fold_accuracies.append(accuracy)
            
        except:
            continue
    
    if fold_accuracies:
        return {
            'mean_accuracy': np.mean(fold_accuracies),
            'std_accuracy': np.std(fold_accuracies),
            'fold_accuracies': fold_accuracies
        }
    return None

def _analyze_inconsistencies(comparison_data, n_items):
    """Analyze transitivity violations and inconsistencies"""
    # Build adjacency matrix for comparisons
    wins = np.zeros((n_items, n_items))
    for winner, loser in comparison_data:
        wins[winner, loser] += 1
    
    # Find circular triads (A beats B, B beats C, C beats A)
    circular_triads = []
    total_triads = 0
    
    for i in range(n_items):
        for j in range(i+1, n_items):
            for k in range(j+1, n_items):
                total_triads += 1
                
                # Check all possible circular patterns in this triad
                patterns = [
                    (i, j, k),  # i>j>k>i
                    (i, k, j),  # i>k>j>i
                    (j, i, k),  # j>i>k>j
                    (j, k, i),  # j>k>i>j
                    (k, i, j),  # k>i>j>k
                    (k, j, i),  # k>j>i>k
                ]
                
                for a, b, c in patterns:
                    if wins[a, b] > 0 and wins[b, c] > 0 and wins[c, a] > 0:
                        circular_triads.append((a, b, c))
                        break
    
    return {
        'total_triads': total_triads,
        'circular_triads': len(circular_triads),
        'inconsistency_rate': len(circular_triads) / max(total_triads, 1),
        'circular_examples': circular_triads[:5]  # First 5 examples
    }

def _display_temporal_analysis_and_stats(analyzed_df, comparison_data, graph_analysis, enhanced_stats=None):
    """Helper function to display temporal analysis and comprehensive statistics"""
    
    # Temporal analysis (only for analyzed tests)
    if len(analyzed_df) > 0:
        st.subheader("Temporal Analysis")
        
        if len(analyzed_df) >= 3:  # Need at least 3 points for meaningful trend
            # Create temporal plot
            fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
            scatter = ax.scatter(analyzed_df['decision_year'], analyzed_df['bt_score'], 
                               s=60, alpha=0.7, c=analyzed_df['bt_score'], cmap='RdYlBu_r')
            
            # Add trend line
            z = pd.DataFrame({'year': analyzed_df['decision_year'], 'score': analyzed_df['bt_score']}).dropna()
            if len(z) >= 2:
                trend_line = sm.OLS(z['score'], sm.add_constant(z['year'])).fit()
                trend_y = trend_line.predict(sm.add_constant(z['year']))
                ax.plot(z['year'], trend_y, 'r--', alpha=0.8, linewidth=2, label=f'Trend (R²={trend_line.rsquared:.3f})')
                
                # Statistical analysis
                slope = trend_line.params[1]
                p_value = trend_line.pvalues[1]
                
                if p_value < 0.05:
                    direction = "increasing" if slope > 0 else "decreasing"
                    st.write(f"📈 **Significant trend detected**: Rule-likeness is {direction} over time (p = {p_value:.4f})")
                else:
                    st.write(f"📊 **No significant temporal trend** detected (p = {p_value:.4f})")
            
            ax.set_xlabel("Decision Year")
            ax.set_ylabel("Rule-Likeness Score")
            ax.set_title("Evolution of Legal Test Rule-Likeness Over Time")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            plt.colorbar(scatter, label='Rule-Likeness Score')
            
            st.pyplot(fig)
            
            # Summary statistics (using analyzed_df instead of results_df)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Most Rule-Like", analyzed_df.iloc[0]['case_name'], 
                         f"{analyzed_df.iloc[0]['bt_score']:.3f}")
            with col2:
                st.metric("Least Rule-Like", analyzed_df.iloc[-1]['case_name'], 
                         f"{analyzed_df.iloc[-1]['bt_score']:.3f}")
            with col3:
                st.metric("Score Range", "", 
                         f"{analyzed_df['bt_score'].max() - analyzed_df['bt_score'].min():.3f}")
            
            # Comprehensive Bradley-Terry Statistical Results
            if st.checkbox("📊 **Show Comprehensive Bradley-Terry Statistical Results**", key="show_bt_stats"):
                st.write("### Model Performance and Reliability")
                
                # Basic descriptive statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Descriptive Statistics:**")
                    scores = analyzed_df['bt_score']
                    st.write(f"• **Mean Score**: {scores.mean():.4f}")
                    st.write(f"• **Median Score**: {scores.median():.4f}")
                    st.write(f"• **Standard Deviation**: {scores.std():.4f}")
                    st.write(f"• **Minimum Score**: {scores.min():.4f}")
                    st.write(f"• **Maximum Score**: {scores.max():.4f}")
                    st.write(f"• **Interquartile Range**: {scores.quantile(0.75) - scores.quantile(0.25):.4f}")
                
                with col2:
                    st.write("**Model Characteristics:**")
                    st.write(f"• **Total Comparisons**: {len(comparison_data)}")
                    st.write(f"• **Tests Analyzed**: {len(analyzed_df)}")
                    st.write(f"• **Graph Connectivity**: {'✅ Connected' if graph_analysis['is_connected'] else '❌ Disconnected'}")
                    st.write(f"• **Connected Components**: {len(graph_analysis['components'])}")
                    
                    # Calculate comparison density
                    max_possible = len(analyzed_df) * (len(analyzed_df) - 1) // 2
                    density = len(comparison_data) / max_possible if max_possible > 0 else 0
                    st.write(f"• **Comparison Density**: {density:.1%}")
                    
                    # Sparsity analysis
                    if density < 0.1:
                        sparsity_level = "Very Sparse"
                    elif density < 0.3:
                        sparsity_level = "Sparse"
                    elif density < 0.7:
                        sparsity_level = "Moderate"
                    else:
                        sparsity_level = "Dense"
                    st.write(f"• **Data Sparsity**: {sparsity_level}")
                
                # Enhanced Bradley-Terry Statistical Metrics
                if enhanced_stats and enhanced_stats.get('bt_statistics'):
                    bt_stats = enhanced_stats['bt_statistics']
                    
                    st.write("### Advanced Statistical Metrics")
                    
                    # Model fit and goodness-of-fit
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("**Model Fit:**")
                        if bt_stats.get('log_likelihood') is not None:
                            st.write(f"• **Log-Likelihood**: {bt_stats['log_likelihood']:.3f}")
                        if bt_stats.get('aic') is not None:
                            st.write(f"• **AIC**: {bt_stats['aic']:.3f}")
                        if bt_stats.get('bic') is not None:
                            st.write(f"• **BIC**: {bt_stats['bic']:.3f}")
                    
                    with col2:
                        st.write("**Goodness-of-Fit:**")
                        if bt_stats.get('deviance') is not None:
                            st.write(f"• **Deviance**: {bt_stats['deviance']:.3f}")
                        if bt_stats.get('null_deviance') is not None:
                            st.write(f"• **Null Deviance**: {bt_stats['null_deviance']:.3f}")
                        if bt_stats.get('likelihood_ratio_stat') is not None:
                            st.write(f"• **LR Statistic**: {bt_stats['likelihood_ratio_stat']:.3f}")
                    
                    with col3:
                        st.write("**Model Significance:**")
                        if bt_stats.get('likelihood_ratio_p') is not None:
                            p_val = bt_stats['likelihood_ratio_p']
                            if p_val < 0.001:
                                significance = "*** (p < 0.001)"
                            elif p_val < 0.01:
                                significance = "** (p < 0.01)"
                            elif p_val < 0.05:
                                significance = "* (p < 0.05)"
                            else:
                                significance = "(not significant)"
                            st.write(f"• **LR Test p-value**: {p_val:.6f} {significance}")
                    
                    # Statistical inference for individual parameters
                    if bt_stats.get('standard_errors') is not None and bt_stats.get('p_values') is not None:
                        st.write("### Statistical Inference for Individual Tests")
                        
                        # Create enhanced results table with statistical inference
                        inference_data = []
                        se_array = bt_stats['standard_errors']
                        p_array = bt_stats['p_values']
                        ci_95 = bt_stats.get('confidence_intervals_95')
                        
                        for i, (_, row) in enumerate(analyzed_df.iterrows()):
                            # Find the parameter index for this test
                            test_id = row['test_id']
                            if enhanced_stats.get('params') is not None and i < len(se_array):
                                se = se_array[i]
                                p_val = p_array[i]
                                
                                # Significance stars
                                if p_val < 0.001:
                                    sig = "***"
                                elif p_val < 0.01:
                                    sig = "**"
                                elif p_val < 0.05:
                                    sig = "*"
                                else:
                                    sig = ""
                                
                                # Confidence interval
                                ci_text = ""
                                if ci_95 is not None and i < ci_95.shape[1]:
                                    ci_lower = ci_95[0, i]
                                    ci_upper = ci_95[1, i]
                                    ci_text = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
                                
                                inference_data.append({
                                    'Case Name': row['case_name'],
                                    'Score': f"{row['bt_score']:.4f}{sig}",
                                    'Std Error': f"{se:.4f}",
                                    'p-value': f"{p_val:.4f}",
                                    '95% CI': ci_text
                                })
                        
                        if inference_data:
                            inference_df = pd.DataFrame(inference_data)
                            st.dataframe(inference_df, use_container_width=True)
                            st.caption("*** p<0.001, ** p<0.01, * p<0.05")
                
                # Cross-validation results
                if enhanced_stats and enhanced_stats.get('cv_results'):
                    cv_results = enhanced_stats['cv_results']
                    st.write("### Cross-Validation Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean CV Accuracy", f"{cv_results['mean_accuracy']:.3f}")
                    with col2:
                        st.metric("CV Std Deviation", f"{cv_results['std_accuracy']:.3f}")
                    with col3:
                        st.metric("Number of Folds", str(len(cv_results['fold_accuracies'])))
                    
                    # Display individual fold accuracies
                    fold_text = ", ".join([f"{acc:.3f}" for acc in cv_results['fold_accuracies']])
                    st.write(f"**Individual fold accuracies**: {fold_text}")
                
                # Inconsistency analysis
                if enhanced_stats and enhanced_stats.get('inconsistency_results'):
                    inconsistency = enhanced_stats['inconsistency_results']
                    st.write("### Transitivity and Consistency Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Triads", str(inconsistency['total_triads']))
                    with col2:
                        st.metric("Circular Triads", str(inconsistency['circular_triads']))
                    with col3:
                        st.metric("Inconsistency Rate", f"{inconsistency['inconsistency_rate']:.1%}")
                    
                    if inconsistency['circular_triads'] > 0:
                        if st.checkbox("Show Examples of Circular Preferences", key="show_circular_examples"):
                            for i, (a, b, c) in enumerate(inconsistency['circular_examples']):
                                if i < len(analyzed_df):
                                    st.write(f"Circular triad {i+1}: Test indices {a} > {b} > {c} > {a}")
                    else:
                        st.success("✅ No circular preferences detected - perfectly transitive comparisons!")
                
                # Score distribution analysis
                st.write("### Score Distribution Analysis")
                
                # Create score distribution plot
                fig_hist, ax_hist = plt.subplots(figsize=(8, 4), dpi=100)
                ax_hist.hist(scores, bins=min(10, len(scores)//2), alpha=0.7, color='skyblue', edgecolor='black')
                ax_hist.set_xlabel("Rule-Likeness Score")
                ax_hist.set_ylabel("Frequency")
                ax_hist.set_title("Distribution of Rule-Likeness Scores")
                ax_hist.grid(True, alpha=0.3)
                
                # Add vertical lines for mean and median
                ax_hist.axvline(scores.mean(), color='red', linestyle='--', label=f'Mean: {scores.mean():.3f}')
                ax_hist.axvline(scores.median(), color='orange', linestyle='--', label=f'Median: {scores.median():.3f}')
                ax_hist.legend()
                
                st.pyplot(fig_hist)
                
                # Statistical significance and confidence
                st.write("### Statistical Reliability")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Confidence Intervals (Approximate):**")
                    # Calculate approximate confidence intervals using normal approximation
                    import scipy.stats as stats
                    n = len(scores)
                    se = scores.std() / np.sqrt(n)
                    ci_95 = stats.t.interval(0.95, n-1, loc=scores.mean(), scale=se)
                    ci_99 = stats.t.interval(0.99, n-1, loc=scores.mean(), scale=se)
                    
                    st.write(f"• **95% CI for Mean**: ({ci_95[0]:.4f}, {ci_95[1]:.4f})")
                    st.write(f"• **99% CI for Mean**: ({ci_99[0]:.4f}, {ci_99[1]:.4f})")
                    
                    # Score reliability
                    if len(scores) >= 30:
                        reliability = "High (n≥30)"
                    elif len(scores) >= 10:
                        reliability = "Moderate (10≤n<30)"
                    else:
                        reliability = "Low (n<10)"
                    st.write(f"• **Sample Size Reliability**: {reliability}")
                
                with col2:
                    st.write("**Model Convergence & Quality:**")
                    
                    # Check for score extremes that might indicate poor convergence
                    extreme_scores = len(scores[(scores > 3) | (scores < -3)])
                    st.write(f"• **Extreme Scores** (|score| > 3): {extreme_scores}")
                    
                    # Balance check
                    positive_scores = len(scores[scores > 0])
                    negative_scores = len(scores[scores < 0])
                    zero_scores = len(scores[scores == 0])
                    st.write(f"• **Positive Scores**: {positive_scores}")
                    st.write(f"• **Negative Scores**: {negative_scores}")
                    st.write(f"• **Zero Scores**: {zero_scores}")
                    
                    # Balance ratio
                    if negative_scores > 0:
                        balance_ratio = positive_scores / negative_scores
                        st.write(f"• **Balance Ratio** (pos/neg): {balance_ratio:.2f}")
                
                # Ranking reliability
                st.write("### Ranking Analysis")
                
                # Calculate rank correlations and stability metrics
                ranks = scores.rank(ascending=False)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Top-Ranked Tests (Most Rule-Like):**")
                    top_5 = analyzed_df.head(min(5, len(analyzed_df)))
                    for i, (_, row) in enumerate(top_5.iterrows(), 1):
                        st.write(f"{i}. **{row['case_name']}** ({row['bt_score']:.3f})")
                
                with col2:
                    st.write("**Bottom-Ranked Tests (Most Standard-Like):**")
                    bottom_5 = analyzed_df.tail(min(5, len(analyzed_df)))
                    for i, (_, row) in enumerate(reversed(list(bottom_5.iterrows())), 1):
                        rank = len(analyzed_df) - i + 1
                        st.write(f"{rank}. **{row['case_name']}** ({row['bt_score']:.3f})")
                
                # Provide explanations for each statistic
                st.write("### Statistical Explanations")
                
                if st.checkbox("📖 **Show Statistical Explanations**", key="show_explanations"):
                    st.write("""
                    **Bradley-Terry Model Basics:**
                    - The Bradley-Terry model estimates the probability that one item beats another in pairwise comparisons
                    - Scores are on a log-odds scale where 0 is neutral, positive values indicate more rule-like tests, negative values indicate more standard-like tests
                    - A difference of 1.0 in scores roughly corresponds to a 73% vs 27% win probability
                    
                    **Key Statistics Explained:**
                    
                    **Descriptive Statistics:**
                    - **Mean/Median**: Central tendency of rule-likeness scores
                    - **Standard Deviation**: Measures how spread out the scores are
                    - **Interquartile Range**: Spread of the middle 50% of scores (robust to outliers)
                    
                    **Model Quality Indicators:**
                    - **Comparison Density**: Percentage of all possible pairwise comparisons that were actually made
                    - **Graph Connectivity**: Whether all tests can be compared through chains of comparisons
                    - **Data Sparsity**: How complete our comparison data is (Dense > 70%, Moderate 30-70%, Sparse < 30%)
                    
                    **Confidence Intervals:**
                    - **95% CI**: Range where we're 95% confident the true mean lies
                    - **99% CI**: Range where we're 99% confident the true mean lies
                    - Narrower intervals indicate more precise estimates
                    
                    **Convergence Indicators:**
                    - **Extreme Scores**: Very high/low scores (>3 or <-3) may indicate convergence issues
                    - **Balance Ratio**: Ratio of positive to negative scores; should be reasonable for balanced data
                    
                    **Sample Size Reliability:**
                    - **High (n≥30)**: Results are statistically reliable
                    - **Moderate (10≤n<30)**: Results are reasonably reliable but interpret with caution
                    - **Low (n<10)**: Results are preliminary; more data needed for robust conclusions
                    
                    **Interpretation Guidelines:**
                    - Scores > 1.0: Strongly rule-like
                    - Scores 0.5 to 1.0: Moderately rule-like  
                    - Scores -0.5 to 0.5: Mixed/neutral characteristics
                    - Scores -1.0 to -0.5: Moderately standard-like
                    - Scores < -1.0: Strongly standard-like
                    
                    **Advanced Statistical Metrics:**
                    
                    **Model Fit Measures:**
                    - **Log-Likelihood**: Higher values indicate better model fit to the data
                    - **AIC (Akaike Information Criterion)**: Lower values indicate better model quality; penalizes complexity
                    - **BIC (Bayesian Information Criterion)**: Lower values indicate better model quality; stronger complexity penalty than AIC
                    
                    **Goodness-of-Fit Tests:**
                    - **Deviance**: Measure of how well the model fits the data (lower is better)
                    - **Null Deviance**: Deviance of a model with no predictors (all items equal)
                    - **Likelihood Ratio Test**: Tests if the model is significantly better than the null model
                    - **LR Statistic**: Higher values indicate stronger evidence against the null hypothesis
                    - **LR p-value**: Probability of observing data if null hypothesis is true (p < 0.05 indicates significant model improvement)
                    
                    **Statistical Inference:**
                    - **Standard Errors**: Measure uncertainty in parameter estimates (smaller is more precise)
                    - **P-values**: Probability that a score is significantly different from 0 (p < 0.05 indicates significant rule-likeness bias)
                    - **Confidence Intervals**: Range of plausible values for each score
                    - **Significance Stars**: *** p<0.001, ** p<0.01, * p<0.05 (conventional significance levels)
                    
                    **Cross-Validation:**
                    - **CV Accuracy**: Average prediction accuracy across folds (higher is better)
                    - **CV Standard Deviation**: Consistency of predictions across folds (lower is more stable)
                    - Good CV accuracy (>0.7) indicates the model generalizes well to new comparisons
                    
                    **Transitivity Analysis:**
                    - **Total Triads**: Number of possible three-way comparisons among tests
                    - **Circular Triads**: Number of violations of transitivity (A>B>C>A patterns)
                    - **Inconsistency Rate**: Percentage of triads that violate transitivity
                    - Lower inconsistency rates indicate more coherent preference structure
                    
                    **Interpretation of Advanced Metrics:**
                    - **Excellent Model**: Low AIC/BIC, significant LR test (p<0.001), CV accuracy >0.8, inconsistency rate <5%
                    - **Good Model**: Moderate AIC/BIC, significant LR test (p<0.05), CV accuracy >0.7, inconsistency rate <10%
                    - **Acceptable Model**: Higher AIC/BIC, marginal significance, CV accuracy >0.6, inconsistency rate <20%
                    - **Poor Model**: Very high AIC/BIC, non-significant LR test, CV accuracy <0.6, inconsistency rate >20%
                    """)
            
        else:
            st.info("Need at least 3 analyzed tests for temporal trend analysis.")

def run_extraction(case_text):
    if 'selected_gemini_model' not in st.session_state or not st.session_state.selected_gemini_model:
        st.error("No Gemini model selected. Please select one from the sidebar.")
        st.stop()
    model = get_gemini_model(st.session_state.selected_gemini_model)
    prompt = read_prompt('prompts/extractor_prompt.txt')
    response = model.generate_content(
        [prompt, case_text],
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json",
            response_schema=ExtractedLegalTest
        )
    )
    return ExtractedLegalTest.parse_raw(response.text)

def run_comparison(test1, test2):
    if 'selected_gemini_model' not in st.session_state or not st.session_state.selected_gemini_model:
        st.error("No Gemini model selected. Please select one from the sidebar.")
        st.stop()
    model = get_gemini_model(st.session_state.selected_gemini_model)
    prompt = read_prompt('prompts/comparator_prompt.txt')
    response = model.generate_content(
        [prompt, f"Test A: {test1}", f"Test B: {test2}"],
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json",
            response_schema=LegalTestComparison
        )
    )
    return response.text

def setup_database():
    """Creates the SQLite database and tables if they don't exist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS cases (
            case_id INTEGER PRIMARY KEY,
            case_name TEXT,
            citation TEXT UNIQUE,
            decision_year INTEGER,
            area_of_law TEXT,
            scc_url TEXT,
            full_text TEXT
        );
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS legal_tests (
            test_id INTEGER PRIMARY KEY,
            case_id INTEGER,
            test_novelty TEXT,
            extracted_test_summary TEXT,
            source_paragraphs TEXT,
            source_type TEXT,
            validation_status TEXT DEFAULT 'pending',
            validator_name TEXT,
            bt_score REAL,
            FOREIGN KEY (case_id) REFERENCES cases (case_id)
        );
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS legal_test_comparisons (
            comparison_id INTEGER PRIMARY KEY,
            test_id_1 INTEGER,
            test_id_2 INTEGER,
            more_rule_like_test_id INTEGER,
            reasoning TEXT,
            comparator_name TEXT,
            comparison_method TEXT DEFAULT 'human',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (test_id_1) REFERENCES legal_tests (test_id),
            FOREIGN KEY (test_id_2) REFERENCES legal_tests (test_id),
            FOREIGN KEY (more_rule_like_test_id) REFERENCES legal_tests (test_id),
            UNIQUE(test_id_1, test_id_2)
        );
    ''')
    
    conn.commit()
    conn.close()

def load_data_from_parquet(uploaded_file):
    """Loads SCC cases from a Parquet file into the database, with robust duplicate handling and filtering by Excel citations."""
    excel_file_path = "/Users/brandon/My Drive/Learning/Coding/SCC Research/scc_analysis_project/SCC Decisions Database.xlsx"
    excel_sheet_name = "Decisions Data"
    excel_citation_col = "Citation"
    excel_subject_col = "Subject"
    excel_url_col = "Decision Link"
    excel_case_name_col = "Case Name"

    try:
        excel_df = pd.read_excel(excel_file_path, sheet_name=excel_sheet_name)
        excel_df['citation_normalized'] = excel_df[excel_citation_col].astype(str).str.lower().str.strip()
        excel_citations_set = set(excel_df['citation_normalized'])
        st.info(f"Excel file contains {len(excel_citations_set)} unique normalized citations.")
        
        # Create mappings from normalized citation to subject and URL
        citation_to_subject = pd.Series(excel_df[excel_subject_col].values, index=excel_df['citation_normalized']).to_dict()
        citation_to_url = pd.Series(excel_df[excel_url_col].values, index=excel_df['citation_normalized']).to_dict()

    except FileNotFoundError:
        st.error(f"Excel file not found at {excel_file_path}. Please ensure it's in the correct directory.")
        return
    except KeyError as e:
        st.error(f"Missing expected column in Excel file: {e}. Please check column names.")
        return
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return

    df = pd.read_parquet(uploaded_file)
    st.info(f"Parquet file contains {len(df)} total rows.")

    if 'dataset' not in df.columns:
        st.error("Parquet file is missing the required 'dataset' column.")
        return

    scc_df = df[df['dataset'] == 'SCC'].copy()
    st.info(f"After filtering for dataset='SCC', {len(scc_df)} rows remain.")
    if scc_df.empty:
        st.warning("No cases with dataset = 'SCC' found in the file.")
        return
    
    # Ensure citation and citation2 columns exist and are normalized
    scc_df['citation_normalized_1'] = scc_df['citation'].astype(str).str.lower().str.strip()
    if 'citation2' in scc_df.columns:
        scc_df['citation_normalized_2'] = scc_df['citation2'].astype(str).str.lower().str.strip()
    else:
        scc_df['citation_normalized_2'] = '' # Create empty column if citation2 doesn't exist

    column_mapping = {
        'name': 'case_name',
        'citation': 'citation',
        'year': 'decision_year',
        'unofficial_text': 'full_text' # scc_url will come from Excel
    }
    
    required_source_columns = list(column_mapping.keys())
    # Add citation2 to required columns if it exists in the parquet file
    if 'citation2' in scc_df.columns:
        required_source_columns.append('citation2')

    if not all(col in scc_df.columns for col in required_source_columns):
        st.error(f"SCC data is missing one or more required columns: {', '.join(required_source_columns)}")
        return

    mapped_df = scc_df[required_source_columns].rename(columns=column_mapping)
    # Use citation_normalized_1 as the primary normalized citation for mapped_df
    mapped_df['citation_normalized'] = scc_df['citation_normalized_1']

    # Filter mapped_df to include only citations present in the Excel file
    initial_match_count = len(mapped_df)
    mapped_df = mapped_df[scc_df['citation_normalized_1'].isin(excel_citations_set) | scc_df['citation_normalized_2'].isin(excel_citations_set)]
    st.info(f"After matching with Excel citations (using citation or citation2), {len(mapped_df)} rows remain (dropped {initial_match_count - len(mapped_df)} due to no Excel match).")

    if mapped_df.empty:
        st.info("No SCC cases from the Parquet file match citations in the Excel database.")
        return

    # Populate area_of_law and scc_url using the mappings from Excel
    def get_mapping_value(citation_norm, mapping_dict, citation_norm_2=''):
        """Get mapping value, trying both normalized citations"""
        value = mapping_dict.get(citation_norm)
        if value is None and citation_norm_2:
            value = mapping_dict.get(citation_norm_2)
        return value
    
    mapped_df['area_of_law'] = mapped_df.apply(lambda row: get_mapping_value(
        row['citation_normalized'], citation_to_subject, 
        scc_df.loc[row.name, 'citation_normalized_2'] if 'citation_normalized_2' in scc_df.columns else ''), axis=1)
    
    mapped_df['scc_url'] = mapped_df.apply(lambda row: get_mapping_value(
        row['citation_normalized'], citation_to_url, 
        scc_df.loc[row.name, 'citation_normalized_2'] if 'citation_normalized_2' in scc_df.columns else ''), axis=1)
    
    # Validate URLs and show warning for problematic ones
    invalid_urls = mapped_df[mapped_df['scc_url'].str.contains('localhost|127.0.0.1', na=False, case=False)]
    if not invalid_urls.empty:
        st.warning(f"Found {len(invalid_urls)} cases with localhost URLs. These links may not work properly.")
    
    # Show some URL examples for debugging
    sample_urls = mapped_df['scc_url'].dropna().head(3).tolist()
    if sample_urls:
        st.info(f"Sample URLs from Excel: {', '.join(sample_urls[:2])}...")

    # Step 1: De-duplicate within the source file itself, keeping the first instance.
    initial_dedupe_count = len(mapped_df)
    mapped_df.drop_duplicates(subset=['citation_normalized'], keep='first', inplace=True)
    st.info(f"After de-duplicating within Parquet data, {len(mapped_df)} rows remain (dropped {initial_dedupe_count - len(mapped_df)} internal duplicates).")

    conn = sqlite3.connect(DB_NAME)
    try:
        # Step 2: Check against citations already in the database.
        existing_citations_df = pd.read_sql("SELECT citation FROM cases", conn)
        if not existing_citations_df.empty:
            existing_citations_normalized = set(existing_citations_df['citation'].astype(str).str.lower().str.strip())
            new_cases_df = mapped_df[~mapped_df['citation_normalized'].isin(existing_citations_normalized)]
            st.info(f"After checking against existing DB citations, {len(new_cases_df)} new cases identified for insertion.")
        else:
            new_cases_df = mapped_df
            st.info(f"No existing DB citations found. {len(new_cases_df)} cases identified for insertion.")

        if new_cases_df.empty:
            st.info("No new SCC cases to load. All cases in the file either already exist in the database or were duplicates within the file.")
        else:
            db_columns = ['case_name', 'citation', 'decision_year', 'area_of_law', 'scc_url', 'full_text']
            new_cases_to_insert = new_cases_df[db_columns]
            new_cases_to_insert.to_sql('cases', conn, if_exists='append', index=False)
            st.success(f"Successfully loaded {len(new_cases_to_insert)} new SCC cases into the database.")
    except sqlite3.IntegrityError:
        st.error("An unexpected error occurred. It seems there was still a duplicate citation. Please check the source file for inconsistencies.")
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("SCC Legal Test Analysis")

    # Load saved API key on startup
    if 'gemini_api_key' not in st.session_state:
        saved_key = load_api_key()
        if saved_key:
            st.session_state.gemini_api_key = saved_key

    # --- API Key Management ---
    st.sidebar.header("API Key Configuration")
    
    # Show current key status
    if 'gemini_api_key' in st.session_state and st.session_state.gemini_api_key:
        masked_key = st.session_state.gemini_api_key[:8] + "..." + st.session_state.gemini_api_key[-4:]
        st.sidebar.success(f"✅ API Key loaded: {masked_key}")
    else:
        st.sidebar.warning("⚠️ No API Key configured")
    
    api_key_input = st.sidebar.text_input("Enter your Gemini API Key", type="password", key="api_key_input")
    
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if st.button("Save Key"):
            if api_key_input:
                st.session_state.gemini_api_key = api_key_input
                if save_api_key(api_key_input):
                    st.success("API Key saved permanently.")
                else:
                    st.error("Failed to save API key.")
            else:
                st.warning("Please enter an API key.")
    with col2:
        if st.button("Clear Key"):
            if "gemini_api_key" in st.session_state:
                del st.session_state.gemini_api_key
                if delete_api_key():
                    st.info("API Key cleared.")
                else:
                    st.error("Failed to clear API key.")
    with col3:
        if st.button("Reload Key"):
            saved_key = load_api_key()
            if saved_key:
                st.session_state.gemini_api_key = saved_key
                st.success("API Key reloaded.")
            else:
                st.warning("No saved API key found.")

    # --- Model Selection ---
    st.sidebar.header("Gemini Model Selection")
    available_models = list_available_models()
    if available_models:
        # Set default model if not already selected
        if 'selected_gemini_model' not in st.session_state:
            st.session_state.selected_gemini_model = DEFAULT_MODEL
        
        # Get current index for selectbox
        current_index = 0
        if st.session_state.selected_gemini_model in available_models:
            current_index = available_models.index(st.session_state.selected_gemini_model)
        
        selected_model = st.sidebar.selectbox("Select a Gemini Model", available_models, index=current_index)
        st.session_state.selected_gemini_model = selected_model
    else:
        st.sidebar.warning("No Gemini models available. Please ensure your API key is valid.")
        st.session_state.selected_gemini_model = None

    # --- Validator Name (Required) ---
    st.sidebar.header("User Information")
    
    # Load saved validator name on startup
    if 'validator_name' not in st.session_state:
        st.session_state.validator_name = ""
    
    validator_name = st.sidebar.text_input(
        "Your Name (Required)", 
        value=st.session_state.validator_name,
        placeholder="Enter your name for auditing purposes"
    )
    
    # Update session state when name changes
    if validator_name != st.session_state.validator_name:
        st.session_state.validator_name = validator_name
    
    # Show status
    if validator_name.strip():
        st.sidebar.success(f"✅ Signed in as: {validator_name}")
    else:
        st.sidebar.error("❌ Name required to use the application")

    # Helper function to check if user can proceed
    def can_user_proceed():
        return bool(validator_name.strip())
    
    def show_name_required_error():
        st.error("❌ Please enter your name in the sidebar before using this feature.")
        st.stop()

    setup_database()

    # Calculate progress indicators
    conn = sqlite3.connect(DB_NAME)
    cases_count = pd.read_sql("SELECT COUNT(*) FROM cases", conn).iloc[0, 0]
    tests_count = pd.read_sql("SELECT COUNT(*) FROM legal_tests", conn).iloc[0, 0]
    comparisons_count = pd.read_sql("SELECT COUNT(*) FROM legal_test_comparisons", conn).iloc[0, 0]
    validated_count = pd.read_sql("SELECT COUNT(*) FROM legal_tests WHERE validation_status = 'accurate'", conn).iloc[0, 0]
    conn.close()

    # Section 1: Data Loading
    data_status = "✅ Complete" if cases_count > 0 else "📋 Pending"
    with st.expander(f"📁 **1. Data Loading** - {data_status} ({cases_count} cases loaded)", expanded=(cases_count == 0)):
        uploaded_file = st.file_uploader("Choose a Parquet file", type="parquet")
        if uploaded_file is not None:
            load_data_from_parquet(uploaded_file)
            st.success("Data loaded successfully!")
            st.rerun()  # Refresh to update progress indicators

        col_clear1, col_clear2 = st.columns(2)
        with col_clear1:
            if st.button("Clear Cases Database"):
                conn = sqlite3.connect(DB_NAME)
                c = conn.cursor()
                # Clear tables in dependency order due to foreign key constraints
                c.execute("DELETE FROM legal_test_comparisons")
                c.execute("DELETE FROM legal_tests")
                c.execute("DELETE FROM cases")
                conn.commit()
                conn.close()
                st.success("Cases, legal tests, and comparisons database cleared.")
                # Clear relevant session state variables
                if 'cases_to_sample' in st.session_state: del st.session_state.cases_to_sample
                if 'test_to_edit' in st.session_state: del st.session_state.test_to_edit
                if 'confirming_extraction' in st.session_state: del st.session_state.confirming_extraction
                st.rerun()
        with col_clear2:
            if st.button("Clear Extracted Tests Database"):
                conn = sqlite3.connect(DB_NAME)
                c = conn.cursor()
                # Clear comparisons first due to foreign key constraint
                c.execute("DELETE FROM legal_test_comparisons")
                c.execute("DELETE FROM legal_tests")
                conn.commit()
                conn.close()
                st.success("Extracted tests and comparisons database cleared.")
                # Clear relevant session state variables
                if 'test_to_edit' in st.session_state: del st.session_state.test_to_edit
                st.rerun()

        col_metric1, col_metric2, col_metric3 = st.columns(3)
        with col_metric1:
            st.metric("Total SCC Cases in DB", cases_count)
        with col_metric2:
            st.metric("Total Legal Tests in DB", tests_count)
        with col_metric3:
            st.metric("Total Comparisons in DB", comparisons_count)

    # Section 2: Extraction & Sampling
    pending_extractions = len(st.session_state.get('cases_to_sample', []))
    extraction_status = f"🔄 {pending_extractions} cases pending" if pending_extractions > 0 else "✅ Ready for extraction" if cases_count > 0 else "📋 Load data first"
    with st.expander(f"⚗️ **2. Extraction & Sampling** - {extraction_status}", expanded=(cases_count > 0 and tests_count < cases_count)):
        if cases_count == 0:
            st.info("Load case data first before running extractions.")
        else:
            # Check if user has entered their name
            if not can_user_proceed():
                show_name_required_error()
            
            num_cases_to_sample = st.number_input("Select N random cases to sample", min_value=1, value=5)

            if st.button("Start Session"):
                conn = sqlite3.connect(DB_NAME)
                cases_to_sample = pd.read_sql(f"SELECT * FROM cases ORDER BY RANDOM() LIMIT {num_cases_to_sample}", conn)
                conn.close()
                st.session_state.cases_to_sample = cases_to_sample
                st.rerun()

            # Display sampling results
            if 'cases_to_sample' in st.session_state and len(st.session_state.cases_to_sample) > 0:
                # Check for completed extractions to remove
                cases_to_remove = []
                for index, row in st.session_state.cases_to_sample.iterrows():
                    conn = sqlite3.connect(DB_NAME)
                    test_info = pd.read_sql(f"SELECT validation_status FROM legal_tests WHERE case_id = {row['case_id']}", conn)
                    conn.close()
                    
                    if not test_info.empty:
                        # Case has been extracted, mark for removal
                        if f"show_success_{row['case_id']}" not in st.session_state:
                            st.session_state[f"show_success_{row['case_id']}"] = True
                            st.session_state[f"success_time_{row['case_id']}"] = time.time()
                        
                        # Show success message for 5 seconds then remove
                        import time
                        if time.time() - st.session_state.get(f"success_time_{row['case_id']}", 0) > 5:
                            cases_to_remove.append(index)
                            if f"show_success_{row['case_id']}" in st.session_state:
                                del st.session_state[f"show_success_{row['case_id']}"]
                            if f"success_time_{row['case_id']}" in st.session_state:
                                del st.session_state[f"success_time_{row['case_id']}"]
                        else:
                            st.success(f"✅ Extraction completed for {row['case_name']}! Removing from list...")
                
                # Remove completed cases
                if cases_to_remove:
                    st.session_state.cases_to_sample = st.session_state.cases_to_sample.drop(cases_to_remove).reset_index(drop=True)
                    st.rerun()
                
                # Display remaining cases
                for index, row in st.session_state.cases_to_sample.iterrows():
                    with st.container():
                        st.subheader(f"{row['case_name']} ({row['citation']})")
                        
                        conn = sqlite3.connect(DB_NAME)
                        test_info = pd.read_sql(f"SELECT validation_status FROM legal_tests WHERE case_id = {row['case_id']}", conn)
                        conn.close()

                        if not test_info.empty:
                            # This case is extracted but not yet removed (showing success message)
                            continue
                        else:
                            st.write("**Status:** Not Extracted")
                            # Use a unique key for each button to avoid conflicts
                            if st.button(f"Run Extraction for {row['case_name']}", key=f"run_extraction_{row['case_id']}"):
                                st.session_state.confirming_extraction = row['case_id']
                                st.rerun()

                            if st.session_state.get('confirming_extraction') == row['case_id']:
                                if 'selected_gemini_model' not in st.session_state or not st.session_state.selected_gemini_model:
                                    st.error("No Gemini model selected. Please select one from the sidebar.")
                                    st.stop()
                                model_name = st.session_state.selected_gemini_model
                                
                                if model_name not in GEMINI_MODELS:
                                    st.warning(f"Pricing for selected model {model_name} is not available. Cost estimation will be approximate.")
                                    input_cost_per_million = 1.25
                                    output_cost_per_million = 10.0
                                else:
                                    model_pricing = GEMINI_MODELS[model_name]
                                    input_cost_per_million = model_pricing['input']
                                    output_cost_per_million = model_pricing['output']

                                input_tokens = len(row['full_text']) / 4 
                                output_tokens = 250 
                                estimated_cost = (input_tokens / 1_000_000 * input_cost_per_million) + (output_tokens / 1_000_000 * output_cost_per_million)
                                
                                if model_name == 'gemini-2.5-pro' and input_tokens > 200_000:
                                    high_volume_cost = (input_tokens / 1_000_000 * model_pricing['input_high_volume']) + (output_tokens / 1_000_000 * model_pricing['output_high_volume'])
                                    st.info(f"Standard pricing: ${estimated_cost:.4f} | High-volume pricing (>200K tokens): ${high_volume_cost:.4f}")
                                else:
                                    st.info(f"Estimated cost for this extraction: ${estimated_cost:.4f}")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button("Confirm and Run Extraction", key=f"confirm_run_{row['case_id']}"):
                                        try:
                                            with st.spinner("Running extraction..."):
                                                extracted_test_obj = run_extraction(row['full_text'])
                                                
                                                conn = sqlite3.connect(DB_NAME)
                                                c = conn.cursor()
                                                c.execute("INSERT INTO legal_tests (case_id, test_novelty, extracted_test_summary, source_paragraphs, source_type, validator_name) VALUES (?, ?, ?, ?, ?, ?)",
                                                          (row['case_id'], extracted_test_obj.test_novelty, extracted_test_obj.extracted_test_summary, extracted_test_obj.source_paragraphs, 'ai_extracted', st.session_state.validator_name))
                                                conn.commit()
                                                conn.close()
                                                
                                                st.session_state.confirming_extraction = None
                                                st.rerun()  # This will trigger the removal logic
                                        except Exception as e:
                                            st.error(f"An error occurred during extraction: {e}")
                                            st.session_state.confirming_extraction = None
                                with col2:
                                    if st.button("Cancel", key=f"cancel_run_{row['case_id']}"):
                                        st.session_state.confirming_extraction = None
                                        st.rerun()
                        st.divider()
            else:
                if 'validator_name' in st.session_state:
                    st.info("All cases in current session have been processed! Start a new session to extract more tests.")

    # Section 3: Overview of Extracted Tests
    overview_status = f"✅ {tests_count} tests" if tests_count > 0 else "📋 No tests yet"
    with st.expander(f"📊 **3. Overview of Extracted Tests** - {overview_status}", expanded=(tests_count > 0 and tests_count <= 5)):
        if tests_count == 0:
            st.info("No legal tests have been extracted yet. Load data and run extractions first.")
        else:
            # Pagination for overview table
            items_per_page = st.selectbox("Tests per page:", [5, 10, 20], index=1, key="overview_pagination")
            
            conn = sqlite3.connect(DB_NAME)
            overview_df = pd.read_sql("""
                SELECT 
                    c.decision_year,
                    c.case_name,
                    c.citation,
                    c.scc_url,
                    lt.test_novelty,
                    lt.extracted_test_summary,
                    lt.validation_status,
                    lt.test_id
                FROM cases c
                JOIN legal_tests lt ON c.case_id = lt.case_id
                ORDER BY c.decision_year DESC, c.case_name
            """, conn)
            conn.close()
            
            # Search functionality
            search_term = st.text_input("🔍 Search tests:", placeholder="Search by case name, citation, or test content...")
            if search_term:
                mask = (overview_df['case_name'].str.contains(search_term, case=False, na=False) |
                       overview_df['citation'].str.contains(search_term, case=False, na=False) |
                       overview_df['extracted_test_summary'].str.contains(search_term, case=False, na=False))
                overview_df = overview_df[mask]
            
            total_pages = (len(overview_df) + items_per_page - 1) // items_per_page
            if total_pages > 1:
                # Enhanced pagination with arrows and page numbers
                col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
                
                # Get current page from session state, defaulting to 0
                if 'overview_current_page' not in st.session_state:
                    st.session_state.overview_current_page = 0
                
                current_page = st.session_state.overview_current_page
                
                with col1:
                    if st.button("◀️ Prev", key="overview_prev", disabled=(current_page == 0)):
                        st.session_state.overview_current_page = max(0, current_page - 1)
                        st.rerun()
                
                with col2:
                    st.write(f"Page {current_page + 1}")
                
                with col3:
                    # Page selector dropdown as backup
                    selected_page = st.selectbox("Go to:", range(1, total_pages + 1), 
                                                index=current_page, key="overview_page_select") - 1
                    if selected_page != current_page:
                        st.session_state.overview_current_page = selected_page
                        st.rerun()
                
                with col4:
                    st.write(f"of {total_pages}")
                
                with col5:
                    if st.button("Next ▶️", key="overview_next", disabled=(current_page >= total_pages - 1)):
                        st.session_state.overview_current_page = min(total_pages - 1, current_page + 1)
                        st.rerun()
                
                page = current_page
            else:
                page = 0
                if 'overview_current_page' in st.session_state:
                    del st.session_state.overview_current_page
            
            start_idx = page * items_per_page
            end_idx = start_idx + items_per_page
            page_df = overview_df.iloc[start_idx:end_idx]
            
            for index, row in page_df.iterrows():
                with st.container():
                    # Check if this test card is expanded
                    is_expanded = st.session_state.get(f'expanded_test_{row["test_id"]}', False)
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{row['case_name']}** ({row['decision_year']}) - {row['citation']}")
                        
                        # Show truncated or full test summary with clickable ellipses
                        if len(row['extracted_test_summary']) > 200 and not is_expanded:
                            col_text, col_expand = st.columns([6, 1])
                            with col_text:
                                st.write(f"*{row['extracted_test_summary'][:200]}*")
                            with col_expand:
                                if st.button("...", key=f"expand_{row['test_id']}", help="Click to expand full details"):
                                    st.session_state[f'expanded_test_{row["test_id"]}'] = True
                                    st.rerun()
                        else:
                            st.write(f"*{row['extracted_test_summary']}*")
                            if is_expanded:
                                if st.button("🔼 Collapse", key=f"collapse_{row['test_id']}"):
                                    st.session_state[f'expanded_test_{row["test_id"]}'] = False
                                    st.rerun()
                        
                        # Show expanded details if card is expanded
                        if is_expanded:
                            st.write("**📋 Full Test Details:**")
                            st.write(f"**Test Novelty:** {row['test_novelty']}")
                            
                            # Get additional details from database
                            conn = sqlite3.connect(DB_NAME)
                            full_details = pd.read_sql(f"""
                                SELECT lt.*, c.area_of_law, c.full_text
                                FROM legal_tests lt 
                                JOIN cases c ON lt.case_id = c.case_id 
                                WHERE lt.test_id = {row['test_id']}
                            """, conn)
                            conn.close()
                            
                            if not full_details.empty:
                                detail = full_details.iloc[0]
                                st.write(f"**Source Paragraphs:** {detail['source_paragraphs']}")
                                st.write(f"**Source Type:** {detail['source_type']}")
                                st.write(f"**Area of Law:** {detail['area_of_law']}")
                                st.write(f"**Validator:** {detail['validator_name'] or 'Not set'}")
                                if detail['bt_score'] is not None:
                                    st.write(f"**Bradley-Terry Score:** {detail['bt_score']:.4f}")
                        
                        st.markdown(f"[🔗 SCC Decision]({row['scc_url']})")
                    
                    with col2:
                        status_color = "green" if row['validation_status'] == 'accurate' else "orange"
                        st.markdown(f"**Status:** :{status_color}[{row['validation_status'].title()}]")
                        if row['validation_status'] == 'pending':
                            if st.button(f"Validate", key=f"validate_overview_{row['test_id']}"):
                                st.session_state.test_to_edit = row
                                st.rerun()
                    st.divider()

    # Section 4: Human Validation
    conn = sqlite3.connect(DB_NAME)
    pending_tests = pd.read_sql("SELECT lt.*, c.case_name, c.scc_url FROM legal_tests lt JOIN cases c ON lt.case_id = c.case_id WHERE lt.validation_status = 'pending'", conn)
    conn.close()
    
    validation_status = f"✅ {validated_count} validated" if validated_count > 0 else f"🔄 {len(pending_tests)} pending" if len(pending_tests) > 0 else "📋 No tests yet"
    with st.expander(f"🔍 **4. Human Validation** - {validation_status}", expanded=(len(pending_tests) > 0 and validated_count < 5)):
        if pending_tests.empty:
            st.info("No tests are currently pending validation.")
        else:
            # Check if user has entered their name
            if not can_user_proceed():
                show_name_required_error()
            
            st.write(f"**{len(pending_tests)} tests pending validation**")
            
            # Create the validation table
            for index, row in pending_tests.iterrows():
                with st.container():
                    st.divider()
                    
                    # Header with case name and link
                    col_header1, col_header2 = st.columns([3, 1])
                    with col_header1:
                        st.subheader(f"{row['case_name']}")
                    with col_header2:
                        st.markdown(f"[🔗 SCC Decision]({row['scc_url']})")
                    
                    # Check if this test is being edited
                    is_editing = (st.session_state.get('editing_test_id') == row['test_id'])
                    
                    if is_editing:
                        # Edit mode
                        st.write("**Editing Mode**")
                        
                        edited_novelty = st.text_area(
                            "Test Novelty", 
                            value=row['test_novelty'], 
                            key=f"edit_novelty_{row['test_id']}"
                        )
                        
                        edited_summary = st.text_area(
                            "Extracted Test Summary", 
                            value=row['extracted_test_summary'], 
                            key=f"edit_summary_{row['test_id']}"
                        )
                        
                        edited_paragraphs = st.text_area(
                            "Source Paragraphs", 
                            value=row['source_paragraphs'], 
                            key=f"edit_paragraphs_{row['test_id']}"
                        )
                        
                        # Action buttons for edit mode
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            if st.button("💾 Save Changes", key=f"save_{row['test_id']}"):
                                conn = sqlite3.connect(DB_NAME)
                                c = conn.cursor()
                                c.execute("""UPDATE legal_tests SET 
                                           test_novelty = ?, 
                                           extracted_test_summary = ?, 
                                           source_paragraphs = ?, 
                                           source_type = 'human_edited', 
                                           validation_status = 'accurate', 
                                           validator_name = ? 
                                           WHERE test_id = ?""", 
                                          (edited_novelty, edited_summary, edited_paragraphs, 
                                           st.session_state.validator_name, row['test_id']))
                                conn.commit()
                                conn.close()
                                st.session_state.editing_test_id = None
                                st.success(f"Test for {row['case_name']} updated and marked as accurate.")
                                st.rerun()
                        
                        with col2:
                            if st.button("🔄 Re-run AI", key=f"rerun_{row['test_id']}"):
                                try:
                                    conn = sqlite3.connect(DB_NAME)
                                    case_data = pd.read_sql(f"SELECT full_text FROM cases WHERE case_id = {row['case_id']}", conn).iloc[0]
                                    full_text = case_data['full_text']
                                    
                                    with st.spinner(f"Re-running extraction for {row['case_name']}..."):
                                        extracted_test_obj = run_extraction(full_text)
                                        
                                        c = conn.cursor()
                                        c.execute("""UPDATE legal_tests SET 
                                                   test_novelty = ?, 
                                                   extracted_test_summary = ?, 
                                                   source_paragraphs = ?, 
                                                   source_type = 'ai_re_extracted', 
                                                   validation_status = 'pending', 
                                                   validator_name = ? 
                                                   WHERE test_id = ?""",
                                                  (extracted_test_obj.test_novelty, extracted_test_obj.extracted_test_summary, 
                                                   extracted_test_obj.source_paragraphs, st.session_state.validator_name, row['test_id']))
                                        conn.commit()
                                        conn.close()
                                    st.session_state.editing_test_id = None
                                    st.success(f"Re-extraction complete for {row['case_name']}!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error during re-extraction: {e}")
                        
                        with col3:
                            if st.button("❌ Cancel", key=f"cancel_{row['test_id']}"):
                                st.session_state.editing_test_id = None
                                st.rerun()
                        
                    else:
                        # Display mode
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"**Test Novelty:** {row['test_novelty']}")
                            st.write(f"**Extracted Test Summary:** {row['extracted_test_summary']}")
                            
                            # Truncate long source paragraphs for display
                            source_display = row['source_paragraphs']
                            if len(source_display) > 200:
                                source_display = source_display[:200] + "..."
                            st.write(f"**Source Paragraphs:** {source_display}")
                        
                        with col2:
                            st.write("**Actions:**")
                            if st.button("✅ Accurate", key=f"accurate_{row['test_id']}"):
                                conn = sqlite3.connect(DB_NAME)
                                c = conn.cursor()
                                c.execute("UPDATE legal_tests SET validation_status = 'accurate', validator_name = ? WHERE test_id = ?", 
                                          (st.session_state.validator_name, row['test_id']))
                                conn.commit()
                                conn.close()
                                st.success(f"Test marked as accurate.")
                                st.rerun()
                            
                            if st.button("❌ Edit", key=f"edit_{row['test_id']}"):
                                st.session_state.editing_test_id = row['test_id']
                                st.rerun()

    # Section 5: Pairwise Comparisons
    # Check for validated tests available for comparison
    conn = sqlite3.connect(DB_NAME)
    validated_tests = pd.read_sql("""
        SELECT lt.*, c.case_name, c.citation, c.scc_url 
        FROM legal_tests lt 
        JOIN cases c ON lt.case_id = c.case_id 
        WHERE lt.validation_status = 'accurate'
        ORDER BY c.decision_year DESC
    """, conn)
    
    comparison_status = f"✅ {comparisons_count} completed" if comparisons_count > 0 else f"🔄 Ready for comparison" if len(validated_tests) >= 2 else f"📋 Need {2 - len(validated_tests)} more validated tests"
    with st.expander(f"⚖️ **5. Pairwise Comparisons** - {comparison_status}", expanded=(len(validated_tests) >= 2 and comparisons_count < 10)):
        if len(validated_tests) < 2:
            st.warning("Need at least 2 validated legal tests to perform pairwise comparisons. Please validate more extractions first.")
        else:
            # Check if user has entered their name
            if not can_user_proceed():
                show_name_required_error()
            st.write(f"**{len(validated_tests)} validated tests available for comparison**")
            
            # Show comparison progress
            existing_comparisons = pd.read_sql("SELECT COUNT(*) as count FROM legal_test_comparisons", conn).iloc[0]['count']
            total_possible = len(validated_tests) * (len(validated_tests) - 1) // 2
            st.metric("Comparisons Completed", f"{existing_comparisons}/{total_possible}")
            
            if existing_comparisons < total_possible:
                # Get next pair to compare
                compared_pairs = pd.read_sql("""
                    SELECT test_id_1, test_id_2 FROM legal_test_comparisons
                    UNION
                    SELECT test_id_2 as test_id_1, test_id_1 as test_id_2 FROM legal_test_comparisons
                """, conn)
                
                # Find first uncompared pair
                next_pair = None
                for i, test1 in validated_tests.iterrows():
                    for j, test2 in validated_tests.iterrows():
                        if i < j:  # Only check each pair once
                            pair_exists = not compared_pairs[
                                ((compared_pairs['test_id_1'] == test1['test_id']) & 
                                 (compared_pairs['test_id_2'] == test2['test_id']))
                            ].empty
                            if not pair_exists:
                                next_pair = (test1, test2)
                                break
                    if next_pair:
                        break
                
                if next_pair:
                    test1, test2 = next_pair
                    st.subheader("Compare These Legal Tests")
                    st.write("**Which test is more 'rule-like'?** (Rule-like tests are highly prescriptive with little judicial discretion, while standard-like tests are more flexible)")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Test A: {test1['case_name']}**")
                        st.write(f"*Novelty:* {test1['test_novelty']}")
                        st.write(f"*Summary:* {test1['extracted_test_summary']}")
                        st.markdown(f"[🔗 Decision]({test1['scc_url']})")
                        
                        if st.button("Test A is More Rule-Like", key="test_a_wins"):
                            st.session_state.comparison_winner = test1['test_id']
                            st.session_state.comparison_loser = test2['test_id']
                            st.session_state.show_reasoning = True
                    
                    with col2:
                        st.write(f"**Test B: {test2['case_name']}**")
                        st.write(f"*Novelty:* {test2['test_novelty']}")
                        st.write(f"*Summary:* {test2['extracted_test_summary']}")
                        st.markdown(f"[🔗 Decision]({test2['scc_url']})")
                        
                        if st.button("Test B is More Rule-Like", key="test_b_wins"):
                            st.session_state.comparison_winner = test2['test_id']
                            st.session_state.comparison_loser = test1['test_id']
                            st.session_state.show_reasoning = True
                    
                    # Handle reasoning input and save comparison
                    if st.session_state.get('show_reasoning', False):
                        st.subheader("Explain Your Choice")
                        reasoning = st.text_area("Why is this test more rule-like?", key="comparison_reasoning")
                        
                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            if st.button("Save Comparison") and reasoning.strip():
                                c = conn.cursor()
                                c.execute("""INSERT INTO legal_test_comparisons 
                                           (test_id_1, test_id_2, more_rule_like_test_id, reasoning, comparator_name, comparison_method)
                                           VALUES (?, ?, ?, ?, ?, ?)""",
                                          (min(test1['test_id'], test2['test_id']), 
                                           max(test1['test_id'], test2['test_id']),
                                           st.session_state.comparison_winner, 
                                           reasoning, 
                                           st.session_state.get('validator_name', 'Unknown'), 
                                           'human'))
                                conn.commit()
                                st.success("Comparison saved!")
                                # Clear session state
                                del st.session_state.comparison_winner
                                del st.session_state.comparison_loser
                                del st.session_state.show_reasoning
                                st.rerun()
                        
                        with col_cancel:
                            if st.button("Cancel"):
                                # Clear session state
                                if 'comparison_winner' in st.session_state:
                                    del st.session_state.comparison_winner
                                if 'comparison_loser' in st.session_state:
                                    del st.session_state.comparison_loser
                                if 'show_reasoning' in st.session_state:
                                    del st.session_state.show_reasoning
                                st.rerun()
                else:
                    st.success("All pairwise comparisons completed! You can now run the Bradley-Terry analysis.")
            
            # AI-assisted comparison option
            if st.button("🤖 Generate AI Comparisons for Remaining Pairs"):
                if 'selected_gemini_model' not in st.session_state or not st.session_state.selected_gemini_model:
                    st.error("Please select a Gemini model first.")
                else:
                    # Get remaining pairs
                    compared_pairs = pd.read_sql("""
                        SELECT test_id_1, test_id_2 FROM legal_test_comparisons
                        UNION
                        SELECT test_id_2 as test_id_1, test_id_1 as test_id_2 FROM legal_test_comparisons
                    """, conn)
                    
                    remaining_pairs = []
                    for i, test1 in validated_tests.iterrows():
                        for j, test2 in validated_tests.iterrows():
                            if i < j:
                                pair_exists = not compared_pairs[
                                    ((compared_pairs['test_id_1'] == test1['test_id']) & 
                                     (compared_pairs['test_id_2'] == test2['test_id']))
                                ].empty
                                if not pair_exists:
                                    remaining_pairs.append((test1, test2))
                    
                    if remaining_pairs:
                        st.write(f"Generating AI comparisons for {len(remaining_pairs)} remaining pairs...")
                        progress_bar = st.progress(0)
                        
                        for idx, (test1, test2) in enumerate(remaining_pairs):
                            try:
                                comparison_result = run_comparison(test1['extracted_test_summary'], test2['extracted_test_summary'])
                                comparison_data = LegalTestComparison.parse_raw(comparison_result)
                                
                                # Determine which test was selected as more rule-like
                                # AI response should contain exactly "Test A" or "Test B"
                                if comparison_data.more_rule_like_test.strip().lower() == "test a":
                                    winner_id = test1['test_id']
                                elif comparison_data.more_rule_like_test.strip().lower() == "test b":
                                    winner_id = test2['test_id']
                                else:
                                    # Fallback: try to match against test content if format is unexpected
                                    st.warning(f"Unexpected AI response format: '{comparison_data.more_rule_like_test}'. Using fallback matching.")
                                    if comparison_data.more_rule_like_test.lower() in test1['extracted_test_summary'].lower():
                                        winner_id = test1['test_id']
                                    else:
                                        winner_id = test2['test_id']
                                
                                # Store the comparison with clear mapping
                                test_id_1 = min(test1['test_id'], test2['test_id'])
                                test_id_2 = max(test1['test_id'], test2['test_id'])
                                
                                # Create consistent reasoning that shows the mapping
                                reasoning_with_mapping = f"AI Comparison: Test A = {test1['case_name']}, Test B = {test2['case_name']}. {comparison_data.reasoning}"
                                
                                c = conn.cursor()
                                c.execute("""INSERT INTO legal_test_comparisons 
                                           (test_id_1, test_id_2, more_rule_like_test_id, reasoning, comparator_name, comparison_method)
                                           VALUES (?, ?, ?, ?, ?, ?)""",
                                          (test_id_1, test_id_2, winner_id, reasoning_with_mapping, 'AI', 'ai_generated'))
                                conn.commit()
                                
                                progress_bar.progress((idx + 1) / len(remaining_pairs))
                            except Exception as e:
                                st.error(f"Failed to compare tests {test1['case_name']} vs {test2['case_name']}: {e}")
                        
                        st.success(f"Generated {len(remaining_pairs)} AI comparisons!")
                        st.rerun()
                    else:
                        st.info("No remaining pairs to compare.")
            
            # Add diagnostic feature for AI comparison mapping issues
            st.write("---")
            if st.checkbox("🔧 Show Debug AI Comparisons (Advanced)", key="show_debug_comparisons"):
                st.write("This section helps diagnose and fix AI comparison mapping issues.")
                
                if st.button("Check AI Comparison Consistency"):
                    conn = sqlite3.connect(DB_NAME)
                    ai_comparisons = pd.read_sql("""
                        SELECT ltc.*, 
                               c1.case_name as case1_name, c2.case_name as case2_name,
                               winner_c.case_name as winner_name
                        FROM legal_test_comparisons ltc
                        JOIN legal_tests lt1 ON ltc.test_id_1 = lt1.test_id
                        JOIN legal_tests lt2 ON ltc.test_id_2 = lt2.test_id
                        JOIN legal_tests winner_lt ON ltc.more_rule_like_test_id = winner_lt.test_id
                        JOIN cases c1 ON lt1.case_id = c1.case_id
                        JOIN cases c2 ON lt2.case_id = c2.case_id
                        JOIN cases winner_c ON winner_lt.case_id = winner_c.case_id
                        WHERE ltc.comparison_method = 'ai_generated'
                        AND ltc.reasoning NOT LIKE 'AI Comparison: Test A =%'
                    """, conn)
                    conn.close()
                    
                    if not ai_comparisons.empty:
                        st.warning(f"Found {len(ai_comparisons)} AI comparisons that may have mapping issues:")
                        for _, comp in ai_comparisons.iterrows():
                            st.write(f"**Comparison #{comp['comparison_id']}**: Winner = {comp['winner_name']}")
                            st.write(f"Reasoning: {comp['reasoning'][:200]}...")
                            st.write("---")
                    else:
                        st.success("All AI comparisons appear to have consistent mapping!")
            
            # Comparison Review Table - as collapsible subsection
            st.write("---")
            review_status = f"📊 {comparisons_count} comparisons" if comparisons_count > 0 else "📋 No comparisons yet"
            if st.checkbox(f"**Review Completed Comparisons** - {review_status}", key="show_comparison_review", value=(comparisons_count > 0 and comparisons_count <= 10)):
                if comparisons_count == 0:
                    st.info("No comparisons completed yet. Complete some comparisons above to see them here.")
                else:
                    # Get all completed comparisons with case details - open new connection
                    conn = sqlite3.connect(DB_NAME)
                    completed_comparisons = pd.read_sql("""
                        SELECT 
                            ltc.*,
                            c1.case_name as case1_name, c1.citation as case1_citation, c1.decision_year as case1_year,
                            c2.case_name as case2_name, c2.citation as case2_citation, c2.decision_year as case2_year,
                            lt1.extracted_test_summary as test1_summary,
                            lt2.extracted_test_summary as test2_summary,
                            winner_c.case_name as winner_name,
                            winner_lt.extracted_test_summary as winner_summary
                        FROM legal_test_comparisons ltc
                        JOIN legal_tests lt1 ON ltc.test_id_1 = lt1.test_id
                        JOIN legal_tests lt2 ON ltc.test_id_2 = lt2.test_id
                        JOIN legal_tests winner_lt ON ltc.more_rule_like_test_id = winner_lt.test_id
                        JOIN cases c1 ON lt1.case_id = c1.case_id
                        JOIN cases c2 ON lt2.case_id = c2.case_id
                        JOIN cases winner_c ON winner_lt.case_id = winner_c.case_id
                        ORDER BY ltc.timestamp DESC
                    """, conn)
                    
                    if not completed_comparisons.empty:
                        st.write(f"**{len(completed_comparisons)} completed comparisons:**")
                        
                        # Pagination for comparisons
                        comp_items_per_page = st.selectbox("Comparisons per page:", [5, 10, 20], index=1, key="comparison_pagination")
                        comp_total_pages = (len(completed_comparisons) + comp_items_per_page - 1) // comp_items_per_page
                        
                        if comp_total_pages > 1:
                            # Enhanced pagination with arrows for comparisons
                            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
                            
                            # Get current page from session state
                            if 'comparison_current_page' not in st.session_state:
                                st.session_state.comparison_current_page = 0
                            
                            comp_current_page = st.session_state.comparison_current_page
                            
                            with col1:
                                if st.button("◀️ Prev", key="comparison_prev", disabled=(comp_current_page == 0)):
                                    st.session_state.comparison_current_page = max(0, comp_current_page - 1)
                                    st.rerun()
                            
                            with col2:
                                st.write(f"Page {comp_current_page + 1}")
                            
                            with col3:
                                selected_comp_page = st.selectbox("Go to:", range(1, comp_total_pages + 1), 
                                                                 index=comp_current_page, key="comparison_page_select") - 1
                                if selected_comp_page != comp_current_page:
                                    st.session_state.comparison_current_page = selected_comp_page
                                    st.rerun()
                            
                            with col4:
                                st.write(f"of {comp_total_pages}")
                            
                            with col5:
                                if st.button("Next ▶️", key="comparison_next", disabled=(comp_current_page >= comp_total_pages - 1)):
                                    st.session_state.comparison_current_page = min(comp_total_pages - 1, comp_current_page + 1)
                                    st.rerun()
                            
                            comp_page = comp_current_page
                        else:
                            comp_page = 0
                            if 'comparison_current_page' in st.session_state:
                                del st.session_state.comparison_current_page
                        
                        # Get page data
                        comp_start_idx = comp_page * comp_items_per_page
                        comp_end_idx = comp_start_idx + comp_items_per_page
                        comp_page_df = completed_comparisons.iloc[comp_start_idx:comp_end_idx]
                        
                        for idx, comp in comp_page_df.iterrows():
                            with st.container():
                                st.divider()
                            
                            # Header with comparison info
                            col_header1, col_header2, col_header3 = st.columns([3, 1, 1])
                            with col_header1:
                                st.write(f"**Comparison #{comp['comparison_id']}** - {comp['comparison_method'].title()}")
                            with col_header2:
                                st.write(f"*{comp['timestamp'][:10]}*")
                            with col_header3:
                                method_emoji = "🤖" if comp['comparison_method'] == 'ai_generated' else "👤"
                                st.write(f"{method_emoji} {comp['comparator_name']}")
                            
                            # Test comparison details
                            col1, col2, col3 = st.columns([2, 2, 1])
                            
                            with col1:
                                st.write("**📋 Test A:**")
                                st.write(f"*{comp['case1_name']}* ({comp['case1_year']})")
                                st.write(f"Citation: {comp['case1_citation']}")
                                
                                # Expandable test summary for Test A
                                is_test_a_expanded = st.session_state.get(f'expanded_test_a_{comp["comparison_id"]}', False)
                                if len(comp['test1_summary']) > 150 and not is_test_a_expanded:
                                    col_text_a, col_expand_a = st.columns([6, 1])
                                    with col_text_a:
                                        st.write(f"Test: {comp['test1_summary'][:150]}")
                                    with col_expand_a:
                                        if st.button("...", key=f"expand_test_a_{comp['comparison_id']}", help="Click to see full test"):
                                            st.session_state[f'expanded_test_a_{comp["comparison_id"]}'] = True
                                            st.rerun()
                                else:
                                    st.write(f"Test: {comp['test1_summary']}")
                                    if is_test_a_expanded:
                                        if st.button("🔼", key=f"collapse_test_a_{comp['comparison_id']}", help="Collapse"):
                                            st.session_state[f'expanded_test_a_{comp["comparison_id"]}'] = False
                                            st.rerun()
                            
                            with col2:
                                st.write("**📋 Test B:**")
                                st.write(f"*{comp['case2_name']}* ({comp['case2_year']})")
                                st.write(f"Citation: {comp['case2_citation']}")
                                
                                # Expandable test summary for Test B
                                is_test_b_expanded = st.session_state.get(f'expanded_test_b_{comp["comparison_id"]}', False)
                                if len(comp['test2_summary']) > 150 and not is_test_b_expanded:
                                    col_text_b, col_expand_b = st.columns([6, 1])
                                    with col_text_b:
                                        st.write(f"Test: {comp['test2_summary'][:150]}")
                                    with col_expand_b:
                                        if st.button("...", key=f"expand_test_b_{comp['comparison_id']}", help="Click to see full test"):
                                            st.session_state[f'expanded_test_b_{comp["comparison_id"]}'] = True
                                            st.rerun()
                                else:
                                    st.write(f"Test: {comp['test2_summary']}")
                                    if is_test_b_expanded:
                                        if st.button("🔼", key=f"collapse_test_b_{comp['comparison_id']}", help="Collapse"):
                                            st.session_state[f'expanded_test_b_{comp["comparison_id"]}'] = False
                                            st.rerun()
                            
                            with col3:
                                st.write("**🏆 Winner:**")
                                st.success(f"**{comp['winner_name']}**")
                                st.write("*More Rule-Like*")
                            
                            # Reasoning
                            st.write("**💭 Reasoning:**")
                            st.write(f"*{comp['reasoning']}*")
                            
                            # Validation actions (only for AI comparisons)
                            if comp['comparison_method'] == 'ai_generated':
                                st.write("**Validate AI Comparison:**")
                                col_val1, col_val2, col_val3 = st.columns(3)
                                
                                with col_val1:
                                    if st.button("✅ Approve", key=f"approve_{comp['comparison_id']}"):
                                        # Mark as human validated (could add a validation_status column)
                                        c = conn.cursor()
                                        c.execute("UPDATE legal_test_comparisons SET comparator_name = ? WHERE comparison_id = ?", 
                                                 (f"{comp['comparator_name']} (Validated)", comp['comparison_id']))
                                        conn.commit()
                                        st.success("Comparison approved!")
                                        st.rerun()
                                
                                with col_val2:
                                    if st.button("❌ Override", key=f"override_{comp['comparison_id']}"):
                                        st.session_state[f'overriding_comparison_{comp["comparison_id"]}'] = True
                                        st.rerun()
                                
                                with col_val3:
                                    if st.button("🗑️ Delete", key=f"delete_{comp['comparison_id']}"):
                                        c = conn.cursor()
                                        c.execute("DELETE FROM legal_test_comparisons WHERE comparison_id = ?", (comp['comparison_id'],))
                                        conn.commit()
                                        st.success("Comparison deleted!")
                                        st.rerun()
                                
                                # Handle override mode
                                if st.session_state.get(f'overriding_comparison_{comp["comparison_id"]}', False):
                                    st.write("**Override this comparison:**")
                                    
                                    # Determine which test should win instead
                                    other_test_id = comp['test_id_2'] if comp['more_rule_like_test_id'] == comp['test_id_1'] else comp['test_id_1']
                                    other_test_name = comp['case2_name'] if comp['more_rule_like_test_id'] == comp['test_id_1'] else comp['case1_name']
                                    
                                    new_reasoning = st.text_area("New reasoning:", key=f"new_reason_{comp['comparison_id']}")
                                    
                                    col_override1, col_override2 = st.columns(2)
                                    with col_override1:
                                        if st.button(f"Make {other_test_name} Winner", key=f"flip_{comp['comparison_id']}") and new_reasoning.strip():
                                            c = conn.cursor()
                                            c.execute("""UPDATE legal_test_comparisons 
                                                       SET more_rule_like_test_id = ?, reasoning = ?, 
                                                           comparator_name = ?, comparison_method = 'human_override' 
                                                       WHERE comparison_id = ?""", 
                                                     (other_test_id, new_reasoning, st.session_state.get('validator_name', 'Human'), comp['comparison_id']))
                                            conn.commit()
                                            del st.session_state[f'overriding_comparison_{comp["comparison_id"]}']
                                            st.success("Comparison updated!")
                                            st.rerun()
                                    
                                    with col_override2:
                                        if st.button("Cancel Override", key=f"cancel_{comp['comparison_id']}"):
                                            del st.session_state[f'overriding_comparison_{comp["comparison_id"]}']
                                        st.rerun()
                
                conn.close()

    # Section 6: Analysis
    # Determine analysis status with out-of-date detection
    if 'analysis_results' in st.session_state and st.session_state.analysis_results is not None:
        stored_count = st.session_state.get('analysis_comparisons_count', 0)
        if comparisons_count > stored_count:
            new_comparisons = comparisons_count - stored_count
            analysis_status = f"⚠️ Analysis out of date ({new_comparisons} new comparison{'s' if new_comparisons != 1 else ''})"
        else:
            analysis_status = f"✅ Analysis current"
    elif validated_count >= 2 and comparisons_count > 0:
        analysis_status = f"✅ Analysis available"
    else:
        analysis_status = f"📋 Need validated tests & comparisons"
    
    # Initialize analysis expander state in session state
    if 'analysis_expander_open' not in st.session_state:
        st.session_state.analysis_expander_open = (validated_count >= 2 and comparisons_count > 0)
    
    with st.expander(f"📈 **6. Analysis** - {analysis_status}", expanded=st.session_state.analysis_expander_open):
        
        # Show analysis results if they exist in session state
        if 'analysis_results' in st.session_state and st.session_state.analysis_results is not None:
            # Display the stored analysis results
            results_data = st.session_state.analysis_results
            analyzed_df = results_data['analyzed_df']
            comparison_data = results_data['comparison_data']
            graph_analysis = results_data['graph_analysis']
            
            st.subheader("Rule-Likeness Analysis Results")
            
            if len(analyzed_df) > 0:
                st.write(f"**{len(analyzed_df)} tests successfully analyzed:**")
                
                # Display results table for analyzed tests with full test summaries
                display_df = analyzed_df[['case_name', 'citation', 'decision_year', 'extracted_test_summary', 'bt_score']].copy()
                display_df['bt_score'] = display_df['bt_score'].round(4)
                display_df.columns = ['Case Name', 'Citation', 'Decision Year', 'Extracted Test Summary', 'Rule-Likeness Score']
                st.dataframe(display_df, use_container_width=True)
                
                # Add temporal analysis and comprehensive stats here
                enhanced_stats = {
                    'bt_statistics': results_data.get('bt_statistics'),
                    'cv_results': results_data.get('cv_results'),
                    'inconsistency_results': results_data.get('inconsistency_results'),
                    'params': results_data.get('params'),
                    'n_items': results_data.get('n_items')
                }
                _display_temporal_analysis_and_stats(analyzed_df, comparison_data, graph_analysis, enhanced_stats)
            
            # Button to clear results and run new analysis
            if st.button("🔄 Run New Analysis"):
                del st.session_state.analysis_results
                if 'analysis_comparisons_count' in st.session_state:
                    del st.session_state.analysis_comparisons_count
                st.rerun()
        
        elif st.button("Run Analysis"):
            # Check if user has entered their name
            if not can_user_proceed():
                show_name_required_error()
            
            conn = sqlite3.connect(DB_NAME)
            
            # Get validated tests and their comparisons
            validated_tests = pd.read_sql("""
                SELECT lt.*, c.case_name, c.citation, c.decision_year
                FROM legal_tests lt 
                JOIN cases c ON lt.case_id = c.case_id 
                WHERE lt.validation_status = 'accurate'
                ORDER BY lt.test_id
            """, conn)
            
            comparisons = pd.read_sql("""
                SELECT test_id_1, test_id_2, more_rule_like_test_id 
                FROM legal_test_comparisons
            """, conn)
            
            if len(validated_tests) < 2:
                st.error("Need at least 2 validated legal tests to run analysis.")
                conn.close()
                st.stop()
                
            if len(comparisons) == 0:
                st.error("No pairwise comparisons found. Please complete some comparisons first.")
                conn.close()
                st.stop()
            
            try:
                # Create mapping from test_id to index for choix
                test_id_to_idx = {test_id: idx for idx, test_id in enumerate(validated_tests['test_id'])}
                idx_to_test_id = {idx: test_id for test_id, idx in test_id_to_idx.items()}
                
                # Build comparison data for choix
                # Each comparison (i, j) means item i beat item j (i is more rule-like)
                comparison_data = []
                for _, comp in comparisons.iterrows():
                    test1_id = comp['test_id_1']
                    test2_id = comp['test_id_2']
                    winner_id = comp['more_rule_like_test_id']
                    
                    # Only include if both tests are in our validated set
                    if test1_id in test_id_to_idx and test2_id in test_id_to_idx:
                        winner_idx = test_id_to_idx[winner_id]
                        loser_id = test2_id if winner_id == test1_id else test1_id
                        loser_idx = test_id_to_idx[loser_id]
                        comparison_data.append((winner_idx, loser_idx))
                
                if len(comparison_data) == 0:
                    st.error("No valid comparisons found between validated tests.")
                    conn.close()
                    st.stop()
                
                n_items = len(validated_tests)
                st.write(f"**Analyzing {n_items} legal tests with {len(comparison_data)} pairwise comparisons**")
                
                # Enhanced connectivity checking
                def analyze_graph_structure(n_items, comparisons):
                    """Analyze the structure of the comparison graph"""
                    # Build adjacency list
                    graph = [[] for _ in range(n_items)]
                    degree = [0] * n_items
                    
                    for winner, loser in comparisons:
                        if winner not in graph[loser]:
                            graph[winner].append(loser)
                            graph[loser].append(winner)
                            degree[winner] += 1
                            degree[loser] += 1
                    
                    # Find connected components using DFS
                    visited = [False] * n_items
                    components = []
                    
                    def dfs(node, component):
                        visited[node] = True
                        component.append(node)
                        for neighbor in graph[node]:
                            if not visited[neighbor]:
                                dfs(neighbor, component)
                    
                    for i in range(n_items):
                        if not visited[i]:
                            component = []
                            dfs(i, component)
                            components.append(component)
                    
                    return {
                        'components': components,
                        'is_connected': len(components) == 1,
                        'degree': degree,
                        'min_degree': min(degree) if degree else 0,
                        'max_degree': max(degree) if degree else 0,
                        'isolated_nodes': [i for i, d in enumerate(degree) if d == 0]
                    }
                
                # Check minimum data requirements
                min_comparisons_needed = n_items - 1  # Minimum for connectivity
                recommended_comparisons = n_items * (n_items - 1) // 4  # At least 25% of all possible pairs
                
                if len(comparison_data) < min_comparisons_needed:
                    st.error(f"❌ Insufficient comparison data: {len(comparison_data)} comparisons found, but need at least {min_comparisons_needed} for {n_items} tests.")
                    st.info("💡 **Tip**: Each test needs to be compared with at least one other test to form a connected graph.")
                    conn.close()
                    st.stop()
                
                graph_analysis = analyze_graph_structure(n_items, comparison_data)
                
                # Display graph analysis
                st.subheader("Graph Structure Analysis")
                col1, col2, col3 = st.columns(3)
                with col1:
                    connectivity_status = "✅ Connected" if graph_analysis['is_connected'] else "❌ Disconnected"
                    st.metric("Graph Connectivity", connectivity_status)
                with col2:
                    st.metric("Connected Components", len(graph_analysis['components']))
                with col3:
                    st.metric("Min/Max Degree", f"{graph_analysis['min_degree']}/{graph_analysis['max_degree']}")
                
                if graph_analysis['isolated_nodes']:
                    isolated_names = [validated_tests.iloc[i]['case_name'] for i in graph_analysis['isolated_nodes']]
                    st.warning(f"⚠️ Isolated tests (no comparisons): {', '.join(isolated_names)}")
                
                if not graph_analysis['is_connected']:
                    st.warning("⚠️ Comparison graph has multiple disconnected components:")
                    for i, component in enumerate(graph_analysis['components']):
                        component_names = [validated_tests.iloc[j]['case_name'] for j in component]
                        st.write(f"   **Component {i+1}**: {', '.join(component_names)}")
                    st.info("💡 **Recommendation**: Add comparisons between tests from different components to connect the graph.")
                
                if len(comparison_data) < recommended_comparisons:
                    st.warning(f"⚠️ Limited comparison data: {len(comparison_data)}/{recommended_comparisons} recommended comparisons. Consider adding more for robust results.")
                
                # Attempt Bradley-Terry analysis with fallback options
                analysis_successful = False
                
                try:
                    # Try standard Bradley-Terry model
                    st.write("🔄 Attempting Bradley-Terry analysis...")
                    params = choix.ilsr_pairwise(n_items, comparison_data, alpha=0.01, max_iter=10000)
                    
                    # Compute enhanced statistical metrics
                    st.write("🔄 Computing statistical diagnostics...")
                    bt_statistics = _compute_bradley_terry_statistics(n_items, comparison_data, params)
                    
                    # Cross-validation analysis
                    cv_results = _cross_validate_bradley_terry(n_items, comparison_data, n_folds=5)
                    
                    # Inconsistency analysis
                    inconsistency_results = _analyze_inconsistencies(comparison_data, n_items)
                    
                    analysis_successful = True
                    st.success("✅ Bradley-Terry analysis completed successfully!")
                    
                except Exception as bt_error:
                    st.error(f"❌ Bradley-Terry analysis failed: {bt_error}")
                    
                    # Try alternative approach: largest connected component only
                    if not graph_analysis['is_connected'] and len(graph_analysis['components']) > 1:
                        largest_component = max(graph_analysis['components'], key=len)
                        if len(largest_component) >= 2:
                            st.info(f"🔄 Attempting analysis on largest connected component ({len(largest_component)} tests)...")
                            
                            # Filter data to largest component
                            component_set = set(largest_component)
                            filtered_comparisons = [(w, l) for w, l in comparison_data 
                                                  if w in component_set and l in component_set]
                            
                            # Remap indices for the component
                            old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(largest_component)}
                            remapped_comparisons = [(old_to_new[w], old_to_new[l]) for w, l in filtered_comparisons]
                            
                            try:
                                params_component = choix.ilsr_pairwise(len(largest_component), remapped_comparisons, alpha=0.01)
                                
                                # Create full params array with NaN for non-component items
                                params = [float('nan')] * n_items
                                for i, component_idx in enumerate(largest_component):
                                    params[component_idx] = params_component[i]
                                
                                analysis_successful = True
                                st.success(f"✅ Analysis completed on largest component ({len(largest_component)} tests)!")
                                st.warning("⚠️ Tests not in the largest component will show NaN scores.")
                                
                            except Exception as component_error:
                                st.error(f"❌ Component analysis also failed: {component_error}")
                    
                    # If all else fails, use simple win-rate analysis
                    if not analysis_successful:
                        st.info("🔄 Falling back to simple win-rate analysis...")
                        try:
                            # Calculate win rates for each test
                            wins = [0] * n_items
                            total_comparisons = [0] * n_items
                            
                            for winner, loser in comparison_data:
                                wins[winner] += 1
                                total_comparisons[winner] += 1
                                total_comparisons[loser] += 1
                            
                            # Calculate win rates (with smoothing for tests with no comparisons)
                            params = []
                            for i in range(n_items):
                                if total_comparisons[i] > 0:
                                    win_rate = wins[i] / total_comparisons[i]
                                    # Convert to log-odds scale to mimic Bradley-Terry scores
                                    if win_rate == 0:
                                        params.append(-2.0)  # Very low score
                                    elif win_rate == 1:
                                        params.append(2.0)   # Very high score
                                    else:
                                        import math
                                        params.append(math.log(win_rate / (1 - win_rate)))
                                else:
                                    params.append(0.0)  # Neutral score for uncompared tests
                            
                            analysis_successful = True
                            st.success("✅ Win-rate analysis completed!")
                            st.info("📊 **Note**: Using win-rate analysis instead of Bradley-Terry due to data limitations.")
                            
                        except Exception as fallback_error:
                            st.error(f"❌ All analysis methods failed: {fallback_error}")
                            st.stop()
                
                if not analysis_successful:
                    st.error("❌ Unable to perform any analysis. Please add more comparison data.")
                    conn.close()
                    st.stop()
                
                # Create results DataFrame
                results_df = validated_tests.copy()
                results_df['bt_score'] = [params[test_id_to_idx[test_id]] for test_id in results_df['test_id']]
                
                # Handle NaN values and sort
                import pandas as pd
                import numpy as np
                results_df['bt_score'] = results_df['bt_score'].replace([np.inf, -np.inf], np.nan)
                
                # Separate analyzed and unanalyzed tests
                analyzed_df = results_df.dropna(subset=['bt_score']).sort_values('bt_score', ascending=False)
                unanalyzed_df = results_df[results_df['bt_score'].isna()]
                
                st.subheader("Rule-Likeness Analysis Results")
                
                if len(analyzed_df) > 0:
                    st.write(f"**{len(analyzed_df)} tests successfully analyzed:**")
                    
                    # Display results table for analyzed tests with full test summaries
                    display_df = analyzed_df[['case_name', 'citation', 'decision_year', 'extracted_test_summary', 'bt_score']].copy()
                    display_df['bt_score'] = display_df['bt_score'].round(4)
                    # Keep full test summary - no truncation for analysis results
                    display_df.columns = ['Case Name', 'Citation', 'Decision Year', 'Extracted Test Summary', 'Rule-Likeness Score']
                    st.dataframe(display_df, use_container_width=True)
                    
                    if len(unanalyzed_df) > 0:
                        with st.expander(f"📋 {len(unanalyzed_df)} tests not analyzed (disconnected from main component)"):
                            unanalyzed_display = unanalyzed_df[['case_name', 'citation', 'decision_year']].copy()
                            unanalyzed_display.columns = ['Case Name', 'Citation', 'Decision Year']
                            st.dataframe(unanalyzed_display, use_container_width=True)
                            st.info("💡 These tests need comparisons with tests in the main component to be included in the analysis.")
                else:
                    st.error("No tests could be analyzed. This suggests a fundamental issue with the comparison data.")
                
                # Update bt_scores in database (only for non-NaN values)
                c = conn.cursor()
                for _, row in results_df.iterrows():
                    if pd.notna(row['bt_score']):
                        c.execute("UPDATE legal_tests SET bt_score = ? WHERE test_id = ?", 
                                 (row['bt_score'], row['test_id']))
                conn.commit()
                
                # Store analysis results in session state
                st.session_state.analysis_results = {
                    'analyzed_df': analyzed_df,
                    'comparison_data': comparison_data,
                    'graph_analysis': graph_analysis,
                    'bt_statistics': bt_statistics if 'bt_statistics' in locals() else None,
                    'cv_results': cv_results if 'cv_results' in locals() else None,
                    'inconsistency_results': inconsistency_results if 'inconsistency_results' in locals() else None,
                    'params': params if 'params' in locals() else None,
                    'n_items': n_items
                }
                
                # Store comparison count when analysis was run for out-of-date detection
                st.session_state.analysis_comparisons_count = len(comparison_data)
                
                # Display temporal analysis and comprehensive statistics
                enhanced_stats = {
                    'bt_statistics': bt_statistics if 'bt_statistics' in locals() else None,
                    'cv_results': cv_results if 'cv_results' in locals() else None,
                    'inconsistency_results': inconsistency_results if 'inconsistency_results' in locals() else None,
                    'params': params if 'params' in locals() else None,
                    'n_items': n_items
                }
                _display_temporal_analysis_and_stats(analyzed_df, comparison_data, graph_analysis, enhanced_stats)

            except Exception as e:
                st.error(f"❌ **Critical Analysis Error**: {e}")
                st.write("**Possible causes:**")
                st.write("• Insufficient comparison data for stable analysis")
                st.write("• Circular comparison patterns (A beats B, B beats C, C beats A)")
                st.write("• Completely disconnected comparison graph")
                st.write("• Invalid data in comparison database")
                
                # Enhanced debug information
                if st.checkbox("🔍 **Show Detailed Debug Information**", key="show_debug_info"):
                    st.write("**Data Overview:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Validated Tests", len(validated_tests))
                    with col2:
                        st.metric("Raw Comparisons", len(comparisons))
                    with col3:
                        st.metric("Valid Comparisons", len(comparison_data) if 'comparison_data' in locals() else 'N/A')
                    
                    if 'comparison_data' in locals():
                        st.write("**Sample Comparison Data (first 10):**")
                        sample_comparisons = []
                        for i, (winner_idx, loser_idx) in enumerate(comparison_data[:10]):
                            winner_name = validated_tests.iloc[winner_idx]['case_name']
                            loser_name = validated_tests.iloc[loser_idx]['case_name']
                            sample_comparisons.append(f"{i+1}. {winner_name} > {loser_name}")
                        st.write("\n".join(sample_comparisons))
                        
                        # Graph connectivity debug
                        if 'graph_analysis' in locals():
                            st.write("**Graph Structure:**")
                            st.write(f"Connected: {graph_analysis['is_connected']}")
                            st.write(f"Components: {len(graph_analysis['components'])}")
                            for i, component in enumerate(graph_analysis['components']):
                                component_names = [validated_tests.iloc[j]['case_name'] for j in component]
                                st.write(f"   Component {i+1}: {', '.join(component_names)}")
                    
                    st.write("**Raw Comparison Records:**")
                    st.dataframe(comparisons.head(10), use_container_width=True)
                    
                    st.write("**Test ID Mapping:**")
                    if 'test_id_to_idx' in locals():
                        mapping_df = pd.DataFrame([
                            {'Test ID': tid, 'Index': idx, 'Case Name': validated_tests.iloc[idx]['case_name']}
                            for tid, idx in test_id_to_idx.items()
                        ])
                        st.dataframe(mapping_df, use_container_width=True)

        conn.close()
