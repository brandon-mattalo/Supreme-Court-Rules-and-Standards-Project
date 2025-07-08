"""
Test script for case selection functionality
"""
import streamlit as st
from utils.case_management import (
    get_database_counts, get_experiment_selected_cases, 
    get_available_cases_for_selection, add_cases_to_experiments
)

st.title("🎯 Case Selection Test")

# Get counts
total_cases, selected_cases, tests_count, comparisons_count, validated_count = get_database_counts()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Cases", total_cases)
with col2:
    st.metric("Selected Cases", selected_cases)
with col3:
    st.metric("Available for Selection", total_cases - selected_cases)

st.divider()

# Test case selection
st.subheader("➕ Add Cases to Experiments")

# Get sample for filters
sample_cases = get_available_cases_for_selection(limit=100)
if not sample_cases.empty:
    min_year = int(sample_cases['decision_year'].min())
    max_year = int(sample_cases['decision_year'].max())
    
    # Filters
    year_filter = st.slider("Year Range", min_year, max_year, (min_year, max_year))
    
    unique_areas = sample_cases['area_of_law'].dropna().unique()
    area_filter = st.multiselect("Areas of Law (optional)", unique_areas)
    
    # Count available
    available_count = len(get_available_cases_for_selection(
        year_range=year_filter if year_filter != (min_year, max_year) else None,
        areas=area_filter if area_filter else None
    ))
    
    num_to_select = st.number_input(
        f"Number to select (max {min(available_count, 50)})",
        min_value=1,
        max_value=min(available_count, 50),
        value=min(10, available_count)
    )
    
    st.info(f"Will randomly select {num_to_select} cases from {available_count} available")
    
    if st.button("🎲 Select Random Cases", type="primary"):
        with st.spinner("Selecting cases..."):
            new_cases = get_available_cases_for_selection(
                year_range=year_filter if year_filter != (min_year, max_year) else None,
                areas=area_filter if area_filter else None,
                limit=num_to_select
            )
            
            if not new_cases.empty:
                success_count, duplicate_count = add_cases_to_experiments(new_cases['case_id'].tolist())
                st.success(f"✅ Added {success_count} cases!")
                
                # Show what was added
                st.dataframe(new_cases[['case_name', 'citation', 'decision_year', 'area_of_law']])
                
                st.rerun()
            else:
                st.error("No cases found")

# Show selected cases
st.divider()
st.subheader("📋 Currently Selected Cases")
selected_df = get_experiment_selected_cases()
if not selected_df.empty:
    st.dataframe(selected_df[['case_name', 'citation', 'decision_year', 'area_of_law']])
else:
    st.info("No cases selected yet")