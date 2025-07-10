"""
Case Management V2 with Sample Groups
Replaces the old case selection interface with a sample group-based approach
"""

import streamlit as st
import pandas as pd
from utils.case_management import get_database_counts, load_data_from_parquet, clear_database, get_available_cases_for_selection
from utils.sample_group_management import (
    create_sample_group, get_sample_groups, get_sample_group_members,
    add_cases_to_sample_group, remove_cases_from_sample_group, delete_sample_group,
    get_available_cases_for_group, duplicate_sample_group, get_sample_group_statistics
)
from config import execute_sql
import numpy as np

def show_case_management_v2():
    """Show experiment case selection interface with sample groups"""
    st.header("📚 Experiment Case Selection")
    st.markdown("*Organize cases into sample groups for your experiments*")
    
    # Get database counts
    total_cases, selected_cases, tests_count, comparisons_count, validated_count = get_database_counts()
    
    # 1. Data Management (same as before)
    with st.expander("1️⃣ **Data Management**", expanded=False):
        # Admin password protection
        admin_password = st.text_input("🔐 Admin Password (required for data operations)", 
                                     type="password", key="admin_password_dashboard_v2")
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
                                st.rerun()
                    
                    with col_b:
                        start_batch = st.number_input("Start Batch", min_value=1, value=1, 
                                                    step=1, help="Skip to batch number (100 cases per batch)")
                        if st.button("⏭️ Load from Batch", type="secondary"):
                            with st.spinner(f"Loading from batch {start_batch}..."):
                                load_data_from_parquet(uploaded_file, start_batch=start_batch)
                                st.rerun()
            
            with col2:
                st.subheader("🗑️ Database Management")
                
                # Clear all data button
                if st.button("🗑️ Clear All Data", disabled=not is_admin, type="secondary"):
                    if st.checkbox("I understand this will delete ALL data", key="confirm_clear_all_v2"):
                        if clear_database():
                            st.rerun()
    
    # 2. Database Overview
    with st.expander("2️⃣ **Database Overview**", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Cases Available", f"{total_cases:,}")
        with col2:
            # Count total unique cases across all sample groups
            unique_cases_in_groups = execute_sql("""
                SELECT COUNT(DISTINCT case_id) FROM v2_sample_group_members
            """, fetch=True)[0][0]
            st.metric("Cases in Sample Groups", f"{unique_cases_in_groups:,}")
        with col3:
            st.metric("Extracted Tests", f"{tests_count:,}")
        with col4:
            st.metric("Comparisons Made", f"{comparisons_count:,}")
    
    # 3. Sample Groups Management
    st.header("3️⃣ **Sample Groups**")
    
    # Get existing sample groups
    sample_groups_df = get_sample_groups()
    
    # Create new sample group
    with st.expander("➕ Create New Sample Group", expanded=False):
        with st.form("create_sample_group_form"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                new_group_name = st.text_input("Group Name", placeholder="e.g., Criminal Law 2020-2024")
                new_group_desc = st.text_area("Description (optional)", 
                                             placeholder="Describe the purpose or criteria for this sample group")
            
            with col2:
                st.write("")  # Spacing
                st.write("")
                submitted = st.form_submit_button("Create Group", type="primary", disabled=False)
            
            if submitted and new_group_name:
                success, message = create_sample_group(new_group_name, new_group_desc)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            elif submitted and not new_group_name:
                st.error("Please enter a group name")
    
    # Display existing sample groups
    if not sample_groups_df.empty:
        st.subheader("📋 Existing Sample Groups")
        
        # Show groups in a table
        display_df = sample_groups_df.copy()
        display_df['Actions'] = display_df['group_id'].apply(lambda x: x)  # Placeholder for actions
        
        # Configure columns
        col_config = {
            'group_name': st.column_config.TextColumn('Group Name', width="medium"),
            'description': st.column_config.TextColumn('Description', width="large"),
            'member_count': st.column_config.NumberColumn('Cases', width="small"),
            'created_date': st.column_config.DatetimeColumn('Created', width="medium"),
            'Actions': st.column_config.Column('Actions', width="medium")
        }
        
        # Display each group with actions
        for idx, group in sample_groups_df.iterrows():
            # Use session state to maintain expander state
            expander_key = f"group_expanded_{group['group_id']}"
            if expander_key not in st.session_state:
                st.session_state[expander_key] = False
            
            with st.expander(f"**{group['group_name']}** ({group['member_count']} cases)", 
                           expanded=st.session_state[expander_key]):
                # Group info
                if group['description']:
                    st.write(f"📝 **Description:** {group['description']}")
                st.write(f"📅 **Created:** {group['created_date']}")
                
                # Get group statistics
                if group['member_count'] > 0:
                    stats = get_sample_group_statistics(group['group_id'])
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Cases", stats['total_cases'])
                    with col2:
                        if stats['year_range'][0]:
                            st.metric("Year Range", f"{stats['year_range'][0]}-{stats['year_range'][1]}")
                    with col3:
                        st.metric("Avg Length", f"{stats['avg_case_length']:,.0f} chars")
                    with col4:
                        st.metric("Areas of Law", stats['distinct_areas'])
                    
                    # Show all cases in the group
                    if group['member_count'] > 0:
                        members_df = get_sample_group_members(group['group_id'])
                        
                        if not members_df.empty:
                            # Show all cases in a scrollable table
                            display_df = members_df[['case_name', 'citation', 'decision_year', 'area_of_law']]
                            st.dataframe(
                                display_df, 
                                use_container_width=True, 
                                hide_index=True,
                                height=300,  # Fixed height to make it scrollable
                                column_config={
                                    'case_name': st.column_config.TextColumn('Case Name', width="large"),
                                    'citation': st.column_config.TextColumn('Citation', width="medium"),
                                    'decision_year': st.column_config.NumberColumn('Year', width="small"),
                                    'area_of_law': st.column_config.TextColumn('Area of Law', width="medium")
                                }
                            )
                
                # Actions
                st.write("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("➕ Add Cases", key=f"add_{group['group_id']}"):
                        st.session_state[f"add_cases_{group['group_id']}"] = True
                        st.session_state[expander_key] = True  # Keep expander open
                        st.rerun()
                
                with col2:
                    if st.button("📋 Duplicate", key=f"dup_{group['group_id']}"):
                        st.session_state[f"duplicate_{group['group_id']}"] = True
                        st.session_state[expander_key] = True  # Keep expander open
                        st.rerun()
                
                with col3:
                    if st.button("🗑️ Delete", key=f"del_{group['group_id']}", 
                                 disabled=stats['experiments_count'] > 0 if group['member_count'] > 0 else False):
                        st.session_state[f"confirm_delete_{group['group_id']}"] = True
                        st.session_state[expander_key] = True  # Keep expander open
                        st.rerun()
                
                # Add cases interface
                if st.session_state.get(f"add_cases_{group['group_id']}", False):
                    st.write("---")
                    st.subheader("Add Cases to Group")
                    
                    # Get available cases for filter options
                    available_df = get_available_cases_for_group(group['group_id'], limit=None)
                    
                    if not available_df.empty:
                        # Initialize session state for search and selections
                        search_key = f"area_search_{group['group_id']}"
                        selection_key = f"area_selection_{group['group_id']}"
                        
                        if search_key not in st.session_state:
                            st.session_state[search_key] = ""
                        if selection_key not in st.session_state:
                            st.session_state[selection_key] = []
                        
                        st.write("**Filters:**")
                        
                        # Year range filter (outside form)
                        min_year = int(available_df['decision_year'].min())
                        max_year = int(available_df['decision_year'].max())
                        year_filter = st.slider("Year Range", min_year, max_year, (min_year, max_year))
                        
                        # Area of law filter with persistent search (outside form)
                        unique_areas = sorted(available_df['area_of_law'].dropna().unique())
                        
                        st.write("**Areas of Law:**")
                        
                        # Single multiselect with enhanced search functionality
                        selected_areas = st.multiselect(
                            "Select areas of law",
                            options=unique_areas,
                            default=st.session_state[selection_key],
                            placeholder="Search and select areas (e.g., tort, contract, criminal)",
                            help=f"Search from {len(unique_areas)} available areas of law. Type to filter options and your selections will persist."
                        )
                        
                        # Update session state with current selections
                        st.session_state[selection_key] = selected_areas
                        
                        # Show selection summary
                        if selected_areas:
                            if len(selected_areas) <= 3:
                                st.success(f"✅ **Selected:** {', '.join(selected_areas)}")
                            else:
                                st.success(f"✅ **Selected {len(selected_areas)} areas:** {', '.join(selected_areas[:2])} and {len(selected_areas)-2} more...")
                        else:
                            st.info("💡 **No areas selected** - will include all areas in results")
                        
                        # Quick action buttons
                        col_select_all, col_clear_all = st.columns(2)
                        
                        with col_select_all:
                            if st.button(f"✅ Select All {len(unique_areas)} Areas", 
                                       key=f"select_all_areas_{group['group_id']}", 
                                       type="secondary",
                                       use_container_width=True):
                                st.session_state[selection_key] = unique_areas.copy()
                                st.rerun()
                        
                        with col_clear_all:
                            if st.button("🗑️ Clear All Selections", 
                                       key=f"clear_all_selected_{group['group_id']}", 
                                       type="secondary",
                                       use_container_width=True):
                                st.session_state[selection_key] = []
                                st.rerun()
                        
                        # Apply filters button with better styling
                        st.write("---")
                        apply_filters = st.button("🔍 Apply Filters & Show Results", 
                                                 key=f"apply_filters_{group['group_id']}", 
                                                 type="primary",
                                                 use_container_width=True)
                        
                        # Process filters when form is submitted
                        if apply_filters or 'last_filters' not in st.session_state:
                            # Store current filters
                            st.session_state['last_filters'] = {
                                'year_filter': year_filter,
                                'selected_areas': selected_areas
                            }
                        
                        # Use stored filters for consistent results
                        current_filters = st.session_state.get('last_filters', {
                            'year_filter': (min_year, max_year),
                            'selected_areas': []
                        })
                        
                        used_year_filter = current_filters['year_filter']
                        used_area_filter = current_filters['selected_areas']
                        
                        # Get filtered results and show count
                        filtered_available = get_available_cases_for_group(
                            group['group_id'],
                            year_range=used_year_filter if used_year_filter != (min_year, max_year) else None,
                            areas=used_area_filter if used_area_filter else None
                        )
                        available_count = len(filtered_available)
                        
                        # Show current filter status
                        if used_year_filter != (min_year, max_year) or used_area_filter:
                            filter_desc = []
                            if used_year_filter != (min_year, max_year):
                                filter_desc.append(f"Years: {used_year_filter[0]}-{used_year_filter[1]}")
                            if used_area_filter:
                                if len(used_area_filter) <= 3:
                                    filter_desc.append(f"Areas: {', '.join(used_area_filter)}")
                                else:
                                    filter_desc.append(f"Areas: {len(used_area_filter)} selected")
                            
                            st.info(f"📊 **{available_count} cases** match filters: {' | '.join(filter_desc)}")
                        else:
                            st.info(f"📊 **{available_count} total cases** available to add")
                        
                        if available_count > 0:
                            # Preview section
                            st.write("**Preview of filtered cases:**")
                            preview_df = get_available_cases_for_group(
                                group['group_id'],
                                year_range=used_year_filter if used_year_filter != (min_year, max_year) else None,
                                areas=used_area_filter if used_area_filter else None,
                                limit=5
                            )
                            st.dataframe(
                                preview_df[['case_name', 'citation', 'decision_year', 'area_of_law']],
                                use_container_width=True,
                                hide_index=True
                            )
                            st.caption(f"Showing first 5 of {available_count} available cases")
                            
                            # Action options
                            st.write("**Add Cases Options:**")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Option 1: Add All Filtered Cases**")
                                if st.button(f"Add All {available_count} Cases", 
                                           key=f"add_all_{group['group_id']}", 
                                           type="primary"):
                                    with st.spinner(f"Adding {available_count} cases..."):
                                        success_count, duplicate_count = add_cases_to_sample_group(
                                            group['group_id'], 
                                            filtered_available['case_id'].tolist()
                                        )
                                        if duplicate_count > 0:
                                            st.success(f"Added {success_count} new cases to the group ({duplicate_count} were already in the group)")
                                        else:
                                            st.success(f"Added {success_count} cases to the group")
                                        # Clear filters and search state after successful addition
                                        if 'last_filters' in st.session_state:
                                            del st.session_state['last_filters']
                                        # Clear area search and selection state
                                        search_key = f"area_search_{group['group_id']}"
                                        selection_key = f"area_selection_{group['group_id']}"
                                        if search_key in st.session_state:
                                            del st.session_state[search_key]
                                        if selection_key in st.session_state:
                                            del st.session_state[selection_key]
                                        st.rerun()
                            
                            with col2:
                                st.write("**Option 2: Add Random Selection**")
                                
                                max_random = min(available_count, 500)
                                default_random = min(15, available_count)
                                
                                num_to_add = st.number_input(
                                    "Number of cases to randomly select",
                                    min_value=1,
                                    max_value=max_random,
                                    value=default_random,
                                    key=f"num_{group['group_id']}",
                                    help=f"Select up to {max_random} cases randomly from the {available_count} filtered cases"
                                )
                                
                                if st.button(f"Add {num_to_add} Random Cases", 
                                           key=f"add_random_{group['group_id']}", 
                                           type="secondary"):
                                    with st.spinner(f"Adding {num_to_add} random cases..."):
                                        cases_to_add = get_available_cases_for_group(
                                            group['group_id'],
                                            year_range=used_year_filter if used_year_filter != (min_year, max_year) else None,
                                            areas=used_area_filter if used_area_filter else None,
                                            limit=num_to_add
                                        )
                                        
                                        if not cases_to_add.empty:
                                            success_count, duplicate_count = add_cases_to_sample_group(
                                                group['group_id'], 
                                                cases_to_add['case_id'].tolist()
                                            )
                                            if duplicate_count > 0:
                                                st.success(f"Added {success_count} new cases to the group ({duplicate_count} were already in the group)")
                                            else:
                                                st.success(f"Added {success_count} cases to the group")
                                            # Clear filters and search state after successful addition
                                            if 'last_filters' in st.session_state:
                                                del st.session_state['last_filters']
                                            # Clear area search and selection state
                                            search_key = f"area_search_{group['group_id']}"
                                            selection_key = f"area_selection_{group['group_id']}"
                                            if search_key in st.session_state:
                                                del st.session_state[search_key]
                                            if selection_key in st.session_state:
                                                del st.session_state[selection_key]
                                            st.rerun()
                        else:
                            st.warning("No cases available with the current filters")
                    else:
                        st.warning("No cases available to add to this group")
                    
                    # Clear filters and Cancel buttons
                    st.write("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Clear Filters", key=f"clear_filters_{group['group_id']}", type="secondary"):
                            if 'last_filters' in st.session_state:
                                del st.session_state['last_filters']
                            # Clear area search and selection state
                            search_key = f"area_search_{group['group_id']}"
                            selection_key = f"area_selection_{group['group_id']}"
                            if search_key in st.session_state:
                                del st.session_state[search_key]
                            if selection_key in st.session_state:
                                del st.session_state[selection_key]
                            st.session_state[expander_key] = True  # Keep expander open
                            st.rerun()
                    
                    with col2:
                        if st.button("Cancel", key=f"cancel_add_{group['group_id']}", type="secondary"):
                            st.session_state[f"add_cases_{group['group_id']}"] = False
                            if 'last_filters' in st.session_state:
                                del st.session_state['last_filters']
                            # Clear area search and selection state
                            search_key = f"area_search_{group['group_id']}"
                            selection_key = f"area_selection_{group['group_id']}"
                            if search_key in st.session_state:
                                del st.session_state[search_key]
                            if selection_key in st.session_state:
                                del st.session_state[selection_key]
                            st.session_state[expander_key] = True  # Keep expander open
                            st.rerun()
                
                # Duplicate group interface
                if st.session_state.get(f"duplicate_{group['group_id']}", False):
                    st.write("---")
                    st.subheader("Duplicate Sample Group")
                    
                    dup_name = st.text_input("New Group Name", 
                                           value=f"{group['group_name']} (Copy)",
                                           key=f"dup_name_{group['group_id']}")
                    dup_desc = st.text_area("New Description", 
                                          value=group['description'] or "",
                                          key=f"dup_desc_{group['group_id']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Duplicate", key=f"confirm_dup_{group['group_id']}", type="primary"):
                            success, message = duplicate_sample_group(group['group_id'], dup_name, dup_desc)
                            if success:
                                st.success(message)
                                st.session_state[f"duplicate_{group['group_id']}"] = False
                                st.rerun()
                            else:
                                st.error(message)
                    
                    with col2:
                        if st.button("Cancel", key=f"cancel_dup_{group['group_id']}"):
                            st.session_state[f"duplicate_{group['group_id']}"] = False
                            st.session_state[expander_key] = True  # Keep expander open
                            st.rerun()
                
                # Delete confirmation
                if st.session_state.get(f"confirm_delete_{group['group_id']}", False):
                    st.write("---")
                    st.warning(f"⚠️ Are you sure you want to delete '{group['group_name']}'?")
                    st.write(f"This will remove the group and all {group['member_count']} case associations.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Yes, Delete", key=f"do_delete_{group['group_id']}", type="secondary"):
                            success, message = delete_sample_group(group['group_id'])
                            if success:
                                st.success(message)
                                # Don't need to maintain expander state since group is deleted
                                st.rerun()
                            else:
                                st.error(message)
                    
                    with col2:
                        if st.button("Cancel", key=f"cancel_delete_{group['group_id']}"):
                            st.session_state[f"confirm_delete_{group['group_id']}"] = False
                            st.session_state[expander_key] = True  # Keep expander open
                            st.rerun()
    else:
        st.info("No sample groups created yet. Create your first sample group above!")
    
    # Migration helper (temporary)
    if selected_cases > 0 and sample_groups_df.empty:
        st.write("---")
        st.info("💡 **Migration Note:** You have cases selected in the old system. Create a sample group above to start using the new system.")