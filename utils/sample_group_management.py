"""
Sample Group Management Utilities
Handles creation, modification, and management of sample groups for experiments
"""

import streamlit as st
from config import execute_sql
from typing import List, Tuple, Optional, Dict
import pandas as pd

def create_sample_group(group_name: str, description: str = None) -> Tuple[bool, str]:
    """
    Create a new sample group.
    
    Args:
        group_name: Name of the sample group
        description: Optional description
        
    Returns:
        Tuple of (success, message)
    """
    try:
        execute_sql("""
            INSERT INTO v2_sample_groups (group_name, description)
            VALUES (?, ?)
        """, params=(group_name, description))
        return True, f"Sample group '{group_name}' created successfully"
    except Exception as e:
        if "unique" in str(e).lower():
            return False, f"Sample group '{group_name}' already exists"
        return False, f"Error creating sample group: {str(e)}"

def get_sample_groups() -> pd.DataFrame:
    """
    Get all sample groups with their member counts.
    
    Returns:
        DataFrame with sample group information
    """
    results = execute_sql("""
        SELECT 
            sg.group_id,
            sg.group_name,
            sg.description,
            COUNT(sgm.case_id) as member_count,
            sg.created_date
        FROM v2_sample_groups sg
        LEFT JOIN v2_sample_group_members sgm ON sg.group_id = sgm.group_id
        GROUP BY sg.group_id, sg.group_name, sg.description, sg.created_date
        ORDER BY sg.created_date DESC
    """, fetch=True)
    
    if results:
        return pd.DataFrame(results, columns=['group_id', 'group_name', 'description', 'member_count', 'created_date'])
    return pd.DataFrame()

def get_sample_group_members(group_id: int) -> pd.DataFrame:
    """
    Get all cases in a sample group.
    
    Args:
        group_id: ID of the sample group
        
    Returns:
        DataFrame with case information
    """
    results = execute_sql("""
        SELECT 
            c.case_id,
            c.case_name,
            c.citation,
            c.decision_year,
            c.area_of_law,
            c.case_length,
            sgm.added_date
        FROM v2_sample_group_members sgm
        JOIN v2_cases c ON sgm.case_id = c.case_id
        WHERE sgm.group_id = ?
        ORDER BY c.decision_year DESC, c.case_name
    """, params=(group_id,), fetch=True)
    
    if results:
        return pd.DataFrame(results, columns=['case_id', 'case_name', 'citation', 
                                             'decision_year', 'area_of_law', 'case_length', 'added_date'])
    return pd.DataFrame()

def add_cases_to_sample_group(group_id: int, case_ids: List[int]) -> Tuple[int, int]:
    """
    Add cases to a sample group.
    
    Args:
        group_id: ID of the sample group
        case_ids: List of case IDs to add
        
    Returns:
        Tuple of (success_count, duplicate_count)
    """
    success_count = 0
    duplicate_count = 0
    
    for case_id in case_ids:
        try:
            execute_sql("""
                INSERT INTO v2_sample_group_members (group_id, case_id)
                VALUES (?, ?)
            """, params=(group_id, case_id))
            success_count += 1
        except Exception as e:
            if "unique" in str(e).lower() or "duplicate" in str(e).lower():
                duplicate_count += 1
            else:
                # Log error but continue
                print(f"Error adding case {case_id} to group {group_id}: {str(e)}")
    
    return success_count, duplicate_count

def remove_cases_from_sample_group(group_id: int, case_ids: List[int]) -> int:
    """
    Remove cases from a sample group.
    
    Args:
        group_id: ID of the sample group
        case_ids: List of case IDs to remove
        
    Returns:
        Number of cases removed
    """
    try:
        placeholders = ','.join(['?'] * len(case_ids))
        result = execute_sql(f"""
            DELETE FROM v2_sample_group_members 
            WHERE group_id = ? AND case_id IN ({placeholders})
        """, params=(group_id, *case_ids))
        
        # Get affected rows count
        if hasattr(result, 'rowcount'):
            return result.rowcount
        return len(case_ids)  # Assume all were deleted
    except Exception as e:
        print(f"Error removing cases from group {group_id}: {str(e)}")
        return 0

def delete_sample_group(group_id: int) -> Tuple[bool, str]:
    """
    Delete a sample group (cascade deletes members).
    
    Args:
        group_id: ID of the sample group to delete
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Check if any experiments use this group
        exp_count = execute_sql("""
            SELECT COUNT(*) FROM v2_experiments 
            WHERE sample_group_id = ?
        """, params=(group_id,), fetch=True)[0][0]
        
        if exp_count > 0:
            return False, f"Cannot delete: {exp_count} experiment(s) use this sample group"
        
        execute_sql("DELETE FROM v2_sample_groups WHERE group_id = ?", params=(group_id,))
        return True, "Sample group deleted successfully"
    except Exception as e:
        return False, f"Error deleting sample group: {str(e)}"

def get_available_cases_for_group(group_id: int, year_range: Optional[Tuple[int, int]] = None, 
                                 areas: Optional[List[str]] = None, limit: int = None) -> pd.DataFrame:
    """
    Get cases available to add to a sample group (not already in the group).
    
    Args:
        group_id: ID of the sample group
        year_range: Optional tuple of (min_year, max_year)
        areas: Optional list of areas of law
        limit: Optional limit on number of results
        
    Returns:
        DataFrame with available cases
    """
    query = """
        SELECT 
            c.case_id,
            c.case_name,
            c.citation,
            c.decision_year,
            c.area_of_law,
            c.case_length
        FROM v2_cases c
        WHERE c.case_id NOT IN (
            SELECT case_id FROM v2_sample_group_members WHERE group_id = ?
        )
    """
    params = [group_id]
    
    if year_range:
        query += " AND c.decision_year BETWEEN ? AND ?"
        params.extend(year_range)
    
    if areas:
        placeholders = ','.join(['?'] * len(areas))
        query += f" AND c.area_of_law IN ({placeholders})"
        params.extend(areas)
    
    query += " ORDER BY c.decision_year DESC, c.case_name"
    
    if limit:
        query += f" LIMIT {limit}"
    
    results = execute_sql(query, params=params, fetch=True)
    
    if results:
        return pd.DataFrame(results, columns=['case_id', 'case_name', 'citation', 
                                             'decision_year', 'area_of_law', 'case_length'])
    return pd.DataFrame()

def duplicate_sample_group(source_group_id: int, new_name: str, new_description: str = None) -> Tuple[bool, str]:
    """
    Duplicate a sample group with all its members.
    
    Args:
        source_group_id: ID of the group to duplicate
        new_name: Name for the new group
        new_description: Optional description for the new group
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Create new group
        success, message = create_sample_group(new_name, new_description)
        if not success:
            return False, message
        
        # Get new group ID
        new_group_id = execute_sql("""
            SELECT group_id FROM v2_sample_groups WHERE group_name = ?
        """, params=(new_name,), fetch=True)[0][0]
        
        # Copy members
        execute_sql("""
            INSERT INTO v2_sample_group_members (group_id, case_id)
            SELECT ?, case_id FROM v2_sample_group_members WHERE group_id = ?
        """, params=(new_group_id, source_group_id))
        
        # Get member count
        member_count = execute_sql("""
            SELECT COUNT(*) FROM v2_sample_group_members WHERE group_id = ?
        """, params=(new_group_id,), fetch=True)[0][0]
        
        return True, f"Sample group duplicated successfully with {member_count} cases"
    except Exception as e:
        return False, f"Error duplicating sample group: {str(e)}"

def get_sample_group_statistics(group_id: int) -> Dict:
    """
    Get detailed statistics for a sample group.
    
    Args:
        group_id: ID of the sample group
        
    Returns:
        Dictionary with statistics
    """
    stats = {}
    
    # Basic counts
    basic_stats = execute_sql("""
        SELECT 
            COUNT(DISTINCT c.case_id) as total_cases,
            MIN(c.decision_year) as min_year,
            MAX(c.decision_year) as max_year,
            AVG(c.case_length) as avg_length,
            COUNT(DISTINCT c.area_of_law) as distinct_areas
        FROM v2_sample_group_members sgm
        JOIN v2_cases c ON sgm.case_id = c.case_id
        WHERE sgm.group_id = ?
    """, params=(group_id,), fetch=True)[0]
    
    stats['total_cases'] = basic_stats[0] or 0
    stats['year_range'] = (basic_stats[1], basic_stats[2]) if basic_stats[1] else (None, None)
    stats['avg_case_length'] = float(basic_stats[3]) if basic_stats[3] else 0
    stats['distinct_areas'] = basic_stats[4] or 0
    
    # Area of law distribution
    area_dist = execute_sql("""
        SELECT c.area_of_law, COUNT(*) as count
        FROM v2_sample_group_members sgm
        JOIN v2_cases c ON sgm.case_id = c.case_id
        WHERE sgm.group_id = ? AND c.area_of_law IS NOT NULL
        GROUP BY c.area_of_law
        ORDER BY count DESC
    """, params=(group_id,), fetch=True)
    
    stats['area_distribution'] = {area: count for area, count in area_dist} if area_dist else {}
    
    # Experiments using this group
    exp_count = execute_sql("""
        SELECT COUNT(*) FROM v2_experiments WHERE sample_group_id = ?
    """, params=(group_id,), fetch=True)[0][0]
    
    stats['experiments_count'] = exp_count
    
    return stats