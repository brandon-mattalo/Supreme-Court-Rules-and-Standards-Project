"""
Case Management Utilities
Shared functionality for loading and managing legal cases across experiments
"""

import streamlit as st
import pandas as pd
from config import execute_sql

def get_database_counts():
    """Get counts for cases, tests, comparisons, and validated tests"""
    cases_count = execute_sql("SELECT COUNT(*) FROM cases", fetch=True)[0][0]
    tests_count = execute_sql("SELECT COUNT(*) FROM legal_tests", fetch=True)[0][0]
    comparisons_count = execute_sql("SELECT COUNT(*) FROM legal_test_comparisons", fetch=True)[0][0]
    validated_count = execute_sql("SELECT COUNT(*) FROM legal_tests WHERE validation_status = 'accurate'", fetch=True)[0][0]
    
    return cases_count, tests_count, comparisons_count, validated_count

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
    st.info("Sample URLs from loaded data:")
    sample_urls = mapped_df['scc_url'].dropna().head(3).tolist()
    for url in sample_urls:
        st.write(f"• {url}")

    # Remove citation_normalized before inserting to avoid column mismatch
    mapped_df = mapped_df.drop('citation_normalized', axis=1)
    
    # Insert data into database
    try:
        # Convert DataFrame to list of tuples for insertion
        data_to_insert = [tuple(row) for row in mapped_df.values]
        columns = ', '.join(mapped_df.columns)
        placeholders = ', '.join(['?' for _ in mapped_df.columns])
        
        # Use batch insert for efficiency
        insert_query = f"INSERT OR IGNORE INTO cases ({columns}) VALUES ({placeholders})"
        
        conn = execute_sql("SELECT name FROM sqlite_master WHERE type='table' AND name='cases'", fetch=True)
        if not conn:
            st.error("Cases table does not exist. Please ensure database is properly initialized.")
            return
            
        # Insert data in smaller batches to avoid memory issues
        batch_size = 100
        total_inserted = 0
        
        for i in range(0, len(data_to_insert), batch_size):
            batch = data_to_insert[i:i + batch_size]
            for row in batch:
                try:
                    execute_sql(insert_query, row)
                    total_inserted += 1
                except Exception as e:
                    st.warning(f"Could not insert case {row[0] if row else 'unknown'}: {e}")
        
        st.success(f"Successfully loaded {total_inserted} unique cases into the database!")
        
        # Show summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Cases in File", len(mapped_df))
        with col2:
            st.metric("Cases Inserted", total_inserted)
        with col3:
            duplicates = len(mapped_df) - total_inserted
            st.metric("Duplicates Skipped", duplicates)
            
    except Exception as e:
        st.error(f"Error inserting data into database: {e}")

def clear_database():
    """Clear all data from the database"""
    try:
        # Clear tables in dependency order due to foreign key constraints
        execute_sql("DELETE FROM legal_test_comparisons")
        execute_sql("DELETE FROM legal_tests")
        execute_sql("DELETE FROM cases")
        
        st.success("All data cleared from database!")
        return True
    except Exception as e:
        st.error(f"Error clearing database: {e}")
        return False

def get_case_summary():
    """Get summary statistics about loaded cases"""
    try:
        # Basic counts
        cases_count, tests_count, comparisons_count, validated_count = get_database_counts()
        
        # Additional case statistics
        year_stats = execute_sql("""
            SELECT 
                MIN(decision_year) as earliest_year,
                MAX(decision_year) as latest_year,
                COUNT(DISTINCT decision_year) as year_span
            FROM cases
        """, fetch=True)
        
        area_stats = execute_sql("""
            SELECT 
                area_of_law,
                COUNT(*) as case_count
            FROM cases
            WHERE area_of_law IS NOT NULL
            GROUP BY area_of_law
            ORDER BY case_count DESC
            LIMIT 5
        """, fetch=True)
        
        return {
            'total_cases': cases_count,
            'total_tests': tests_count,
            'total_comparisons': comparisons_count,
            'validated_tests': validated_count,
            'year_range': year_stats[0] if year_stats else (None, None, 0),
            'top_areas': area_stats if area_stats else []
        }
    except Exception as e:
        st.error(f"Error getting case summary: {e}")
        return None

def get_available_cases():
    """Get list of available cases for analysis"""
    try:
        cases = execute_sql("""
            SELECT case_id, case_name, citation, decision_year, area_of_law
            FROM cases
            ORDER BY decision_year DESC, case_name ASC
        """, fetch=True)
        
        if cases:
            columns = ['case_id', 'case_name', 'citation', 'decision_year', 'area_of_law']
            return pd.DataFrame(cases, columns=columns)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving cases: {e}")
        return pd.DataFrame()

def filter_cases_by_criteria(year_range=None, areas=None, sample_size=None):
    """Filter cases based on criteria and return a sample"""
    try:
        query = "SELECT case_id, case_name, citation, decision_year, area_of_law FROM cases WHERE 1=1"
        params = []
        
        if year_range:
            query += " AND decision_year BETWEEN ? AND ?"
            params.extend(year_range)
        
        if areas:
            placeholders = ','.join(['?' for _ in areas])
            query += f" AND area_of_law IN ({placeholders})"
            params.extend(areas)
        
        query += " ORDER BY decision_year DESC, case_name ASC"
        
        if sample_size:
            query += " LIMIT ?"
            params.append(sample_size)
        
        cases = execute_sql(query, params, fetch=True)
        
        if cases:
            columns = ['case_id', 'case_name', 'citation', 'decision_year', 'area_of_law']
            return pd.DataFrame(cases, columns=columns)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error filtering cases: {e}")
        return pd.DataFrame()