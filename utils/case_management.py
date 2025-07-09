"""
Case Management Utilities
Shared functionality for loading and managing legal cases across experiments
"""

import streamlit as st
import pandas as pd
from config import execute_sql

def calculate_bradley_terry_comparisons(n_cases=None):
    """
    Calculate required number of comparisons using Bradley-Terry linked block design.
    Now uses actual block structure when available.
    
    Parameters:
    - n_cases: Total number of cases (optional, will use actual structure if available)
    
    Returns:
    - required_comparisons: Number of comparisons needed for Bradley-Terry analysis
    """
    # Try to use actual block structure first, but only if n_cases matches the structure
    try:
        structure = get_bradley_terry_structure()
        if structure and n_cases is not None:
            # Check if the structure matches the requested n_cases
            unique_cases = set(case['case_id'] for case in structure)
            if len(unique_cases) == n_cases:
                # Calculate exact comparisons from actual block structure
                blocks = {}
                for case in structure:
                    block_num = case['block_number']
                    if block_num not in blocks:
                        blocks[block_num] = []
                    blocks[block_num].append(case['case_id'])
                
                total_comparisons = 0
                for block_cases in blocks.values():
                    block_size = len(block_cases)
                    block_comparisons = (block_size * (block_size - 1)) // 2
                    total_comparisons += block_comparisons
                
                return total_comparisons
    except:
        # Fall back to mathematical calculation if structure unavailable
        pass
    
    # Fallback: Mathematical calculation based on n_cases
    if n_cases is None or n_cases <= 0:
        return 0
    
    # Bradley-Terry parameters
    block_size = 15  # 12 core + 3 bridge cases per block
    core_cases_per_block = 12
    comparisons_per_block = (block_size * (block_size - 1)) // 2  # 105 comparisons per block
    
    if n_cases <= block_size:
        # If sample fits in one block, use standard pairwise comparisons
        return (n_cases * (n_cases - 1)) // 2
    else:
        # Calculate blocks needed for linked block design
        num_blocks = (n_cases + core_cases_per_block - 1) // core_cases_per_block  # Ceiling division
        return num_blocks * comparisons_per_block

# Cache database counts for 30 seconds to avoid repeated queries
@st.cache_data(ttl=30)
def _cached_database_counts():
    """Cached version of database counts query"""
    try:
        cases_count = execute_sql("SELECT COUNT(*) FROM v2_cases", fetch=True)[0][0]
    except:
        cases_count = 0
    
    try:
        selected_cases_count = execute_sql("SELECT COUNT(*) FROM v2_experiment_selected_cases", fetch=True)[0][0]
    except:
        selected_cases_count = 0
    
    try:
        tests_count = execute_sql("SELECT COUNT(*) FROM v2_experiment_extractions", fetch=True)[0][0]
    except:
        tests_count = 0
    
    try:
        comparisons_count = execute_sql("SELECT COUNT(*) FROM v2_experiment_comparisons", fetch=True)[0][0]
    except:
        comparisons_count = 0
    
    try:
        validated_count = execute_sql("SELECT COUNT(*) FROM v2_experiment_extractions WHERE validation_status = 'accurate'", fetch=True)[0][0]
    except:
        validated_count = 0
    
    return cases_count, selected_cases_count, tests_count, comparisons_count, validated_count

def get_database_counts():
    """Get counts for cases, extractions, comparisons, and validated extractions (v2)"""
    return _cached_database_counts()

def load_data_from_parquet(uploaded_file, start_batch=1):
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
        'unofficial_text': 'case_text' # Store full text in case_text column for v2
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
    
    mapped_df['decision_url'] = mapped_df.apply(lambda row: get_mapping_value(
        row['citation_normalized'], citation_to_url, 
        scc_df.loc[row.name, 'citation_normalized_2'] if 'citation_normalized_2' in scc_df.columns else ''), axis=1)
    
    # Validate URLs and show warning for problematic ones
    invalid_urls = mapped_df[mapped_df['decision_url'].str.contains('localhost|127.0.0.1', na=False, case=False)]
    if not invalid_urls.empty:
        st.warning(f"Found {len(invalid_urls)} cases with localhost URLs. These links may not work properly.")
    
    # Show some URL examples for debugging
    st.info("Sample URLs from loaded data:")
    sample_urls = mapped_df['decision_url'].dropna().head(3).tolist()
    for url in sample_urls:
        st.write(f"• {url}")

    # Add case_length and subject columns for v2 schema
    mapped_df['case_length'] = mapped_df['case_text'].str.len()
    mapped_df['subject'] = mapped_df['area_of_law']  # Copy area_of_law to subject for v2
    
    # Remove columns that don't exist in v2_cases table
    columns_to_remove = ['citation_normalized']
    if 'citation2' in mapped_df.columns:
        columns_to_remove.append('citation2')
    
    mapped_df = mapped_df.drop(columns=columns_to_remove, errors='ignore')
    
    # Reorder columns to match v2_cases table schema (excluding case_id and created_date which are auto-generated)
    expected_columns = ['case_name', 'citation', 'decision_year', 'area_of_law', 'subject', 'decision_url', 'case_text', 'case_length']
    mapped_df = mapped_df[expected_columns]
    
    # Insert data into database
    try:
        # Convert DataFrame to list of tuples for insertion
        data_to_insert = [tuple(row) for row in mapped_df.values]
        columns = ', '.join(mapped_df.columns)
        placeholders = ', '.join(['?' for _ in mapped_df.columns])  # Use ? for both - execute_sql will convert for PostgreSQL
        
        # Use batch insert for efficiency (v2) - PostgreSQL compatible
        from config import DB_TYPE
        if DB_TYPE == 'postgresql':
            insert_query = f"INSERT INTO v2_cases ({columns}) VALUES ({placeholders}) ON CONFLICT (citation) DO NOTHING"
            # Check if table exists in PostgreSQL
            table_check = execute_sql("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'v2_cases'
                )
            """, fetch=True)
            table_exists = table_check[0][0] if table_check else False
        else:
            insert_query = f"INSERT OR IGNORE INTO v2_cases ({columns}) VALUES ({placeholders})"
            # Check if table exists in SQLite
            table_check = execute_sql("SELECT name FROM sqlite_master WHERE type='table' AND name='v2_cases'", fetch=True)
            table_exists = bool(table_check)
        
        if not table_exists:
            st.error("V2 Cases table does not exist. Please ensure database is properly initialized.")
            return
            
        # Insert data in smaller batches to avoid memory issues
        batch_size = 100
        total_batches = (len(data_to_insert) - 1) // batch_size + 1
        start_index = (start_batch - 1) * batch_size
        
        # Skip to the specified batch
        if start_batch > 1:
            st.info(f"⏭️ Skipping to batch {start_batch} (starting at case {start_index + 1})")
            if start_index >= len(data_to_insert):
                st.error(f"Start batch {start_batch} is beyond available data. Max batch: {total_batches}")
                return
        
        total_inserted = 0
        total_errors = 0
        total_skipped = start_index
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        error_container = st.container()
        
        with error_container:
            st.subheader("🔄 Loading Progress")
            if start_batch > 1:
                st.info(f"📍 Starting from batch {start_batch}/{total_batches}")
        
        for i in range(start_index, len(data_to_insert), batch_size):
            batch = data_to_insert[i:i + batch_size]
            
            # Update progress
            progress = (i + len(batch)) / len(data_to_insert)
            progress_bar.progress(progress)
            current_batch = i // batch_size + 1
            status_text.text(f"Processing batch {current_batch}/{total_batches}: Cases {i+1}-{min(i+len(batch), len(data_to_insert))}")
            
            for j, row in enumerate(batch):
                try:
                    execute_sql(insert_query, row)
                    total_inserted += 1
                    
                    # Show every 50th successful insert
                    if total_inserted % 50 == 0:
                        with error_container:
                            st.success(f"✅ Successfully inserted {total_inserted} cases so far...")
                            
                except Exception as e:
                    total_errors += 1
                    case_name = row[0] if row else 'unknown'
                    case_year = row[2] if len(row) > 2 else 'unknown'
                    
                    with error_container:
                        st.error(f"❌ Error inserting case '{case_name}' ({case_year}): {str(e)[:200]}...")
                    
                    # Show detailed error for first few failures
                    if total_errors <= 5:
                        with error_container:
                            st.code(f"Full error: {e}")
        
        # Final progress update
        progress_bar.progress(1.0)
        status_text.text("✅ Loading complete!")
        
        with error_container:
            if total_errors > 0:
                st.warning(f"⚠️ Encountered {total_errors} errors during loading")
            if total_skipped > 0:
                st.info(f"📊 Final stats: {total_skipped} skipped, {total_inserted} inserted, {total_errors} errors")
            else:
                st.info(f"📊 Final stats: {total_inserted} inserted, {total_errors} errors")
        
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
        execute_sql("DELETE FROM v2_experiment_comparisons")
        execute_sql("DELETE FROM v2_experiment_extractions")
        execute_sql("DELETE FROM v2_cases")
        
        st.success("All data cleared from database!")
        return True
    except Exception as e:
        st.error(f"Error clearing database: {e}")
        return False

def clear_selected_cases():
    """Clear all selected cases for experiments (preserves main case database)"""
    try:
        # Clear tables that depend on selected cases
        execute_sql("DELETE FROM v2_bradley_terry_structure")
        execute_sql("DELETE FROM v2_experiment_selected_cases")
        
        # Clear caches after modifying data
        get_experiment_selected_cases.clear()
        _cached_database_counts.clear()
        
        st.success("All selected experiment cases cleared! Main case database preserved.")
        return True
    except Exception as e:
        st.error(f"Error clearing selected cases: {e}")
        return False

# Cache case summary for 60 seconds
@st.cache_data(ttl=60)
def get_case_summary():
    """Get summary statistics about loaded cases"""
    try:
        # Basic counts (updated for v2 format)
        cases_count, selected_cases_count, tests_count, comparisons_count, validated_count = get_database_counts()
        
        # Additional case statistics
        year_stats = execute_sql("""
            SELECT 
                MIN(decision_year) as earliest_year,
                MAX(decision_year) as latest_year,
                COUNT(DISTINCT decision_year) as year_span
            FROM v2_cases
        """, fetch=True)
        
        area_stats = execute_sql("""
            SELECT 
                area_of_law,
                COUNT(*) as case_count
            FROM v2_cases
            WHERE area_of_law IS NOT NULL
            GROUP BY area_of_law
            ORDER BY case_count DESC
            LIMIT 5
        """, fetch=True)
        
        return {
            'total_cases': cases_count,
            'selected_cases': selected_cases_count,
            'total_tests': tests_count,
            'total_comparisons': comparisons_count,
            'validated_tests': validated_count,
            'year_range': year_stats[0] if year_stats else (None, None, 0),
            'top_areas': area_stats if area_stats else []
        }
    except Exception as e:
        st.error(f"Error getting case summary: {e}")
        return None

# Cache available cases for 2 minutes
@st.cache_data(ttl=120)
def get_available_cases():
    """Get list of available cases for analysis"""
    try:
        cases = execute_sql("""
            SELECT case_id, case_name, citation, decision_year, area_of_law
            FROM v2_cases
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
        query = "SELECT case_id, case_name, citation, decision_year, area_of_law FROM v2_cases WHERE 1=1"
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

# Cache selected cases for 30 seconds (changes more frequently)
@st.cache_data(ttl=30)
def get_experiment_selected_cases():
    """Get all cases currently selected for experiments"""
    try:
        query = """
            SELECT c.case_id, c.case_name, c.citation, c.decision_year, c.area_of_law, 
                   s.selected_date, s.selected_by
            FROM v2_experiment_selected_cases s
            JOIN v2_cases c ON s.case_id = c.case_id
            ORDER BY s.selected_date ASC
        """
        cases = execute_sql(query, fetch=True)
        
        if cases:
            columns = ['case_id', 'case_name', 'citation', 'decision_year', 'area_of_law', 'selected_date', 'selected_by']
            return pd.DataFrame(cases, columns=columns)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving selected cases: {e}")
        return pd.DataFrame()

def get_available_cases_for_selection(year_range=None, areas=None, limit=None):
    """Get cases available for selection (not already selected for experiments)"""
    try:
        query = """
            SELECT c.case_id, c.case_name, c.citation, c.decision_year, c.area_of_law
            FROM v2_cases c
            LEFT JOIN v2_experiment_selected_cases s ON c.case_id = s.case_id
            WHERE s.case_id IS NULL
        """
        params = []
        
        if year_range:
            query += " AND c.decision_year BETWEEN ? AND ?"
            params.extend(year_range)
        
        if areas:
            placeholders = ','.join(['?' for _ in areas])
            query += f" AND c.area_of_law IN ({placeholders})"
            params.extend(areas)
        
        query += " ORDER BY RANDOM()" if areas or year_range else " ORDER BY RANDOM()"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cases = execute_sql(query, params, fetch=True)
        
        if cases:
            columns = ['case_id', 'case_name', 'citation', 'decision_year', 'area_of_law']
            return pd.DataFrame(cases, columns=columns)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving available cases: {e}")
        return pd.DataFrame()

def add_cases_to_experiments(case_ids, selected_by="researcher"):
    """Add selected cases to the experiment case pool"""
    try:
        success_count = 0
        duplicate_count = 0
        
        for case_id in case_ids:
            try:
                execute_sql("""
                    INSERT INTO v2_experiment_selected_cases (case_id, selected_by)
                    VALUES (?, ?)
                """, (case_id, selected_by))
                success_count += 1
            except Exception as e:
                if "unique" in str(e).lower() or "duplicate" in str(e).lower():
                    duplicate_count += 1
                else:
                    st.error(f"Error adding case {case_id}: {e}")
        
        # Clear caches after modifying data
        get_experiment_selected_cases.clear()
        _cached_database_counts.clear()
        
        return success_count, duplicate_count
    except Exception as e:
        st.error(f"Error in batch case addition: {e}")
        return 0, 0

def remove_cases_from_experiments(case_ids):
    """Remove cases from the experiment case pool"""
    try:
        success_count = 0
        
        for case_id in case_ids:
            try:
                execute_sql("DELETE FROM v2_experiment_selected_cases WHERE case_id = ?", (case_id,))
                success_count += 1
            except Exception as e:
                st.error(f"Error removing case {case_id}: {e}")
        
        # Clear caches after modifying data
        get_experiment_selected_cases.clear()
        _cached_database_counts.clear()
        
        return success_count
    except Exception as e:
        st.error(f"Error in batch case removal: {e}")
        return 0

def calculate_case_importance_scores():
    """
    Calculate importance scores for all selected cases based on multiple factors:
    - Citation frequency (how often this case is referenced)
    - Temporal significance (recent landmark cases vs older foundational cases)
    - Area of law breadth (cases covering important/broad legal areas)
    - Case length (longer cases often contain more comprehensive legal analysis)
    
    Returns: dict {case_id: importance_score}
    """
    try:
        # Get all selected cases with their metadata
        selected_cases = execute_sql("""
            SELECT c.case_id, c.case_name, c.citation, c.decision_year, 
                   c.area_of_law, c.subject, c.case_length, c.case_text
            FROM v2_cases c
            JOIN v2_experiment_selected_cases s ON c.case_id = s.case_id
            ORDER BY c.case_id
        """, fetch=True)
        
        if not selected_cases:
            return {}
        
        importance_scores = {}
        
        # Convert to easier processing format
        cases_data = []
        for row in selected_cases:
            case_id, name, citation, year, area, subject, length, text = row
            cases_data.append({
                'case_id': case_id,
                'name': name or '',
                'citation': citation or '',
                'year': year or 2020,
                'area': area or '',
                'subject': subject or '',
                'length': length or 0,
                'text': text or ''
            })
        
        # Calculate citation frequency scores
        citation_scores = _calculate_citation_scores(cases_data)
        
        # Calculate temporal significance scores  
        temporal_scores = _calculate_temporal_scores(cases_data)
        
        # Calculate area of law importance scores
        area_scores = _calculate_area_scores(cases_data)
        
        # Calculate case length scores (normalized)
        length_scores = _calculate_length_scores(cases_data)
        
        # Combine scores with weights
        weights = {
            'citation': 0.4,    # Citation frequency is most important
            'temporal': 0.2,    # Temporal significance
            'area': 0.25,       # Area importance  
            'length': 0.15      # Case comprehensiveness
        }
        
        for case in cases_data:
            case_id = case['case_id']
            
            # Weighted combination of all factors
            combined_score = (
                weights['citation'] * citation_scores.get(case_id, 0.0) +
                weights['temporal'] * temporal_scores.get(case_id, 0.0) +
                weights['area'] * area_scores.get(case_id, 0.0) +
                weights['length'] * length_scores.get(case_id, 0.0)
            )
            
            importance_scores[case_id] = round(combined_score, 4)
        
        return importance_scores
    
    except Exception as e:
        st.error(f"Error calculating case importance scores: {e}")
        return {}

def _calculate_citation_scores(cases_data):
    """Calculate citation frequency scores by counting case name/citation references"""
    citation_scores = {}
    
    # Create searchable terms for each case
    case_terms = {}
    for case in cases_data:
        # Extract key terms from case name and citation
        terms = []
        if case['name']:
            # Get main parties (before 'v.' or 'vs.' or 'c.')
            name_parts = case['name'].replace(' v. ', ' vs. ').replace(' c. ', ' vs. ').split(' vs. ')
            if len(name_parts) >= 2:
                terms.extend([part.strip() for part in name_parts[:2]])
        
        if case['citation']:
            # Add citation components
            terms.append(case['citation'].strip())
        
        case_terms[case['case_id']] = terms
    
    # Count references to each case in other cases' text
    for target_case in cases_data:
        target_id = target_case['case_id']
        reference_count = 0
        
        target_terms = case_terms.get(target_id, [])
        
        # Search for this case's terms in all other cases' text
        for other_case in cases_data:
            if other_case['case_id'] == target_id:
                continue
                
            other_text = (other_case['text'] or '').lower()
            
            # Count how many times this case is referenced
            for term in target_terms:
                if term and len(term) > 3:  # Avoid matching very short terms
                    term_lower = term.lower()
                    reference_count += other_text.count(term_lower)
        
        # Normalize score (0-1 scale)
        max_possible_refs = len(cases_data) * 5  # Rough estimate
        citation_scores[target_id] = min(reference_count / max_possible_refs, 1.0) if max_possible_refs > 0 else 0.0
    
    return citation_scores

def _calculate_temporal_scores(cases_data):
    """Calculate temporal significance scores - recent cases and landmark older cases get higher scores"""
    temporal_scores = {}
    
    years = [case['year'] for case in cases_data if case['year']]
    if not years:
        return {case['case_id']: 0.5 for case in cases_data}
    
    min_year = min(years)
    max_year = max(years) 
    year_range = max_year - min_year if max_year > min_year else 1
    
    for case in cases_data:
        year = case['year']
        case_id = case['case_id']
        
        if not year:
            temporal_scores[case_id] = 0.5
            continue
        
        # Recent cases (last 10 years) get high scores
        years_old = max_year - year
        if years_old <= 10:
            temporal_scores[case_id] = 1.0 - (years_old / 20)  # Scale from 1.0 to 0.5
        else:
            # Older cases - give higher scores to landmark older cases
            # Assume cases from key decades are more important
            decade = (year // 10) * 10
            landmark_decades = [1980, 1990, 2000, 2010]  # Adjust as needed
            
            if decade in landmark_decades:
                temporal_scores[case_id] = 0.7
            else:
                # Linear decay for very old cases
                temporal_scores[case_id] = max(0.1, 1.0 - (years_old / year_range))
    
    return temporal_scores

def _calculate_area_scores(cases_data):
    """Calculate area of law importance scores - broader/more fundamental areas get higher scores"""
    area_scores = {}
    
    # Define importance weights for different areas of law
    area_weights = {
        'constitutional': 1.0,
        'charter': 1.0,
        'criminal': 0.9,
        'contract': 0.8,
        'tort': 0.8,
        'property': 0.8,
        'administrative': 0.7,
        'family': 0.6,
        'employment': 0.6,
        'tax': 0.5,
        'immigration': 0.5,
        'default': 0.5
    }
    
    for case in cases_data:
        case_id = case['case_id']
        area = (case['area'] or '').lower()
        
        # Find matching area weight
        score = area_weights['default']
        for area_key, weight in area_weights.items():
            if area_key in area:
                score = weight
                break
        
        area_scores[case_id] = score
    
    return area_scores

def _calculate_length_scores(cases_data):
    """Calculate normalized case length scores - longer cases often contain more comprehensive analysis"""
    length_scores = {}
    
    lengths = [case['length'] for case in cases_data if case['length'] and case['length'] > 0]
    
    if not lengths:
        return {case['case_id']: 0.5 for case in cases_data}
    
    min_length = min(lengths)
    max_length = max(lengths)
    length_range = max_length - min_length if max_length > min_length else 1
    
    for case in cases_data:
        case_id = case['case_id']
        length = case['length'] or 0
        
        if length <= 0:
            length_scores[case_id] = 0.2  # Short/missing cases get low score
        else:
            # Normalize to 0-1 scale, with slight preference for longer cases
            normalized = (length - min_length) / length_range
            # Apply logarithmic scaling to avoid over-weighting extremely long cases
            import math
            length_scores[case_id] = min(0.9, 0.3 + 0.6 * math.log(1 + normalized))
    
    return length_scores

def generate_bradley_terry_structure(force_regenerate=False):
    """
    Generate the canonical Bradley-Terry block structure for all selected cases.
    
    This creates a centralized structure that all experiments will use, ensuring
    methodological consistency across experiments.
    
    Parameters:
    - force_regenerate: If True, regenerates structure even if it already exists
    
    Returns:
    - success: Boolean indicating if structure was generated successfully
    - message: Status message
    """
    try:
        # Check if structure already exists
        existing_structure = execute_sql("""
            SELECT COUNT(*) FROM v2_bradley_terry_structure
        """, fetch=True)
        
        if existing_structure and existing_structure[0][0] > 0 and not force_regenerate:
            return True, "Bradley-Terry structure already exists. Use force_regenerate=True to recreate."
        
        # Get count of selected cases
        selected_count = execute_sql("SELECT COUNT(*) FROM v2_experiment_selected_cases", fetch=True)[0][0]
        
        if selected_count == 0:
            return False, "No cases selected for experiments. Please select cases first."
        
        # Check if case count is divisible by 15
        if selected_count % 15 != 0:
            return False, f"Selected case count ({selected_count}) must be divisible by 15. Current remainder: {selected_count % 15}"
        
        # Calculate importance scores for all selected cases
        importance_scores = calculate_case_importance_scores()
        
        if not importance_scores:
            return False, "Could not calculate importance scores for selected cases."
        
        # Sort cases by importance score (highest first)
        sorted_cases = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Clear existing structure if regenerating
        if force_regenerate:
            execute_sql("DELETE FROM v2_bradley_terry_structure")
        
        # Calculate block parameters
        num_blocks = selected_count // 15
        core_cases_per_block = 12
        bridge_cases_per_block = 3
        
        # Assign core cases (top-ranked cases)
        core_cases = sorted_cases[:num_blocks * core_cases_per_block]
        
        # Assign bridge cases (remaining cases)
        bridge_cases = sorted_cases[num_blocks * core_cases_per_block:]
        
        # Generate block assignments
        block_assignments = []
        
        # Assign core cases to blocks (distribute evenly)
        for i, (case_id, score) in enumerate(core_cases):
            block_number = (i // core_cases_per_block) + 1
            block_assignments.append({
                'case_id': case_id,
                'block_number': block_number,
                'case_role': 'core',
                'importance_score': score
            })
        
        # Assign bridge cases to blocks (distribute to connect blocks)
        bridge_case_index = 0
        for block_num in range(1, num_blocks + 1):
            # Each block gets bridge_cases_per_block bridge cases
            for _ in range(bridge_cases_per_block):
                if bridge_case_index < len(bridge_cases):
                    case_id, score = bridge_cases[bridge_case_index]
                    block_assignments.append({
                        'case_id': case_id,
                        'block_number': block_num,
                        'case_role': 'bridge',
                        'importance_score': score
                    })
                    bridge_case_index += 1
        
        # Insert assignments into database
        for assignment in block_assignments:
            execute_sql("""
                INSERT INTO v2_bradley_terry_structure 
                (case_id, block_number, case_role, importance_score, assigned_by)
                VALUES (?, ?, ?, ?, ?)
            """, (
                assignment['case_id'],
                assignment['block_number'], 
                assignment['case_role'],
                assignment['importance_score'],
                'system'
            ))
        
        return True, f"Successfully generated Bradley-Terry structure: {num_blocks} blocks, {len(core_cases)} core cases, {len(bridge_cases)} bridge cases."
    
    except Exception as e:
        return False, f"Error generating Bradley-Terry structure: {str(e)}"

def get_bradley_terry_structure():
    """
    Get the current Bradley-Terry block structure.
    
    Returns:
    - structure: List of dicts with case assignments, or None if no structure exists
    """
    try:
        structure_data = execute_sql("""
            SELECT s.case_id, s.block_number, s.case_role, s.importance_score,
                   c.case_name, c.citation, c.decision_year, c.area_of_law
            FROM v2_bradley_terry_structure s
            JOIN v2_cases c ON s.case_id = c.case_id
            ORDER BY s.block_number, s.case_role DESC, s.importance_score DESC
        """, fetch=True)
        
        if not structure_data:
            return None
        
        structure = []
        for row in structure_data:
            case_id, block_num, role, score, name, citation, year, area = row
            structure.append({
                'case_id': case_id,
                'block_number': block_num,
                'case_role': role,
                'importance_score': float(score) if score else 0.0,
                'case_name': name,
                'citation': citation,
                'decision_year': year,
                'area_of_law': area
            })
        
        return structure
    
    except Exception as e:
        st.error(f"Error retrieving Bradley-Terry structure: {e}")
        return None

def get_block_summary():
    """
    Get a summary of the Bradley-Terry block structure.
    
    Returns:
    - summary: Dict with block statistics
    """
    try:
        structure = get_bradley_terry_structure()
        if not structure:
            return None
        
        # Calculate summary statistics
        blocks = {}
        total_core = 0
        total_bridge = 0
        
        for case in structure:
            block_num = case['block_number']
            role = case['case_role']
            
            if block_num not in blocks:
                blocks[block_num] = {'core': 0, 'bridge': 0, 'total': 0}
            
            blocks[block_num][role] += 1
            blocks[block_num]['total'] += 1
            
            if role == 'core':
                total_core += 1
            else:
                total_bridge += 1
        
        return {
            'total_blocks': len(blocks),
            'total_cases': len(structure),
            'total_core_cases': total_core,
            'total_bridge_cases': total_bridge,
            'cases_per_block': 15,
            'core_per_block': 12,
            'bridge_per_block': 3,
            'blocks_detail': blocks
        }
    
    except Exception as e:
        st.error(f"Error generating block summary: {e}")
        return None

def generate_bradley_terry_comparison_pairs():
    """
    Generate the optimal comparison pairs based on the Bradley-Terry block structure.
    
    Returns:
    - pairs: List of (case_id_1, case_id_2) tuples for all required comparisons
    - block_info: Dict mapping each pair to its block information
    """
    try:
        structure = get_bradley_terry_structure()
        if not structure:
            return [], {}
        
        # Group cases by block
        blocks = {}
        for case in structure:
            block_num = case['block_number']
            if block_num not in blocks:
                blocks[block_num] = []
            blocks[block_num].append(case)
        
        # Generate all pairwise comparisons within each block
        comparison_pairs = []
        pair_block_info = {}
        
        for block_num, block_cases in blocks.items():
            # Generate all pairwise combinations within this block
            for i in range(len(block_cases)):
                for j in range(i + 1, len(block_cases)):
                    case1 = block_cases[i]
                    case2 = block_cases[j]
                    
                    pair = (case1['case_id'], case2['case_id'])
                    comparison_pairs.append(pair)
                    
                    # Store block info for this pair
                    pair_block_info[pair] = {
                        'block_number': block_num,
                        'case1_role': case1['case_role'],
                        'case2_role': case2['case_role'],
                        'case1_name': case1['case_name'],
                        'case2_name': case2['case_name'],
                        'case1_importance': case1['importance_score'],
                        'case2_importance': case2['importance_score']
                    }
        
        return comparison_pairs, pair_block_info
    
    except Exception as e:
        st.error(f"Error generating comparison pairs: {e}")
        return [], {}