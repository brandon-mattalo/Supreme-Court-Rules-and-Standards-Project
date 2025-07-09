#!/usr/bin/env python3
"""
Systematic Secondary Citation Update Script

This script implements a clean, systematic approach to update secondary citations:
1. Filter Excel to 1517 records with both citations
2. Build concordance mapping with database (handling bidirectional matching)
3. Filter to cases needing updates (~1038 cases)
4. Execute selective updates with proper citation matching
5. Validate results
"""

import pandas as pd
from config import execute_sql
import sys

def load_and_filter_excel():
    """Phase 1: Load Excel file and filter to records with both citations"""
    print("=== Phase 1: Loading and Filtering Excel Data ===")
    
    try:
        # Load Excel file
        excel_path = 'SCC Decisions Database_with_all_citations.xlsx'
        df = pd.read_excel(excel_path)
        print(f"Loaded Excel file with {len(df)} total rows")
        
        # Filter to records with both primary and secondary citations
        df_both = df[
            df['Citation'].notna() & 
            (df['Citation'] != '') &
            df['Alternate Citation'].notna() & 
            (df['Alternate Citation'] != '')
        ].copy()
        
        print(f"Filtered to {len(df_both)} records with both primary and secondary citations")
        
        # Clean up the data
        df_both['Citation'] = df_both['Citation'].astype(str).str.strip()
        df_both['Alternate Citation'] = df_both['Alternate Citation'].astype(str).str.strip()
        df_both['Case Name'] = df_both['Case Name'].astype(str).str.strip()
        
        return df_both
        
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None

def analyze_database_state():
    """Phase 2: Query database to analyze current state"""
    print("\n=== Phase 2: Analyzing Database State ===")
    
    try:
        # Get total cases in database
        total_cases = execute_sql("SELECT COUNT(*) FROM v2_cases", fetch=True)[0][0]
        print(f"Total cases in database: {total_cases}")
        
        # Get cases with secondary citations
        cases_with_secondary = execute_sql(
            "SELECT COUNT(*) FROM v2_cases WHERE secondary_citation IS NOT NULL", 
            fetch=True
        )[0][0]
        print(f"Cases with secondary citations: {cases_with_secondary}")
        
        # Get cases without secondary citations
        cases_without_secondary = execute_sql(
            "SELECT COUNT(*) FROM v2_cases WHERE secondary_citation IS NULL", 
            fetch=True
        )[0][0]
        print(f"Cases without secondary citations: {cases_without_secondary}")
        
        return {
            'total_cases': total_cases,
            'with_secondary': cases_with_secondary,
            'without_secondary': cases_without_secondary
        }
        
    except Exception as e:
        print(f"Error analyzing database state: {e}")
        return None

def build_concordance_mapping(excel_df):
    """Phase 3: Build concordance mapping between Excel records and database cases"""
    print("\n=== Phase 3: Building Concordance Mapping ===")
    
    try:
        # Load all database cases once (much more efficient)
        print("Loading all database cases...")
        all_db_cases = execute_sql(
            "SELECT case_id, case_name, citation, secondary_citation FROM v2_cases",
            fetch=True
        )
        
        # Create lookup dictionary for fast matching
        db_citation_lookup = {}
        for case_id, case_name, citation, secondary_citation in all_db_cases:
            db_citation_lookup[citation] = {
                'case_id': case_id,
                'case_name': case_name,
                'citation': citation,
                'secondary_citation': secondary_citation
            }
        
        print(f"Loaded {len(all_db_cases)} database cases")
        print(f"Processing {len(excel_df)} Excel records...")
        
        mapping = []
        found_count = 0
        not_found_count = 0
        
        for idx, row in excel_df.iterrows():
            excel_primary = row['Citation']
            excel_secondary = row['Alternate Citation']
            excel_case_name = row['Case Name']
            
            # Try to match using Excel primary citation
            db_match = db_citation_lookup.get(excel_primary)
            match_type = 'primary'
            
            # If not found, try Excel secondary citation
            if not db_match:
                db_match = db_citation_lookup.get(excel_secondary)
                match_type = 'secondary'
            
            if db_match:
                # Determine which Excel citation to use as secondary
                if match_type == 'primary':
                    # Excel primary matches DB primary, so use Excel secondary
                    secondary_to_add = excel_secondary
                else:
                    # Excel secondary matches DB primary, so use Excel primary
                    secondary_to_add = excel_primary
                
                mapping.append({
                    'excel_index': idx,
                    'excel_case_name': excel_case_name,
                    'excel_primary': excel_primary,
                    'excel_secondary': excel_secondary,
                    'db_case_id': db_match['case_id'],
                    'db_case_name': db_match['case_name'],
                    'db_citation': db_match['citation'],
                    'db_secondary_citation': db_match['secondary_citation'],
                    'match_type': match_type,
                    'secondary_to_add': secondary_to_add,
                    'needs_update': db_match['secondary_citation'] is None
                })
                found_count += 1
            else:
                not_found_count += 1
                if not_found_count <= 10:  # Show first 10 not found
                    print(f"  Not found: {excel_case_name} | {excel_primary} | {excel_secondary}")
            
            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{len(excel_df)} records...")
        
        print(f"\nConcordance Results:")
        print(f"  Found in database: {found_count}")
        print(f"  Not found in database: {not_found_count}")
        
        return mapping
        
    except Exception as e:
        print(f"Error building concordance mapping: {e}")
        return None

def filter_cases_needing_updates(mapping):
    """Phase 4: Filter to cases needing secondary citation updates"""
    print("\n=== Phase 4: Filtering Cases Needing Updates ===")
    
    try:
        # Filter to cases that need updates (don't already have secondary citations)
        cases_needing_updates = [m for m in mapping if m['needs_update']]
        
        print(f"Cases needing secondary citation updates: {len(cases_needing_updates)}")
        
        # Show some examples
        print("\nSample cases needing updates:")
        for i, case in enumerate(cases_needing_updates[:5]):
            print(f"  {i+1}. {case['db_case_name'][:50]}...")
            print(f"     DB Citation: {case['db_citation']}")
            print(f"     Will add: {case['secondary_to_add']}")
            print(f"     Match type: {case['match_type']}")
            print()
        
        return cases_needing_updates
        
    except Exception as e:
        print(f"Error filtering cases needing updates: {e}")
        return None

def execute_selective_updates(cases_to_update):
    """Phase 5: Execute selective updates with proper citation matching"""
    print("\n=== Phase 5: Executing Selective Updates ===")
    
    try:
        print(f"Updating {len(cases_to_update)} cases...")
        
        success_count = 0
        error_count = 0
        
        for i, case in enumerate(cases_to_update):
            try:
                # Update the secondary citation
                execute_sql(
                    "UPDATE v2_cases SET secondary_citation = ? WHERE case_id = ?",
                    (case['secondary_to_add'], case['db_case_id'])
                )
                
                success_count += 1
                
                if success_count % 100 == 0:
                    print(f"  Updated {success_count}/{len(cases_to_update)} cases...")
                    
            except Exception as e:
                error_count += 1
                if error_count <= 5:  # Show first 5 errors
                    print(f"  Error updating case {case['db_case_id']}: {e}")
        
        print(f"\nUpdate Results:")
        print(f"  Successfully updated: {success_count}")
        print(f"  Errors: {error_count}")
        
        return success_count, error_count
        
    except Exception as e:
        print(f"Error executing updates: {e}")
        return 0, 0

def validate_results():
    """Phase 6: Validate final results"""
    print("\n=== Phase 6: Validating Results ===")
    
    try:
        # Check final counts
        total_with_secondary = execute_sql(
            "SELECT COUNT(*) FROM v2_cases WHERE secondary_citation IS NOT NULL",
            fetch=True
        )[0][0]
        
        print(f"Final count of cases with secondary citations: {total_with_secondary}")
        
        # Show some examples
        examples = execute_sql("""
            SELECT case_name, citation, secondary_citation 
            FROM v2_cases 
            WHERE secondary_citation IS NOT NULL 
            ORDER BY case_id 
            LIMIT 5
        """, fetch=True)
        
        print("\nSample updated cases:")
        for case_name, primary, secondary in examples:
            print(f"  {case_name[:50]}...")
            print(f"    Primary: {primary}")
            print(f"    Secondary: {secondary}")
            print()
        
        return total_with_secondary
        
    except Exception as e:
        print(f"Error validating results: {e}")
        return 0

def main():
    """Main execution function"""
    print("Starting Systematic Secondary Citation Update")
    print("=" * 50)
    
    # Phase 1: Load and filter Excel data
    excel_df = load_and_filter_excel()
    if excel_df is None:
        print("Failed to load Excel data. Exiting.")
        return
    
    # Phase 2: Analyze database state
    db_state = analyze_database_state()
    if db_state is None:
        print("Failed to analyze database state. Exiting.")
        return
    
    # Phase 3: Build concordance mapping
    mapping = build_concordance_mapping(excel_df)
    if mapping is None:
        print("Failed to build concordance mapping. Exiting.")
        return
    
    # Phase 4: Filter to cases needing updates
    cases_to_update = filter_cases_needing_updates(mapping)
    if cases_to_update is None:
        print("Failed to filter cases needing updates. Exiting.")
        return
    
    # Phase 5: Execute updates
    success_count, error_count = execute_selective_updates(cases_to_update)
    
    # Phase 6: Validate results
    final_count = validate_results()
    
    print(f"\n=== Summary ===")
    print(f"Excel records processed: {len(excel_df)}")
    print(f"Database matches found: {len(mapping)}")
    print(f"Cases updated: {success_count}")
    print(f"Final cases with secondary citations: {final_count}")
    print(f"Expected improvement: {success_count} new secondary citations added")

if __name__ == "__main__":
    main()