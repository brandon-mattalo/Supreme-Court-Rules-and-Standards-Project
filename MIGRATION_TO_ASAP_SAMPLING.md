# Migration to ASAP Active Sampling - Implementation Summary

## Overview

This document summarizes the implementation of the new active sampling system using ASAP (Active Sampling for Pairwise Comparisons) and sample groups, replacing the previous block-based Bradley-Terry approach.

## Key Changes Made

### 1. Database Schema Updates ✅
- **New Tables Added:**
  - `v2_sample_groups` - Store sample group definitions
  - `v2_sample_group_members` - Many-to-many relationship between groups and cases
  - `v2_active_sampling_state` - Track active sampling state per experiment
  
- **New Columns Added:**
  - `v2_experiments.sample_group_id` - Link experiments to sample groups
  - `v2_experiment_comparisons.sampling_iteration` - Track iteration in active sampling
  - `v2_experiment_comparisons.information_gain` - Track information gain per comparison

### 2. Active Sampling Implementation ✅
- **Created `utils/asap_integration.py`:**
  - `ASAPSampler` class for managing active sampling process
  - Spearman correlation convergence check (>0.99)
  - Fallback implementation when ASAP library not available
  - O(n log n) cost estimation formula

### 3. Sample Group Management ✅
- **Created `utils/sample_group_management.py`:**
  - Create, view, modify, and delete sample groups
  - Add/remove cases from groups with any number (not restricted to multiples of 15)
  - Group statistics and member management
  - Duplicate groups functionality

### 4. Updated User Interface ✅
- **Created `pages/case_management_v2.py`:**
  - New sample group-based case management UI
  - Replace single global case selection with multiple sample groups
  - Interactive group management with add/remove cases
  - No restrictions on group size (any number of cases allowed)

### 5. Cost Estimation Updates ✅
- **Updated `utils/case_management.py`:**
  - Replaced block-based O(n²) estimation with O(n log n) active sampling
  - Uses `estimate_required_comparisons()` function
  - Significant efficiency gains for large datasets

### 6. Integration Updates ✅
- **Updated `pages/dashboard.py`:**
  - Added new database schema initialization
  - Updated navigation to use new case management UI
  - Import new sample group functionality

## Files Created
1. `utils/asap_integration.py` - ASAP wrapper and active sampling logic
2. `utils/sample_group_management.py` - Sample group database operations
3. `pages/case_management_v2.py` - New case management UI
4. `database_schema_updates.sql` - SQL schema updates
5. `ASAP_INSTALL_NOTES.md` - Installation instructions
6. `MIGRATION_TO_ASAP_SAMPLING.md` - This summary document

## Files Modified
1. `pages/dashboard.py` - Database schema, navigation, imports
2. `utils/case_management.py` - Cost estimation formula
3. `requirements.txt` - Added scipy dependency

## Remaining Tasks

### High Priority
1. **Update experiment configuration** to include sample group selection
2. **Update experiment execution** to use sample groups instead of global selection
3. **Replace block-based comparison logic** with ASAP active sampling in execution

### Medium Priority
1. **Test the new system** with sample data
2. **Create migration scripts** for existing experiments

### Low Priority
1. **Remove old block-based code** and references
2. **Update documentation** for new workflow

## Key Benefits

### Efficiency Gains
- **O(n log n) vs O(n²)** comparison requirements
- **Example:** 1000 cases: ~13,000 comparisons vs 500,000 (97% reduction)
- **Adaptive convergence** stops when Spearman correlation >0.99

### Flexibility Improvements
- **Multiple sample groups** instead of single global selection
- **Any group size** (not restricted to multiples of 15)
- **Experiment-specific** case selection
- **Group management** with add/remove/duplicate functionality

### Scientific Improvements
- **Active sampling** for more efficient data collection
- **Convergence detection** ensures statistical reliability
- **Information gain** optimization for pair selection

## Migration Strategy

### For Existing Users
1. **Backward compatibility** - Old system still works
2. **Gradual migration** - Create sample groups for new experiments
3. **Data preservation** - Existing experiments unchanged

### For New Users
1. **Start with sample groups** - Use new case management UI
2. **Create multiple groups** for different experiment types
3. **Leverage active sampling** for efficient comparisons

## Technical Notes

### ASAP Library Installation
- **Not available via pip** - requires manual installation
- **Fallback implementation** provided for when ASAP unavailable
- **See `ASAP_INSTALL_NOTES.md`** for detailed instructions

### Database Compatibility
- **PostgreSQL and SQLite** support maintained
- **Automatic schema updates** on first run
- **Indexes added** for performance optimization

## Next Steps

1. **Test the new UI** by running the application
2. **Create sample groups** and add cases
3. **Complete remaining experiment execution updates**
4. **Test active sampling** with small datasets
5. **Migrate existing experiments** if needed

## Contact

For questions about this implementation, refer to the individual module documentation or the CLAUDE.md file in the project root.