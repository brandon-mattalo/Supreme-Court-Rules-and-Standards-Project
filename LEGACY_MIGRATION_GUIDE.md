# Legacy System Migration Guide

## Overview
The application has been completely migrated from the legacy case selection system to the new **Sample Groups** system with **ASAP Active Sampling**. This document explains what changed and how to migrate existing experiments.

## ✅ **What Was Removed**

### Legacy Case Selection System
- **`v2_experiment_selected_cases` table**: No longer actively used
- **Global case selection**: Experiments previously shared a single pool of selected cases
- **Block-based Bradley-Terry**: Fixed 15-case blocks with full pairwise comparisons

### Deprecated Functions
- `get_experiment_selected_cases()` - Use sample groups instead
- `add_cases_to_experiments()` - Use sample group management instead
- `clear_selected_cases()` - Use sample group deletion instead
- Legacy case management UI - Use "Case Management (Sample Groups)" page

## ✅ **New System Benefits**

### Sample Groups
- **Multiple groups per experiment**: Each experiment can use a different sample group
- **Flexible group sizes**: Any number of cases (not restricted to multiples of 15)
- **Reusable groups**: Sample groups can be shared across experiments
- **Better organization**: Group cases by topic, time period, or other criteria

### ASAP Active Sampling
- **99%+ efficiency gain**: O(n log n) vs O(n²) comparisons
- **Adaptive convergence**: Stops when Spearman correlation > 0.99
- **Intelligent pair selection**: ASAP algorithm chooses most informative comparisons
- **Real-time cost estimates**: Based on actual comparison requirements

## ⚠️ **Required Actions for Existing Users**

### For Existing Experiments
1. **Legacy experiments without sample groups**:
   - Cannot run new extractions or comparisons
   - Must configure a sample group to proceed
   - Cost estimates will show 0 until sample group is assigned

2. **Migration steps**:
   - Go to "Case Management (Sample Groups)"
   - Create new sample groups with desired cases
   - Edit experiment configuration to assign sample group

### For New Experiments
1. **Create sample groups first**:
   - Use "Case Management (Sample Groups)" page
   - Add cases using filters (year, area of law, etc.)
   - Create multiple groups for different experiment types

2. **Configure experiments**:
   - Select sample group during experiment creation
   - Configure button is disabled for non-draft experiments
   - Cost estimates are calculated per sample group

## 🔧 **Technical Changes**

### Database Schema
- **New tables**: `v2_sample_groups`, `v2_sample_group_members`, `v2_active_sampling_state`
- **New column**: `v2_experiments.sample_group_id`
- **Legacy table**: `v2_experiment_selected_cases` kept for compatibility but not used

### API Changes
- **Extraction function**: Now queries sample group members instead of global selection
- **Comparison function**: Uses ASAP active sampling instead of block-based pairs
- **Cost calculation**: Based on sample group size and O(n log n) formula

### UI Changes
- **Library Overview**: Shows accurate case counts per experiment's sample group
- **Experiment Detail**: Displays sample group information and warnings
- **Configure button**: Disabled for in-progress/complete experiments

## 📝 **Migration Script Example**

If you need to migrate data programmatically:

```sql
-- Create sample group from legacy selected cases
INSERT INTO v2_sample_groups (group_name, description)
VALUES ('Legacy Selected Cases', 'Migrated from old case selection system');

-- Get the new group ID
SET @group_id = LAST_INSERT_ID();

-- Copy cases to sample group
INSERT INTO v2_sample_group_members (group_id, case_id)
SELECT @group_id, case_id 
FROM v2_experiment_selected_cases;

-- Update experiments to use the new sample group
UPDATE v2_experiments 
SET sample_group_id = @group_id 
WHERE sample_group_id IS NULL;
```

## ✅ **Verification Steps**

1. **Check experiments**: All should show sample group information in detail view
2. **Test extraction**: Should work with sample group cases
3. **Test comparison**: Should use ASAP active sampling
4. **Verify costs**: Should show realistic O(n log n) estimates

## 🆘 **Troubleshooting**

### "No sample group assigned" Error
- **Solution**: Configure a sample group for the experiment
- **Steps**: Edit experiment → Select sample group → Save

### "Cannot run extractions" Error  
- **Solution**: Ensure experiment has sample group with cases
- **Steps**: Check sample group has members, verify experiment assignment

### Cost estimates show $0.00
- **Solution**: Assign sample group to experiment
- **Steps**: Edit experiment configuration and select sample group

## 📞 **Support**

For questions about the migration:
1. Check experiment configuration has sample group assigned
2. Verify sample groups have cases added
3. Ensure experiments are in "draft" status for configuration changes

The new system provides much better efficiency and organization while maintaining all functionality of the previous system.