-- Database Schema Updates for Sample Groups and Active Sampling
-- Version 2.1 - Moving from block-based to ASAP active sampling

-- 1. Create sample groups table
CREATE TABLE IF NOT EXISTS v2_sample_groups (
    group_id SERIAL PRIMARY KEY,
    group_name TEXT UNIQUE NOT NULL,
    description TEXT,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT DEFAULT 'researcher'
);

-- 2. Create sample group members table (many-to-many relationship)
CREATE TABLE IF NOT EXISTS v2_sample_group_members (
    member_id SERIAL PRIMARY KEY,
    group_id INTEGER NOT NULL,
    case_id INTEGER NOT NULL,
    added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    added_by TEXT DEFAULT 'researcher',
    FOREIGN KEY (group_id) REFERENCES v2_sample_groups (group_id) ON DELETE CASCADE,
    FOREIGN KEY (case_id) REFERENCES v2_cases (case_id),
    UNIQUE(group_id, case_id)
);

-- 3. Add sample_group_id to experiments table
ALTER TABLE v2_experiments ADD COLUMN sample_group_id INTEGER;
ALTER TABLE v2_experiments ADD FOREIGN KEY (sample_group_id) REFERENCES v2_sample_groups (group_id);

-- 4. Create active sampling state table
CREATE TABLE IF NOT EXISTS v2_active_sampling_state (
    state_id SERIAL PRIMARY KEY,
    experiment_id INTEGER NOT NULL,
    comparison_matrix TEXT, -- JSON serialized numpy array
    current_scores TEXT, -- JSON serialized scores
    spearman_correlation REAL,
    iteration_number INTEGER DEFAULT 0,
    convergence_reached BOOLEAN DEFAULT FALSE,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES v2_experiments (experiment_id),
    UNIQUE(experiment_id)
);

-- 5. Add columns to track active sampling progress
ALTER TABLE v2_experiment_comparisons ADD COLUMN sampling_iteration INTEGER;
ALTER TABLE v2_experiment_comparisons ADD COLUMN information_gain REAL;

-- 6. Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_v2_sample_groups_name ON v2_sample_groups (group_name);
CREATE INDEX IF NOT EXISTS idx_v2_sample_group_members_group_id ON v2_sample_group_members (group_id);
CREATE INDEX IF NOT EXISTS idx_v2_sample_group_members_case_id ON v2_sample_group_members (case_id);
CREATE INDEX IF NOT EXISTS idx_v2_experiments_sample_group_id ON v2_experiments (sample_group_id);
CREATE INDEX IF NOT EXISTS idx_v2_active_sampling_state_experiment_id ON v2_active_sampling_state (experiment_id);
CREATE INDEX IF NOT EXISTS idx_v2_experiment_comparisons_iteration ON v2_experiment_comparisons (sampling_iteration);

-- 7. Drop the old Bradley-Terry structure table (no longer needed with active sampling)
-- DROP TABLE IF EXISTS v2_bradley_terry_structure;

-- 8. Remove the old v2_experiment_selected_cases table (replaced by sample groups)
-- We'll keep it for now for backward compatibility but stop using it
-- DROP TABLE IF EXISTS v2_experiment_selected_cases;