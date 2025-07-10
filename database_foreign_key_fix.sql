-- Fix Foreign Key Constraints to Add CASCADE Deletes
-- This prevents the foreign key violation error when deleting experiment data

-- PostgreSQL version
-- Note: PostgreSQL doesn't allow modifying existing foreign keys directly
-- We need to drop and recreate them

-- Drop existing foreign key constraints on v2_experiment_comparisons
ALTER TABLE v2_experiment_comparisons DROP CONSTRAINT IF EXISTS v2_experiment_comparisons_extraction_id_1_fkey;
ALTER TABLE v2_experiment_comparisons DROP CONSTRAINT IF EXISTS v2_experiment_comparisons_extraction_id_2_fkey;
ALTER TABLE v2_experiment_comparisons DROP CONSTRAINT IF EXISTS v2_experiment_comparisons_winner_id_fkey;

-- Recreate with CASCADE delete
ALTER TABLE v2_experiment_comparisons 
ADD CONSTRAINT v2_experiment_comparisons_extraction_id_1_fkey 
FOREIGN KEY (extraction_id_1) REFERENCES v2_experiment_extractions (extraction_id) ON DELETE CASCADE;

ALTER TABLE v2_experiment_comparisons 
ADD CONSTRAINT v2_experiment_comparisons_extraction_id_2_fkey 
FOREIGN KEY (extraction_id_2) REFERENCES v2_experiment_extractions (extraction_id) ON DELETE CASCADE;

ALTER TABLE v2_experiment_comparisons 
ADD CONSTRAINT v2_experiment_comparisons_winner_id_fkey 
FOREIGN KEY (winner_id) REFERENCES v2_experiment_extractions (extraction_id) ON DELETE CASCADE;

-- Also fix the experiment_id constraint for complete cleanup
ALTER TABLE v2_experiment_comparisons DROP CONSTRAINT IF EXISTS v2_experiment_comparisons_experiment_id_fkey;
ALTER TABLE v2_experiment_comparisons 
ADD CONSTRAINT v2_experiment_comparisons_experiment_id_fkey 
FOREIGN KEY (experiment_id) REFERENCES v2_experiments (experiment_id) ON DELETE CASCADE;

-- Fix extractions table as well
ALTER TABLE v2_experiment_extractions DROP CONSTRAINT IF EXISTS v2_experiment_extractions_experiment_id_fkey;
ALTER TABLE v2_experiment_extractions 
ADD CONSTRAINT v2_experiment_extractions_experiment_id_fkey 
FOREIGN KEY (experiment_id) REFERENCES v2_experiments (experiment_id) ON DELETE CASCADE;

-- Fix active sampling state table
ALTER TABLE v2_active_sampling_state DROP CONSTRAINT IF EXISTS v2_active_sampling_state_experiment_id_fkey;
ALTER TABLE v2_active_sampling_state 
ADD CONSTRAINT v2_active_sampling_state_experiment_id_fkey 
FOREIGN KEY (experiment_id) REFERENCES v2_experiments (experiment_id) ON DELETE CASCADE;

-- Note: For SQLite, we would need to recreate the tables entirely as it doesn't support dropping constraints
-- The immediate fix in the code (deleting comparisons before extractions) handles both PostgreSQL and SQLite