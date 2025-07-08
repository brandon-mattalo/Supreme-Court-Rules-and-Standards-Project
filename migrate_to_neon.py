#!/usr/bin/env python3
"""
Data Migration Script: SQLite to Neon PostgreSQL
Transfers all data from local SQLite database to Neon PostgreSQL cloud database
"""

import os
import sqlite3
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

def get_local_connection():
    """Get connection to local SQLite database"""
    db_path = os.path.join(os.path.dirname(__file__), "parquet", "scc_cases.db")
    if not os.path.exists(db_path):
        print(f"❌ Local database not found at: {db_path}")
        sys.exit(1)
    return sqlite3.connect(db_path)

def get_neon_connection():
    """Get connection to Neon PostgreSQL database"""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ DATABASE_URL environment variable not set!")
        print("Please set your Neon PostgreSQL connection string in .env file")
        print("Example: DATABASE_URL=postgresql://username:password@hostname:port/database")
        sys.exit(1)
    
    try:
        engine = create_engine(database_url)
        return engine
    except Exception as e:
        print(f"❌ Failed to connect to Neon database: {e}")
        sys.exit(1)

def create_postgresql_tables(neon_engine):
    """Create tables in PostgreSQL with proper schema"""
    
    # PostgreSQL-compatible CREATE TABLE statements
    create_statements = [
        """
        CREATE TABLE IF NOT EXISTS cases (
            case_id SERIAL PRIMARY KEY,
            case_name TEXT,
            citation TEXT UNIQUE,
            decision_year INTEGER,
            area_of_law TEXT,
            scc_url TEXT,
            full_text TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS legal_tests (
            test_id SERIAL PRIMARY KEY,
            case_id INTEGER REFERENCES cases(case_id),
            test_novelty TEXT,
            extracted_test_summary TEXT,
            source_paragraphs TEXT,
            source_type TEXT,
            validation_status TEXT DEFAULT 'pending',
            validator_name TEXT,
            bt_score REAL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS legal_test_comparisons (
            comparison_id SERIAL PRIMARY KEY,
            test_id_1 INTEGER REFERENCES legal_tests(test_id),
            test_id_2 INTEGER REFERENCES legal_tests(test_id),
            more_rule_like_test_id INTEGER REFERENCES legal_tests(test_id),
            reasoning TEXT,
            comparator_name TEXT,
            comparison_method TEXT DEFAULT 'human',
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(test_id_1, test_id_2)
        );
        """
    ]
    
    print("🔧 Creating PostgreSQL tables...")
    with neon_engine.connect() as conn:
        for statement in create_statements:
            conn.execute(text(statement))
        conn.commit()
    print("✅ Tables created successfully")

def migrate_table(table_name, sqlite_conn, neon_engine, id_mapping=None):
    """Migrate a single table from SQLite to PostgreSQL"""
    
    print(f"📦 Migrating table: {table_name}")
    
    # Read data from SQLite
    df = pd.read_sql(f"SELECT * FROM {table_name}", sqlite_conn)
    
    if df.empty:
        print(f"⚠️  Table {table_name} is empty, skipping...")
        return {}
    
    print(f"📊 Found {len(df)} rows in {table_name}")
    
    # Handle ID mapping for foreign key relationships
    if id_mapping:
        for old_col, new_col in id_mapping.items():
            if old_col in df.columns:
                df[old_col] = df[old_col].map(new_col)
    
    # For tables with auto-incrementing IDs, we need to handle ID mapping
    id_map = {}
    
    if table_name == 'cases':
        # Drop the case_id column to let PostgreSQL auto-generate it
        old_ids = df['case_id'].tolist()
        df_to_insert = df.drop('case_id', axis=1)
        
        # Insert data and get new IDs
        with neon_engine.connect() as conn:
            df_to_insert.to_sql(table_name, conn, if_exists='append', index=False, method='multi')
            
            # Get the mapping of old IDs to new IDs by matching on citation (unique field)
            result = conn.execute(text("SELECT case_id, citation FROM cases ORDER BY case_id"))
            new_data = result.fetchall()
            
            # Create mapping based on citation
            citation_to_new_id = {row[1]: row[0] for row in new_data}
            for i, old_id in enumerate(old_ids):
                citation = df.iloc[i]['citation']
                if citation in citation_to_new_id:
                    id_map[old_id] = citation_to_new_id[citation]
    
    elif table_name == 'legal_tests':
        # Map case_id using the cases mapping, drop test_id for auto-generation
        old_ids = df['test_id'].tolist()
        df_to_insert = df.drop('test_id', axis=1)
        
        # Insert data
        with neon_engine.connect() as conn:
            df_to_insert.to_sql(table_name, conn, if_exists='append', index=False, method='multi')
            
            # Get new IDs (simplified - in order of insertion)
            result = conn.execute(text("SELECT test_id FROM legal_tests ORDER BY test_id"))
            new_ids = [row[0] for row in result.fetchall()]
            
            # Create mapping (this assumes insertions maintain order)
            start_idx = len(new_ids) - len(old_ids)
            for i, old_id in enumerate(old_ids):
                id_map[old_id] = new_ids[start_idx + i]
    
    else:
        # For comparison table, just insert (IDs will be auto-generated)
        df_to_insert = df.drop('comparison_id', axis=1) if 'comparison_id' in df.columns else df
        with neon_engine.connect() as conn:
            df_to_insert.to_sql(table_name, conn, if_exists='append', index=False, method='multi')
    
    print(f"✅ Successfully migrated {len(df)} rows to {table_name}")
    return id_map

def validate_migration(sqlite_conn, neon_engine):
    """Validate that the migration was successful"""
    
    print("🔍 Validating migration...")
    
    tables = ['cases', 'legal_tests', 'legal_test_comparisons']
    
    for table in tables:
        # Count rows in SQLite
        sqlite_count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", sqlite_conn)['count'][0]
        
        # Count rows in PostgreSQL
        with neon_engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            pg_count = result.fetchone()[0]
        
        print(f"📊 {table}: SQLite={sqlite_count}, PostgreSQL={pg_count}")
        
        if sqlite_count != pg_count:
            print(f"⚠️  Row count mismatch in {table}!")
        else:
            print(f"✅ {table} migration verified")

def main():
    print("🚀 Starting migration from SQLite to Neon PostgreSQL")
    print("=" * 60)
    
    # Get connections
    print("🔌 Connecting to databases...")
    sqlite_conn = get_local_connection()
    neon_engine = get_neon_connection()
    
    try:
        # Create PostgreSQL tables
        create_postgresql_tables(neon_engine)
        
        # Migrate tables in dependency order
        print("📦 Starting data migration...")
        
        # 1. Migrate cases first (no dependencies)
        cases_id_map = migrate_table('cases', sqlite_conn, neon_engine)
        
        # 2. Migrate legal_tests (depends on cases)
        tests_id_map = migrate_table('legal_tests', sqlite_conn, neon_engine, 
                                   {'case_id': cases_id_map})
        
        # 3. Migrate legal_test_comparisons (depends on legal_tests)
        migrate_table('legal_test_comparisons', sqlite_conn, neon_engine,
                     {'test_id_1': tests_id_map, 'test_id_2': tests_id_map, 
                      'more_rule_like_test_id': tests_id_map})
        
        # Validate migration
        validate_migration(sqlite_conn, neon_engine)
        
        print("=" * 60)
        print("🎉 Migration completed successfully!")
        print("💡 You can now update your .env file to use the Neon database:")
        print(f"   DATABASE_URL={os.getenv('DATABASE_URL')}")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)
        
    finally:
        sqlite_conn.close()
        neon_engine.dispose()

if __name__ == "__main__":
    main()