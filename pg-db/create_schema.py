import psycopg2

conn = psycopg2.connect(
    host="localhost", port="5432", database="postgres", 
    user="postgres", password="mypassword"
)
cursor = conn.cursor()

# Drop existing tables
cursor.execute("DROP TABLE IF EXISTS argo_measurements CASCADE;")
cursor.execute("DROP TABLE IF EXISTS argo_profiles CASCADE;")
cursor.execute("DROP TABLE IF EXISTS argo_calibration CASCADE;")

# Profiles table (one row per profile)
cursor.execute("""
    CREATE TABLE argo_profiles (
        id SERIAL PRIMARY KEY,
        platform_number INTEGER,
        cycle_number INTEGER,
        direction CHAR(1),
        data_centre CHAR(2),
        dc_reference VARCHAR(64),
        data_state_indicator CHAR(4),
        data_mode CHAR(1),
        platform_type VARCHAR(32),
        float_serial_no VARCHAR(32),
        firmware_version VARCHAR(32),
        wmo_inst_type INTEGER,
        project_name VARCHAR(64),
        pi_name VARCHAR(256),
        juld TIMESTAMP,
        juld_qc INTEGER,
        juld_location TIMESTAMP,
        latitude DOUBLE PRECISION,
        longitude DOUBLE PRECISION,
        position_qc INTEGER,
        positioning_system VARCHAR(16),
        profile_pres_qc CHAR(1),
        profile_temp_qc CHAR(1),
        profile_psal_qc CHAR(1),
        vertical_sampling_scheme TEXT,
        config_mission_number INTEGER,
        description TEXT,
        embedding vector(768)
    );
""")

# Measurements table (one row per depth level per profile)
cursor.execute("""
    CREATE TABLE argo_measurements (
        id SERIAL PRIMARY KEY,
        profile_id INTEGER REFERENCES argo_profiles(id),
        level_index INTEGER,
        pres REAL,
        pres_qc CHAR(1),
        pres_adjusted REAL,
        pres_adjusted_qc CHAR(1),
        pres_adjusted_error REAL,
        temp REAL,
        temp_qc CHAR(1), 
        temp_adjusted REAL,
        temp_adjusted_qc CHAR(1),
        temp_adjusted_error REAL,
        psal REAL,
        psal_qc CHAR(1),
        psal_adjusted REAL,
        psal_adjusted_qc CHAR(1),
        psal_adjusted_error REAL
    );
""")

# Global metadata table (file-level info)
cursor.execute("""
    CREATE TABLE argo_file_metadata (
        id SERIAL PRIMARY KEY,
        filename VARCHAR(255),
        data_type VARCHAR(32),
        format_version VARCHAR(16),
        handbook_version VARCHAR(16),
        date_creation TIMESTAMP,
        date_update TIMESTAMP
    );
""")

# Create indexes
cursor.execute("CREATE INDEX idx_profiles_platform ON argo_profiles(platform_number);")
cursor.execute("CREATE INDEX idx_profiles_location ON argo_profiles(latitude, longitude);")
cursor.execute("CREATE INDEX idx_profiles_time ON argo_profiles(juld);")
cursor.execute("CREATE INDEX idx_measurements_profile ON argo_measurements(profile_id);")

conn.commit()
cursor.close()
conn.close()
print("Final comprehensive schema created!")