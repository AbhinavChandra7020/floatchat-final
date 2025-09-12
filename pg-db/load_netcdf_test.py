import xarray as xr
import psycopg2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

def clean_byte_string(value):
    """Clean byte strings from NetCDF data"""
    if isinstance(value, bytes):
        return value.decode('utf-8').strip()
    return str(value).strip() if value is not None else None

def convert_datetime64_to_python(dt64):
    """Convert numpy datetime64 to Python datetime"""
    if pd.isna(dt64):
        return None
    return pd.to_datetime(dt64).to_pydatetime()

def process_netcdf_file(file_path, cursor):
    """Process one NetCDF file and insert into database"""
    print(f"Processing: {file_path.name}")
    
    ds = xr.open_dataset(file_path)
    n_profiles = ds.sizes['n_prof']
    n_levels = ds.sizes['n_levels']
    
    profiles_inserted = 0
    measurements_inserted = 0
    
    # Insert file metadata
    cursor.execute("""
        INSERT INTO argo_file_metadata (filename, data_type, format_version, handbook_version)
        VALUES (%s, %s, %s, %s);
    """, (
        file_path.name,
        clean_byte_string(ds.data_type.values),
        clean_byte_string(ds.format_version.values),
        clean_byte_string(ds.handbook_version.values)
    ))
    
    for prof_idx in range(n_profiles):
        # Extract and convert profile data
        platform_num = int(clean_byte_string(ds.platform_number.values[prof_idx]))
        cycle_num = int(ds.cycle_number.values[prof_idx])
        lat = float(ds.latitude.values[prof_idx])
        lon = float(ds.longitude.values[prof_idx])
        juld = convert_datetime64_to_python(ds.juld.values[prof_idx])
        
        # Create description for embedding
        description = f"Argo float {platform_num} cycle {cycle_num} at {lat:.3f}°N, {lon:.3f}°E on {str(juld)[:10] if juld else 'unknown date'}"
        
        # Insert profile
        cursor.execute("""
            INSERT INTO argo_profiles (
                platform_number, cycle_number, direction, data_centre, data_mode,
                latitude, longitude, juld, position_qc, profile_pres_qc, 
                profile_temp_qc, profile_psal_qc, description
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
            RETURNING id;
        """, (
            platform_num, cycle_num,
            clean_byte_string(ds.direction.values[prof_idx]),
            clean_byte_string(ds.data_centre.values[prof_idx]),
            clean_byte_string(ds.data_mode.values[prof_idx]),
            lat, lon, juld,
            int(clean_byte_string(ds.position_qc.values[prof_idx])),
            clean_byte_string(ds.profile_pres_qc.values[prof_idx]),
            clean_byte_string(ds.profile_temp_qc.values[prof_idx]),
            clean_byte_string(ds.profile_psal_qc.values[prof_idx]),
            description
        ))
        
        profile_id = cursor.fetchone()[0]
        profiles_inserted += 1
        
        # Insert measurements
        for level_idx in range(n_levels):
            pres = ds.pres.values[prof_idx, level_idx]
            temp = ds.temp.values[prof_idx, level_idx]
            psal = ds.psal.values[prof_idx, level_idx]
            
            # Only insert if we have valid pressure data
            if not np.isnan(pres):
                cursor.execute("""
                    INSERT INTO argo_measurements (
                        profile_id, level_index, pres, temp, psal,
                        pres_qc, temp_qc, psal_qc,
                        pres_adjusted, temp_adjusted, psal_adjusted
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """, (
                    profile_id, level_idx,
                    float(pres),
                    float(temp) if not np.isnan(temp) else None,
                    float(psal) if not np.isnan(psal) else None,
                    clean_byte_string(ds.pres_qc.values[prof_idx, level_idx]),
                    clean_byte_string(ds.temp_qc.values[prof_idx, level_idx]),
                    clean_byte_string(ds.psal_qc.values[prof_idx, level_idx]),
                    float(ds.pres_adjusted.values[prof_idx, level_idx]) if not np.isnan(ds.pres_adjusted.values[prof_idx, level_idx]) else None,
                    float(ds.temp_adjusted.values[prof_idx, level_idx]) if not np.isnan(ds.temp_adjusted.values[prof_idx, level_idx]) else None,
                    float(ds.psal_adjusted.values[prof_idx, level_idx]) if not np.isnan(ds.psal_adjusted.values[prof_idx, level_idx]) else None
                ))
                measurements_inserted += 1
    
    ds.close()
    return profiles_inserted, measurements_inserted

# Main processing - load first 3 files for testing
conn = psycopg2.connect(
    host="localhost", port="5432", database="postgres",
    user="postgres", password="mypassword"
)
cursor = conn.cursor()
data_root = Path("ocean-data/data/incois")

total_profiles = 0
total_measurements = 0
file_count = 0

for date_folder in sorted(data_root.iterdir()):
    if date_folder.is_dir() and file_count < 3:
        nc_files = list(date_folder.glob("*.nc"))
        for nc_file in nc_files:
            if file_count < 3:
                try:
                    profiles, measurements = process_netcdf_file(nc_file, cursor)
                    total_profiles += profiles
                    total_measurements += measurements
                    file_count += 1
                    conn.commit()
                    print(f"  ✓ {profiles} profiles, {measurements} measurements")
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    conn.rollback()

print(f"\nTOTAL: {total_profiles} profiles, {total_measurements} loaded")
cursor.close()
conn.close()