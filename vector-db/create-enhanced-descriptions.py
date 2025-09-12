import psycopg2
import numpy as np

conn = psycopg2.connect(
    host="localhost", port="5432", database="postgres",
    user="postgres", password="mypassword"
)
cursor = conn.cursor()

def create_enhanced_description(profile_id):
    """Create rich description combining profile + measurement data"""
    
    # Get profile info
    cursor.execute("""
        SELECT platform_number, cycle_number, latitude, longitude, juld, description
        FROM argo_profiles WHERE id = %s
    """, (profile_id,))
    profile = cursor.fetchone()
    
    if not profile:
        return None
    
    platform_num, cycle_num, lat, lon, juld, orig_desc = profile
    
    # Get measurement summary statistics
    cursor.execute("""
        SELECT 
            COUNT(*) as num_measurements,
            MIN(pres) as min_depth, MAX(pres) as max_depth,
            AVG(temp) as avg_temp, MIN(temp) as min_temp, MAX(temp) as max_temp,
            AVG(psal) as avg_salinity, MIN(psal) as min_salinity, MAX(psal) as max_salinity
        FROM argo_measurements 
        WHERE profile_id = %s AND pres IS NOT NULL AND temp IS NOT NULL AND psal IS NOT NULL
    """, (profile_id,))
    
    stats = cursor.fetchone()
    if not stats or stats[0] == 0:
        return orig_desc  # Return original if no measurements
    
    num_meas, min_depth, max_depth, avg_temp, min_temp, max_temp, avg_sal, min_sal, max_sal = stats
    
    # Create enhanced description
    enhanced_desc = f"""
{orig_desc}. 
Oceanographic profile: {num_meas} measurements from {min_depth:.1f}m to {max_depth:.1f}m depth. 
Temperature range: {min_temp:.2f}°C to {max_temp:.2f}°C (avg: {avg_temp:.2f}°C). 
Salinity range: {min_sal:.3f} to {max_sal:.3f} PSU (avg: {avg_sal:.3f} PSU). 
Water column characteristics: {'shallow' if max_depth < 500 else 'deep'} profile, 
{'warm' if avg_temp > 20 else 'temperate' if avg_temp > 10 else 'cold'} waters, 
{'high' if avg_sal > 35 else 'normal' if avg_sal > 34 else 'low'} salinity.
    """.strip().replace('\n', ' ')
    
    return enhanced_desc

# Process all profiles to create enhanced descriptions
cursor.execute("SELECT COUNT(*) FROM argo_profiles")
total_profiles = cursor.fetchone()[0]
print(f"Enhancing descriptions for {total_profiles} profiles...")

cursor.execute("SELECT id FROM argo_profiles ORDER BY id")
profile_ids = cursor.fetchall()

batch_size = 100
processed = 0

for i in range(0, len(profile_ids), batch_size):
    batch = profile_ids[i:i+batch_size]
    
    for (profile_id,) in batch:
        enhanced_desc = create_enhanced_description(profile_id)
        if enhanced_desc:
            cursor.execute("""
                UPDATE argo_profiles 
                SET description = %s 
                WHERE id = %s
            """, (enhanced_desc, profile_id))
        
        processed += 1
    
    conn.commit()
    print(f"Processed {processed}/{total_profiles} profiles...")

print("Enhanced descriptions complete!")

# Show sample of enhanced description
cursor.execute("SELECT description FROM argo_profiles WHERE id = 1")
sample = cursor.fetchone()[0]
print(f"\nSample enhanced description:\n{sample}")

cursor.close()
conn.close()