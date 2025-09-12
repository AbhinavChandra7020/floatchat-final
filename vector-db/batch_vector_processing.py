import ollama
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port="5432",
    database="postgres",
    user="postgres", 
    password="mypassword"  # Your container password
)
cur = conn.cursor()

def process_batch_fixed(batch_size=50):
    """Process records in batches - FIXED VERSION for argo_profiles"""
    
    # Get initial count
    cur.execute("SELECT COUNT(*) FROM argo_profiles WHERE description IS NOT NULL AND embedding IS NULL")
    initial_total = cur.fetchone()[0]
    print(f"Total records to process: {initial_total}")
    
    if initial_total == 0:
        print("No records need processing - all done!")
        return
    
    batch_num = 1
    processed_count = 0
    
    while True:
        print(f"Processing batch {batch_num}")
        
        # Always get the FIRST batch_size records that need processing
        cur.execute("""
            SELECT id, description 
            FROM argo_profiles 
            WHERE description IS NOT NULL AND embedding IS NULL 
            ORDER BY id
            LIMIT %s
        """, (batch_size,))
        
        batch = cur.fetchall()
        if not batch:
            print("No more records to process!")
            break
            
        print(f"  Processing IDs {batch[0][0]} to {batch[-1][0]} ({len(batch)} records)")
        
        batch_success_count = 0
        for record_id, description in batch:
            try:
                response = ollama.embeddings(model='nomic-embed-text:v1.5', prompt=description)
                embedding = response['embedding']
                vector_str = '[' + ','.join(map(str, embedding)) + ']'
                cur.execute("UPDATE argo_profiles SET embedding = %s WHERE id = %s", 
                           (vector_str, record_id))
                batch_success_count += 1
            except Exception as e:
                print(f"  Error processing record {record_id}: {e}")
        
        # Commit the batch
        conn.commit()
        processed_count += batch_success_count
        
        print(f"  Batch {batch_num} complete: {batch_success_count}/{len(batch)} successful")
        print(f"  Total processed so far: {processed_count}/{initial_total}")
        
        batch_num += 1

    print("All records processed!")
    
    # Final verification
    cur.execute("SELECT COUNT(*) FROM argo_profiles WHERE description IS NOT NULL AND embedding IS NULL")
    remaining = cur.fetchone()[0]
    print(f"Final check: {remaining} records still need processing")

# Run the batch processing
process_batch_fixed()
cur.close()
conn.close()