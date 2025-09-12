import ollama
import psycopg2
import time

conn = psycopg2.connect(
    host="localhost", port="5432", database="postgres",
    user="postgres", password="mypassword"
)
cur = conn.cursor()

def process_half_database(batch_size=50):
    """Process only half the records for testing"""
    
    # Get total count and calculate half
    cur.execute("SELECT COUNT(*) FROM argo_profiles WHERE description IS NOT NULL AND embedding IS NULL")
    total_records = cur.fetchone()[0]
    half_records = total_records // 2
    
    print(f"Total records available: {total_records}")
    print(f"Processing only half: {half_records} records")
    
    if total_records == 0:
        print("No records need processing - all done!")
        return
    
    batch_num = 1
    processed_count = 0
    
    while processed_count < half_records:
        print(f"\n=== Processing batch {batch_num} ===")
        
        # Calculate remaining records to process
        remaining_to_process = half_records - processed_count
        current_batch_size = min(batch_size, remaining_to_process)
        
        cur.execute("""
            SELECT id, description 
            FROM argo_profiles 
            WHERE description IS NOT NULL AND embedding IS NULL 
            ORDER BY id
            LIMIT %s
        """, (current_batch_size,))
        
        batch = cur.fetchall()
        if not batch:
            print("No more records to process!")
            break
            
        print(f"Batch has {len(batch)} records (IDs {batch[0][0]} to {batch[-1][0]})")
        
        batch_success_count = 0
        for i, (record_id, description) in enumerate(batch, 1):
            try:
                print(f"  {i}/{len(batch)}: Processing record ID {record_id}...")
                
                response = ollama.embeddings(model='nomic-embed-text:v1.5', prompt=description)
                embedding = response['embedding']
                vector_str = '[' + ','.join(map(str, embedding)) + ']'
                
                cur.execute("UPDATE argo_profiles SET embedding = %s WHERE id = %s", 
                           (vector_str, record_id))
                
                batch_success_count += 1
                print(f"    SUCCESS: Record {record_id} embedded ({len(embedding)} dimensions)")
                
            except Exception as e:
                print(f"    ERROR: Record {record_id} failed - {e}")
        
        # Commit the batch
        conn.commit()
        processed_count += batch_success_count
        
        print(f"\nBatch {batch_num} summary:")
        print(f"  Successful: {batch_success_count}/{len(batch)}")
        print(f"  Total processed: {processed_count}/{half_records}")
        print(f"  Progress: {processed_count/half_records*100:.1f}%")
        
        batch_num += 1
        
        # Stop if we've reached our target
        if processed_count >= half_records:
            print(f"\nReached target of {half_records} records!")
            break

    print(f"\n=== HALF DATABASE PROCESSING COMPLETE ===")
    print(f"Records processed: {processed_count}")
    print(f"Records remaining unprocessed: {total_records - processed_count}")

# Run the half database processing
process_half_database()
cur.close()
conn.close()