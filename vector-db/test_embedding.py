import ollama
import psycopg2

conn = psycopg2.connect(
    host="localhost", port="5432", database="postgres",
    user="postgres", password="mypassword"
)
cursor = conn.cursor()

print("Testing Ollama embedding on first 3 records...")

# Get first 3 records
cursor.execute("SELECT id, description FROM argo_profiles WHERE description IS NOT NULL ORDER BY id LIMIT 3")
records = cursor.fetchall()

for record_id, description in records:
    try:
        print(f"Testing record {record_id}: {description[:50]}...")
        
        response = ollama.embeddings(model='nomic-embed-text:v1.5', prompt=description)
        embedding = response['embedding']
        
        print(f"  Got embedding vector with {len(embedding)} dimensions")
        print(f"  First 5 values: {embedding[:5]}")
        
        # Test inserting into database
        vector_str = '[' + ','.join(map(str, embedding)) + ']'
        cursor.execute("UPDATE argo_profiles SET embedding = %s WHERE id = %s", 
                      (vector_str, record_id))
        conn.commit()
        print(f"  Successfully stored in database")
        
    except Exception as e:
        print(f"  Error: {e}")

cursor.close()
conn.close()
print("Test complete!")