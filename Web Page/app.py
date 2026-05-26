from flask import Flask, request, render_template, jsonify, session, redirect, url_for
import pymysql
import os
from datetime import datetime
from urllib.parse import urlparse
import pandas as pd
from itertools import permutations

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # TODO: Replace with actual secret key

# ------------------ 1) Latin Square Generation ------------------
def generate_latin_square(N):
    """
    Generate an N×N Latin square where each row is a different permutation.
    """
    
    # Base row (1,2,...,N)
    base_row = list(range(1, N+1))
    
    # Get all possible permutations
    all_permutations = list(permutations(base_row))
    
    # Select N distinct permutations
    square = []
    for i in range(N):
        # Use a fixed algorithm to select permutations, ensuring each selection is different
        square.append(list(all_permutations[i * (len(all_permutations) // N)]))
    
    return square

# Pre-generate a 9×9 Latin square for arranging 9 documents
LATIN_9x9 = generate_latin_square(9)

# ------------------ 2) User-Row Mapping Function ------------------
def get_user_row_index(user_id, total_rows=9):
    """
    Deterministically assign a Latin square row index based on user ID
    """
    # Ensure user_id is a string
    user_id_str = str(user_id)
    
    # Use a better hash function - position-weighted character hash
    hash_value = sum(ord(c) * (i+1) for i, c in enumerate(user_id_str))
    
    # Modulo to get row index
    return hash_value % total_rows
    
# ------------------ 2) MySQL Connection ------------------
def get_connection():
    """
    Parse MySQL DSN (host, port, user, password, db) from MYSQL_URL and return connection
    """
    url = os.environ["MYSQL_URL"]  # e.g. "mysql://root:xxxx@containers-xxx:3306/railway"
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port
    user = parsed.username
    password = parsed.password
    db = parsed.path.lstrip('/')
    return pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        db=db,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

# ------------------ 3) Initialize Database Tables ------------------
def init_db():
    """
    Create four tables: queries, documents, orders, logs (if they don't exist).
    Does not insert any sample data.
    """
    conn = get_connection()
    try:
        with conn.cursor() as c:
            # --- 1) logs table ---
            c.execute("SHOW TABLES LIKE 'logs'")
            if not c.fetchone():
                c.execute('''
                    CREATE TABLE logs (
                      id INT AUTO_INCREMENT PRIMARY KEY,
                      user_id VARCHAR(100) NOT NULL,     -- User ID
                      qid INT DEFAULT 0,                 -- Query ID (optional)
                      docno VARCHAR(255) DEFAULT '',     -- Document number, using VARCHAR
                      event_type VARCHAR(100) NOT NULL,  -- "PASSAGE_SELECTION", "OPEN_DOC", ...
                      start_idx INT DEFAULT -1,          -- Selection start index, -1 if none
                      end_idx INT DEFAULT -1,            -- Selection end index, -1 if none
                      duration INT DEFAULT 0,            -- Time elapsed from last to current event
                      pass_flag TINYINT DEFAULT 0,       -- 0 or 1
                      timestamp DATETIME                 -- Record time
                    )
                ''')
                print("Created table: logs")

            # --- 2) documents table ---
            c.execute("SHOW TABLES LIKE 'documents'")
            if not c.fetchone():
                # If documents table doesn't exist, create it with docno as VARCHAR
                c.execute('''
                    CREATE TABLE documents (
                        id INT PRIMARY KEY,
                        qid INT,
                        docno VARCHAR(255),
                        content TEXT
                    )
                ''')
                print("Created table: documents")
            else:
                # Modify docno column type to VARCHAR
                try:
                    c.execute("ALTER TABLE documents MODIFY COLUMN docno VARCHAR(255)")
                    print("Modified docno column to VARCHAR(255)")
                except Exception as e:
                    print(f"Error modifying column type: {e}")

            # --- 3) orders table ---
            c.execute("SHOW TABLES LIKE 'orders'")
            if not c.fetchone():
                c.execute('''
                    CREATE TABLE orders (
                        user_id VARCHAR(100) NOT NULL,
                        query_id INT NOT NULL,
                        doc_order TEXT,
                        PRIMARY KEY (user_id, query_id)
                    )
                ''')
                print("Created table: orders")

            # --- 4) queries table ---
            c.execute("SHOW TABLES LIKE 'queries'")
            if not c.fetchone():
                # If queries table doesn't exist, create it
                c.execute('''
                    CREATE TABLE queries (
                        id INT PRIMARY KEY,
                        content TEXT
                    )
                ''')
                print("Created table: queries")

        conn.commit()
    finally:
        conn.close()

# ------------------ 4) Route Logic ------------------

AVAILABLE_QUERY_IDS = []

def get_available_query_ids():
    """
    Get all available query IDs from database
    Returns a list sorted by ID
    """
    query_ids = []
    try:
        conn = get_connection()
        try:
            with conn.cursor() as c:
                c.execute("SELECT id FROM queries ORDER BY id")
                query_ids = [row['id'] for row in c.fetchall()]
        finally:
            conn.close()
    except Exception as e:
        print(f"ERROR: Failed to get available query IDs: {str(e)}")
    return query_ids

# Load query ID list when application starts
@app.before_first_request
def load_query_ids():
    """Load all available query IDs when application starts"""
    global AVAILABLE_QUERY_IDS
    AVAILABLE_QUERY_IDS = get_available_query_ids()
    print(f"INFO: Loaded {len(AVAILABLE_QUERY_IDS)} available query IDs: {AVAILABLE_QUERY_IDS}")

# Replace original index route, add query ID information
@app.route("/", methods=["GET", "POST"])
def index():
    """
    Homepage: User enters user_id and accepts T&C.
    """
    global AVAILABLE_QUERY_IDS
    
    # Ensure query ID list is loaded
    if not AVAILABLE_QUERY_IDS:
        AVAILABLE_QUERY_IDS = get_available_query_ids()
        
    if request.method == "POST":
        user_id = request.form.get("user_id")
        terms = request.form.get("terms")
        if not terms:
            return render_template("index.html", error="Please accept the Terms and Conditions.")
        if not user_id:
            return render_template("index.html", error="Please enter your User ID.")
        
        session["user_id"] = user_id
        
        # If there are available query IDs, redirect to the first one
        if AVAILABLE_QUERY_IDS:
            first_query_position = 1  # This is position, not ID
            return redirect(url_for("query_page", query_position=first_query_position))
        else:
            return render_template("index.html", error="No queries available in the database.")
            
    return render_template("index.html", query_count=len(AVAILABLE_QUERY_IDS))

# Modify query_page route to use position instead of ID
@app.route("/query/<int:query_position>", methods=["GET", "POST"])
def query_page(query_position):
    """
    For each user_id + query_position:
    1) Get actual query ID through position (1,2,3...)
    2) Use hash of user_id to determine Latin square row index (deterministic method)
    3) Directly apply Latin square permutation to documents
    4) Store ordering on first visit (for logging only)
    """
    global AVAILABLE_QUERY_IDS
    
    # Ensure query ID list is loaded
    if not AVAILABLE_QUERY_IDS:
        AVAILABLE_QUERY_IDS = get_available_query_ids()
    
    # Ensure query position is valid
    if query_position < 1 or query_position > len(AVAILABLE_QUERY_IDS):
        return redirect(url_for("index"))
    
    # Get actual query ID
    query_id = AVAILABLE_QUERY_IDS[query_position - 1]
    print(f"INFO: Query position {query_position} maps to database query ID {query_id}")
    
    if "user_id" not in session:
        return redirect(url_for("index"))
    
    user_id = session["user_id"]
    print(f"DEBUG: Processing query_page for user_id={user_id}, query_position={query_position}, query_id={query_id}")
    
    # Record when user first visits a query
    is_first_visit = False
    if "visited_positions" not in session:
        session["visited_positions"] = []
        
    if query_position not in session["visited_positions"]:
        session["visited_positions"].append(query_position)
        is_first_visit = True
        print(f"DEBUG: First visit to query position {query_position} for user {user_id}")

    if request.method == "POST":
        if query_position < len(AVAILABLE_QUERY_IDS):
            next_position = query_position + 1
            return redirect(url_for("query_page", query_position=next_position))
        else:
            return redirect(url_for("thanks"))

    conn = get_connection()
    try:
        with conn.cursor() as c:
            # Get query content
            c.execute("SELECT content FROM queries WHERE id=%s", (query_id,))
            q_row = c.fetchone()
            if not q_row:
                return "Query not found", 404
            query_text = q_row["content"]

            # Get all documents for this query
            c.execute("SELECT id, docno, content FROM documents WHERE qid=%s", (query_id,))
            raw_docs = c.fetchall()

            # Use deterministic hash method to assign user to a Latin square row
            user_row_index = get_user_row_index(user_id, total_rows=9)
            print(f"DEBUG: User {user_id} assigned to Latin square row {user_row_index}")
            
            # Get Latin square row and directly apply permutation
            latin_order = LATIN_9x9[user_row_index]
            print(f"DEBUG: Latin square row {user_row_index} order: {latin_order}")
            
            # Apply permutation directly to documents (no need for further shuffling)
            documents = [raw_docs[i-1] for i in latin_order]
            
            # Only store ordering on first visit (for logging)
            if is_first_visit:
                doc_order_str = ",".join(str(d["id"]) for d in documents)
                c.execute("""
                    INSERT INTO orders (user_id, query_id, doc_order)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE doc_order=%s
                """, (user_id, query_id, doc_order_str, doc_order_str))
                conn.commit()
                print(f"DEBUG: Stored document order for user {user_id}, query {query_id}: {doc_order_str}")

    finally:
        conn.close()

    # Calculate navigation info
    current_query_number = query_position
    total_queries = len(AVAILABLE_QUERY_IDS)
    
    # Determine if this is the last query
    is_last_query = (query_position == len(AVAILABLE_QUERY_IDS))

    return render_template(
        "query.html",
        user_id=user_id,
        qid=query_id,
        query_text=query_text,
        docs=documents,
        query_position=query_position,
        current_query_number=current_query_number,
        total_queries=total_queries,
        is_last_query=is_last_query
    )

@app.route("/thanks")
def thanks():
    return render_template("thanks.html")

@app.route("/pause")
def pause():
    return render_template("pause.html")

@app.route("/api/log", methods=["POST"])
def log_event():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    user_id    = data.get('userId')
    qid        = data.get('qid', 0)
    docno      = data.get('docno', 0)
    event_type = data.get('eventType', "")
    start_idx  = data.get('startIndex', -1)
    end_idx    = data.get('endIndex', -1)
    duration   = data.get('duration', 0)
    pass_flag  = data.get('passFlag', 0)
    timestamp  = datetime.now()  # Use datetime.now() for a proper DATETIME value

    conn = get_connection()
    try:
        with conn.cursor() as c:
            sql = """
                INSERT INTO logs (user_id, qid, docno, event_type, start_idx, end_idx, duration, pass_flag, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            c.execute(sql, (user_id, qid, docno, event_type, start_idx, end_idx, duration, pass_flag, timestamp))
        conn.commit()
    finally:
        conn.close()

    return jsonify({'message': 'Log received'}), 200

def import_df_to_database(df):
    """
    Import data from DataFrame to database queries and documents tables.
    DataFrame should contain qid, query, docno, text columns.
    If tables don't exist, will create table structure first.
    """
    # Ensure table structure exists first
    init_db()
    
    # Insert queries first, then documents
    insert_queries_from_df(df)
    insert_documents_from_df(df)
    
def insert_queries_from_df(df):
    """
    Import unique queries from DataFrame to queries table.
    """
    # Get unique queries
    unique_queries = df[['qid', 'query']].drop_duplicates().reset_index(drop=True)
    
    conn = get_connection()
    try:
        with conn.cursor() as c:
            # Check existing query IDs
            c.execute("SELECT id FROM queries")
            existing_ids = [row['id'] for row in c.fetchall()]
            
            # Prepare insert data
            insert_data = []
            for _, row in unique_queries.iterrows():
                # Skip if query ID already exists
                if row['qid'] in existing_ids:
                    continue
                    
                insert_data.append((
                    row['qid'],      # id
                    row['query']     # content
                ))
                
            if insert_data:
                # Batch insert data to queries table
                insert_sql = """
                    INSERT INTO queries (id, content)
                    VALUES (%s, %s)
                """
                c.executemany(insert_sql, insert_data)
                
        conn.commit()
        print(f"Successfully inserted {len(insert_data)} records to queries table")
    except Exception as e:
        print(f"Error inserting queries: {e}")
        conn.rollback()
    finally:
        conn.close()

def insert_documents_from_df(df):
    """
    Import document data from DataFrame to documents table.
    Maps 'text' column to database 'content' column.
    """
    conn = get_connection()
    try:
        with conn.cursor() as c:
            # Get current max id as starting point
            c.execute("SELECT MAX(id) as max_id FROM documents")
            result = c.fetchone()
            start_id = result['max_id'] if result['max_id'] is not None else 0
            
            # Prepare insert data
            insert_data = []
            for i, row in df.iterrows():
                doc_id = start_id + i + 1
                insert_data.append((
                    doc_id,           # id
                    row['qid'],       # qid
                    row['docno'],     # docno (VARCHAR type, can directly insert string)
                    row['text']       # content (mapped from 'text')
                ))
                
            # Batch insert data to documents table
            insert_sql = """
                INSERT INTO documents (id, qid, docno, content)
                VALUES (%s, %s, %s, %s)
            """
            c.executemany(insert_sql, insert_data)
            
        conn.commit()
        print(f"Successfully inserted {len(df)} records to documents table")
    except Exception as e:
        print(f"Error inserting documents: {e}")
        conn.rollback()
    finally:
        conn.close()


# Automatically create tables and insert initial data when container/local starts (if empty)
init_db()

def clear_tables_before_import():
    """Simply clear all related tables"""
    try:
        conn = get_connection()
        try:
            with conn.cursor() as c:
                
                # Delete documents table data
                c.execute("DELETE FROM documents")
                
                # Delete queries table data
                c.execute("DELETE FROM queries")
                
                # Commit transaction
                conn.commit()
                
                print("Cleared all related tables, ready to import new data")
                return True
        except Exception as e:
            conn.rollback()  # Rollback on error
            print(f"Error clearing tables: {e}")
            return False
        finally:
            conn.close()
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return False
        
clear_tables_before_import()

try:
    # Read CSV file
    df = pd.read_csv("result_preference_based_with_text_gold.csv")
    
    # Check if required columns exist
    required_columns = ['qid', 'query', 'docno', 'text']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: CSV file missing the following columns: {', '.join(missing_columns)}")
    else:
        # Keep only data from the last 21 queries
        unique_qids = df['qid'].unique()[22:]
        filtered_df = df[df['qid'].isin(unique_qids)]
        
        # Check if documents table is empty
        is_table_empty = True  # Default assume table is empty
        conn = get_connection()
        try:
            with conn.cursor() as c:
                c.execute("SELECT COUNT(*) AS count FROM documents")
                result = c.fetchone()
                document_count = result['count'] if result else 0
                
                if document_count > 0:
                    print(f"Skipping import: documents table already contains {document_count} records")
                    is_table_empty = False
        finally:
            conn.close()  # Close connection immediately
        
        # If table is empty, execute import
        if is_table_empty:
            import_df_to_database(filtered_df)  # Use filtered DataFrame
            print(f"Successfully imported data from last 21 queries from result_preference_based_with_text_gold.csv")

except Exception as e:
    print(f"Error importing data: {e}")

def check_query_document_counts():
    """Check document count for each query ID and output statistics"""
    conn = get_connection()
    try:
        with conn.cursor() as c:
            # Get all query IDs
            c.execute("SELECT id FROM queries ORDER BY id")
            query_ids = [row['id'] for row in c.fetchall()]
            
            print(f"Total of {len(query_ids)} queries")
            
            # Check document count for each query ID
            query_counts = {}
            for qid in query_ids:
                c.execute("SELECT COUNT(*) as doc_count FROM documents WHERE qid=%s", (qid,))
                count = c.fetchone()['doc_count']
                query_counts[qid] = count
                print(f"Query ID {qid} has {count} documents")
            
            # Statistical analysis
            exact_nine = sum(1 for count in query_counts.values() if count == 9)
            less_than_nine = sum(1 for count in query_counts.values() if count < 9)
            more_than_nine = sum(1 for count in query_counts.values() if count > 9)
            zero_docs = sum(1 for count in query_counts.values() if count == 0)
            
            print("\nStatistics:")
            print(f"Queries with exactly 9 documents: {exact_nine}")
            print(f"Queries with less than 9 documents: {less_than_nine}")
            print(f"Queries with more than 9 documents: {more_than_nine}")
            print(f"Queries with no documents: {zero_docs}")
            
            # Find query IDs with less than 9 documents
            if less_than_nine > 0:
                print("\nQuery IDs with less than 9 documents:")
                for qid, count in query_counts.items():
                    if count < 9:
                        print(f"Query ID {qid}: {count} documents")
    finally:
        conn.close()

# Call function
check_query_document_counts()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)