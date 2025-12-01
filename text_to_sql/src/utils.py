import sqlite3

def get_tables(cursor):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    return [row[0] for row in cursor.fetchall()]

def get_columns(cursor, table):
    cursor.execute(f"PRAGMA table_info({table});")
    return [(row[1], row[2]) for row in cursor.fetchall()]  # (column_name, data_type)

def get_foreign_keys(cursor, table):
    cursor.execute(f"PRAGMA foreign_key_list({table});")
    return [(row[3], row[2], row[4]) for row in cursor.fetchall()]  # (from_col, ref_table, ref_col)

def describe_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    tables = get_tables(cursor)
    
    description = []
    for table in tables:
        description.append(f"\n🧱 Table: {table}")
        # Columns
        columns = get_columns(cursor, table)
        for name, dtype in columns:
            description.append(f"   🔸 {name} ({dtype})")
        # Foreign keys
        fks = get_foreign_keys(cursor, table)
        if fks:
            description.append("   🔗 Foreign Keys:")
            for from_col, ref_table, ref_col in fks:
                description.append(f"      {from_col} → {ref_table}.{ref_col}")
    conn.close()
    
    return "\n".join(description)

# For backwards compatibility, also provide a print version
def print_database_description(db_path):
    print(describe_database(db_path))

if __name__ == "__main__":
    # Use it
    db_name = "formula_1"
    path_to_db = f"dbs/dev_databases/{db_name}/{db_name}.sqlite"
    print_database_description(path_to_db)
