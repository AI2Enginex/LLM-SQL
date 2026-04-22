import os
import pyodbc
import pandas as pd
# This class is responsible for forming a basic SQL Server connection string
class ConnectionString:

    def __init__(self, database_name: str, server_name: str):
        # Constructs a connection string for SQL Server using provided database and server names
        self.conn_str = (
            'DRIVER={SQL Server};' 
            f'SERVER={server_name};'
            f'DATABASE={database_name};'
        )

# This subclass creates an actual database connection using the constructed connection string
class MakeConnection(ConnectionString):

    def __init__(self, database: str, database_server: str):
        # Inherit and initialize the connection string from the base class
        super().__init__(database_name=database, server_name=database_server)

    def cursor_connection(self):
        """
        Establishes a connection to the SQL Server and returns the connection object.
        Includes exception handling for connection errors.
        """
        try:
            connection = pyodbc.connect(self.conn_str, timeout=10)
            print("success")  # Optional: Indicates a successful connection
            return connection
        except Exception as e:
            raise e  # Returns the exception if connection fails
        
# This class extends MakeConnection and performs database operations using a connected cursor
class DatabaseOperations(MakeConnection):

    def __init__(self, db_name: str, server: str):
        # Initialize the base connection and immediately create a cursor object
        super().__init__(database=db_name, database_server=server)
        self.cursor = self.cursor_connection().cursor()

    def read_from_table(self, table_name: str):
        """
        Fetches all records from the specified table and returns them as a pandas DataFrame.
        """
        try:
            sql_query = self.cursor.execute(f"SELECT * FROM {table_name};")

            # Fetch all rows and construct DataFrame using column names from query metadata
            data = sql_query.fetchall()
            df = pd.DataFrame(
                [list(data) for data in data],
                columns=[columns[0] for columns in sql_query.description]
            )
            return df
        except Exception as e:
            raise e  # Return any error encountered

    def get_table_schema(self, table_name: str):
        """
        Retrieves schema details (column names, types, nullability, defaults, primary keys)
        for a given table using INFORMATION_SCHEMA.
        """
        try:
            query = f"""
                SELECT 
                    c.COLUMN_NAME,
                    c.DATA_TYPE,
                    c.IS_NULLABLE,
                    c.COLUMN_DEFAULT,
                    CASE WHEN k.COLUMN_NAME IS NOT NULL THEN 'YES' ELSE 'NO' END AS IS_PRIMARY_KEY
                FROM INFORMATION_SCHEMA.COLUMNS c
                LEFT JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE k 
                    ON c.TABLE_NAME = k.TABLE_NAME 
                    AND c.COLUMN_NAME = k.COLUMN_NAME
                    AND k.CONSTRAINT_NAME IN (
                        SELECT CONSTRAINT_NAME 
                        FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS 
                        WHERE CONSTRAINT_TYPE = 'PRIMARY KEY'
                    )
                WHERE c.TABLE_NAME = '{table_name}';
            """

            self.cursor.execute(query)
            rows = self.cursor.fetchall()

            # Build a readable schema output
            schema_lines = [f"Table: {table_name}", "Columns:"]
            for row in rows:
                col, dtype, nullable, default, is_pk = row
                line = f"- {col} ({dtype})"
                if is_pk == "YES":
                    line += " [PRIMARY KEY]"
                if nullable == "NO":
                    line += " [NOT NULL]"
                if default:
                    line += f" [DEFAULT: {default}]"
                schema_lines.append(line)

            return "\n".join(schema_lines)

        except Exception as e:
            return str(e)  # Return the error message as a string
        
    def get_multiple_table_schemas(self, table_names: list):
        """
        Fetches schemas for multiple tables and returns them as a combined string.
        """
        try:
            all_schemas = []
            for table in table_names:
                schema = self.get_table_schema(table)
                all_schemas.append(schema)
            return "\n\n".join(all_schemas)
        except Exception as e:
            raise e  # Return any encountered exception


