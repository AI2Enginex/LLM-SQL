import pymysql

import sqlalchemy
import pandas as pd

# Class for connecting to a MySQL database
class DatabaseConnect:
    
    # Initialize the class with password and database name
    def __init__(self, password: str, database: str, server: str):
        self.password = password
        self.database_name = database
        self.server = server
        # Create SQLAlchemy engine and establish a connection
        self.engine = sqlalchemy.create_engine(
                f'mysql+pymysql://root:{self.password}@localhost:3306/{self.database_name}')
        
    # Method to try establishing a database connection
    def try_connection(self):
        try:
            self.conn = pymysql.connect(
                host=self.server,
                user='root',
                password=self.password,
                db=self.database_name,
            )

            # Return cursor if connection is successful
            return self.conn.cursor()
        except:
            return None
        
    # Method to read an entire table into a pandas DataFrame
    def read_table(self, table_name: str):
        try:
            if self.engine is None:
                raise ValueError("Database connection not established. Call try_connection() first.")
            print("connection successfull")
            query = f"SELECT * FROM {table_name};"
            return pd.read_sql(query, self.engine)
        except Exception as e:
            raise e
    
    # Method to read a table's schema
    def get_table_schema(self, table_name: str):
        try:
            if self.engine is None:
                raise ValueError("Database connection not established. Call try_connection() first.")

            query = f"""
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_KEY
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{self.database_name}' AND TABLE_NAME = '{table_name}';
            """

            df = pd.read_sql(query, self.engine)

            if df.empty:
                return f"Table '{table_name}' does not exist or has no columns."

            # Build a human-readable schema string
            schema_lines = [f"Table: {table_name}", "Columns:"]
            for _, row in df.iterrows():
                col_info = f"- {row['COLUMN_NAME']} ({row['DATA_TYPE']})"
                if row['COLUMN_KEY'] == 'PRI':
                    col_info += " [PRIMARY KEY]"
                if row['IS_NULLABLE'] == 'NO':
                    col_info += " [NOT NULL]"
                schema_lines.append(col_info)

            return "\n".join(schema_lines)
        except Exception as e:
            raise e
    

    # Method to read Schemas for multiple tables
    def get_multiple_table_schemas(self, table_names: list):
        try:
            all_schemas = []
            for table in table_names:
                schema = self.get_table_schema(table)
                all_schemas.append(schema)
            return "\n\n".join(all_schemas)
        except Exception as e:
            raise e
