import os
import pandas as pd
import pyodbc
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import warnings
warnings.filterwarnings('ignore')

# Setting the API key for Google Generative AI service by assigning it to the environment variable 'GOOGLE_API_KEY'
api_key = os.environ['GOOGLE_API_KEY'] = "AIzaSyD3W4aK-sfGlWudxYvbve-kgijN_PD4S20"

# Configuring Google Generative AI module with the provided API key
genai.configure(api_key=api_key)
key = os.environ.get('GOOGLE_API_KEY')


def read_as_dataframe(data: None, feature_list: list):

    try:
        
        rows = [list(value) for value in data]
        return pd.DataFrame(rows, columns=feature_list)
    except Exception as e:
        return e
    
class GeminiModel:
    def __init__(self):

        # Initializing the GenerativeModel object with the 'gemini-pro' model
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        # Creating a GenerationConfig object with specific configuration parameters
        self.generation_config = genai.GenerationConfig(
            temperature=0,
            top_p=1.0,
            top_k=32,
            candidate_count=1,
            max_output_tokens=8192,
        )

class GeminiChatModel(GeminiModel):
    def __init__(self):
        super().__init__()  # Calling the constructor of the superclass (GeminiModel)
        # Starting a chat using the model inherited from GeminiModel
        self.chat = self.model.start_chat()

class ChatGoogleGENAI:
    def __init__(self):
        
        # Initializing the ChatGoogleGenerativeAI object with specified parameters
        self.llm=ChatGoogleGenerativeAI(temperature=0.7,model="gemini-2.5-flash", google_api_key=key,top_p=1.0,
            top_k=32,
            candidate_count=1,
            max_output_tokens=3000)

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
            return e  # Returns the exception if connection fails

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
            return e  # Return any error encountered

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
            return e  # Return any encountered exception


class PromptTemplates:
    """
    Contains different prompt templates for diffrent tasks.
    """
    @classmethod
    def query_table(cls):
        
        try:
            template="""
                You are an expert SQL assistant.

                Given the following table:

                Table: {table_name}

                Schema:
                {schema}

                Convert the following natural language request into a syntactically correct SQL query:
                {nl_query}

                Only return the SQL query, no explanations.
                

                NOTE: You are working with a Microsoft SQL Server database.
                      Avoid using `strftime()`. Instead, use `YEAR()` and `MONTH()` to extract year and month from a datetime field.
                """
            return PromptTemplate(template=template.strip(), input_variables=["table_name","schema","nl_query"])
        except Exception as e:
            return e
        

# This class integrates database interaction and LLM-based SQL query generation
class QueryTable(DatabaseOperations, ChatGoogleGENAI):

    def __init__(self, database_name: str, select_server: str):
        # Initialize the parent class for database operations
        DatabaseOperations.__init__(self, db_name=database_name, server=select_server)
        
        # Initialize the LLM interface (e.g., Google Gemini via ChatGoogleGENAI)
        ChatGoogleGENAI.__init__(self)

    def get_sql_query(self, table_name: list, user_query: str):
        """
        Constructs a prompt using the provided table names and user natural language query.
        Sends it to the LLM and returns the generated SQL query.
        """
        try:
            # Get schema information of the provided tables
            table_schema = self.get_multiple_table_schemas(table_name)

            # Retrieve prompt template and fill it with actual table and query details
            prompt_template = PromptTemplates.query_table()
            prompt = prompt_template.format(
                table_name=table_name,
                schema=table_schema,
                nl_query=user_query
            )

            # Invoke the LLM with the prompt and return the response
            return self.llm.invoke(prompt)
        except Exception as e:
            return e  # Return any encountered exception

    def extract_sql_from_response(self, response):
        """
        Extracts raw SQL query string from LLM response. Handles both plain and markdown formatted SQL.
        """
        try:
            # If response has `.content` attribute (e.g., from Gemini), use it
            if hasattr(response, "content"):
                sql = response.content.strip()
            else:
                sql = str(response).strip()

            # Clean markdown formatting like ```sql and ```
            if sql.startswith("```sql"):
                sql = sql.replace("```sql", "").strip()
            if sql.endswith("```"):
                sql = sql[:-3].strip()

            return sql
        except Exception as e:
            return  # Silently fail if anything goes wrong (can log this for better debug)

    def execuete_query(self, table: str, query: str):
        """
        Main function to process a user query:
        - Generates SQL from natural language using LLM
        - Executes the SQL on the database
        - Returns result rows and column names
        """
        try:
            # Generate SQL query from the user's prompt
            response = self.get_sql_query(table_name=table, user_query=query)

            # Extract clean SQL from the LLM response
            sql_to_run = self.extract_sql_from_response(response)

            # Execute the extracted SQL
            self.cursor.execute(sql_to_run)

            # Extract column names and result rows
            columns = [desc[0] for desc in self.cursor.description]
            result = self.cursor.fetchall()

            return result, columns
        except Exception as e:
            return e  # Return error if execution fails

    
        
if __name__ == "__main__":
    q = QueryTable(database_name='pizzasales', select_server='LAPTOP-B17JMI03\\SQLEXPRESS')
    query, table_columns = q.execuete_query(table=["pizzas","pizza_types",'orders',"order_details"],
                    query="display the categories and the sum of quantity accross all the categories")

    
    print(table_columns)
    df = read_as_dataframe(query, feature_list=table_columns)
    print(df)