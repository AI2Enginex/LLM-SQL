import os
import pymysql
import warnings
import sqlalchemy
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import google.generativeai as genai  
warnings.filterwarnings('ignore')

# Setting the API key for Google Generative AI service by assigning it to the environment variable 'GOOGLE_API_KEY'
api_key = os.environ['GOOGLE_API_KEY'] = "AIzaSyDaHofSA0rPEv28pznJZ6vhbJh0W9uU4oM"

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
            return e
    
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
            return e
    

    # Method to read Schemas for multiple tables
    def get_multiple_table_schemas(self, table_names: list):
        try:
            all_schemas = []
            for table in table_names:
                schema = self.get_table_schema(table)
                all_schemas.append(schema)
            return "\n\n".join(all_schemas)
        except Exception as e:
            return e


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

                Only return the SQL query, no explanations. Display your answer in tablusr format.
                """
            return PromptTemplate(template=template.strip(), input_variables=["table_name","schema","nl_query"])
        except Exception as e:
            return e

class QueryTable(DatabaseConnect, ChatGoogleGENAI):

    def __init__(self, password: str, database: str, server: str):
        DatabaseConnect.__init__(self,password, database, server)
        ChatGoogleGENAI.__init__(self)

    def get_sql_query(self,table_name: list, user_query: str):

        try:
            table_schema = self.get_multiple_table_schemas(table_name)
            prompt_template = PromptTemplates.query_table()
            prompt = prompt_template.format(table_name=table_name, schema=table_schema,nl_query=user_query)
            return self.llm.invoke(prompt)
            
        except Exception as e:
            return e
    
    def extract_sql_from_response(self,response):
        try:
            if hasattr(response, "content"):
                sql = response.content.strip()
            else:
                sql = str(response).strip()

            # Remove markdown ```sql and ``` blocks if present
            if sql.startswith("```sql"):
                sql = sql.replace("```sql", "").strip()
            if sql.endswith("```"):
                sql = sql[:-3].strip()

            return sql
        except Exception as e:
            return

    def execuete_query(self,table: str, query: str):

        try:
            response = self.get_sql_query(table_name=table, user_query=query)
            con = self.try_connection()
            con.execute(self.extract_sql_from_response(response))
            columns = [desc[0] for desc in con.description]
            result = con.fetchall()
            return result, columns
        except Exception as e:
            return e
        
if __name__ == "__main__":

    q = QueryTable(password="vasu",database="employees",server='localhost')
    query, columns = q.execuete_query(table=["employees","employee_projects","projects"],
                         query="display the entire data from the employees table")
    
    print(query)
    llm_answer = read_as_dataframe(data=query, feature_list=columns)
    print(llm_answer)