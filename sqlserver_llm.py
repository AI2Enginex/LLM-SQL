import os
import pandas as pd
import pyodbc
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import warnings
warnings.filterwarnings('ignore')

# Setting the API key for Google Generative AI service by assigning it to the environment variable 'GOOGLE_API_KEY'
api_key = os.environ['GOOGLE_API_KEY'] = "AIzaSyCLPqDNMYnE6GTR6V3X5NbLaQGHpol_d78"

# Configuring Google Generative AI module with the provided API key
genai.configure(api_key=api_key)
key = os.environ.get('GOOGLE_API_KEY')


def read_as_dataframe(data: None, feature_list: list):

    try:
        return pd.DataFrame(data, columns=feature_list)
    except Exception as e:
        return e
    
class GeminiModel:
    def __init__(self):

        # Initializing the GenerativeModel object with the 'gemini-pro' model
        self.model = genai.GenerativeModel('gemini-1.5-flash')
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
        self.llm=ChatGoogleGenerativeAI(temperature=0.7,model="gemini-1.5-flash", google_api_key=key,top_p=1.0,
            top_k=32,
            candidate_count=1,
            max_output_tokens=3000)

class ConnectionString:

    def __init__(self):
        self.conn_str = (
                'DRIVER={SQL Server};' 
                'SERVER=LAPTOP-B17JMI03\\SQLEXPRESS;'
                'DATABASE=pizzasales;'

                )
    
class MakeConnection(ConnectionString):

    def __init__(self):
        super().__init__()

    def cursor_connection(self):

        try:
            connection = pyodbc.connect(self.conn_str,timeout=10)
            print("success")
            return connection
        except Exception as e:
            return e

class DatabaseOperations(MakeConnection):

    def __init__(self):
        super().__init__()
        self.cursor = self.cursor_connection().cursor()
    

    def read_from_table(self,table_name: str):

        try:
            sql_query = self.cursor.execute(f"SELECT * FROM {table_name};")
            # Fetch the data and column names
            data = sql_query.fetchall()
            df = pd.DataFrame([list(data) for data in data], columns=[columns[0] for columns in sql_query.description])
            return df
        except Exception as e:
            return e
    
    def get_table_schema(self, table_name: str):
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
            return str(e)
        
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

                Only return the SQL query, no explanations.
                

                NOTE: You are working with a Microsoft SQL Server database.
                      Avoid using `strftime()`. Instead, use `YEAR()` and `MONTH()` to extract year and month from a datetime field.
                """
            return PromptTemplate(template=template.strip(), input_variables=["table_name","schema","nl_query"])
        except Exception as e:
            return e
        

class QueryTable(DatabaseOperations, ChatGoogleGENAI):

    def __init__(self):
        DatabaseOperations.__init__(self)
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
            self.cursor.execute(self.extract_sql_from_response(response))
            columns = [desc[0] for desc in self.cursor.description]
            result = self.cursor.fetchall()
            final_result = [list(data) for data in result]
            return final_result, columns
        except Exception as e:
            return e
    
        
if __name__ == "__main__":
    q = QueryTable()
    query, table_columns = q.execuete_query(table=["pizzas","pizza_types",'orders',"order_details"],
                    query="display the categories and the sum of quantity accross all the categories")

    
    print(table_columns)
    df = read_as_dataframe(query, feature_list=table_columns)
    print(df)