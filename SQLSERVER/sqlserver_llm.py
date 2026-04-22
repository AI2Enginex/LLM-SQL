import pandas as pd
from SQLSERVER.connection import DatabaseOperations
from LLMUtils.configs import ChatGoogleGENAI
from LLMUtils.prompt_templates import PromptTemplates
import warnings
warnings.filterwarnings('ignore')

def read_as_dataframe(data: None, feature_list: list):

    try:
        
        rows = [list(value) for value in data]
        return pd.DataFrame(rows, columns=feature_list)
    except Exception as e:
        raise e
    
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
            prompt_template = PromptTemplates.query_table_()
            prompt = prompt_template.format(
                table_name=table_name,
                schema=table_schema,
                nl_query=user_query
            )

            # Invoke the LLM with the prompt and return the response

            model_response = self.llm.invoke(prompt)
            
            text_value = model_response.content[0].get('text', '').replace('\n', ' ').strip()
            print("Generated SQL Query:", text_value)  # Optional: Print the generated SQL for debugging
            return text_value
            
        except Exception as e:
            raise e  # Return any encountered exception

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
            raise e  # raise error if execution fails

    
        
if __name__ == "__main__":
    q = QueryTable(database_name='Vibhor', select_server='ULTRON\SQLEXPRESS')
    query, table_columns = q.execuete_query(table=["pizzas","pizza_types",'orders',"order_details"],
                    query="display the total quantity sold on 27th january 2015 for each pizza category ")

    
    print(table_columns)
    df = read_as_dataframe(query, feature_list=table_columns)
    print(df)