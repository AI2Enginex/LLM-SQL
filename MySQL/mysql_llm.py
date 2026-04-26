import warnings
import pandas as pd
from MySQL.connection import DatabaseConnect
from LLMUtils.prompt_templates import PromptTemplates
from LLMUtils.configs import ChatGoogleGENAI
warnings.filterwarnings('ignore')


def read_as_dataframe(data: None, feature_list: list):

    try:
        
        rows = [list(value) for value in data]
        return pd.DataFrame(rows, columns=feature_list)
    except Exception as e:
        raise e


class QueryTable(DatabaseConnect, ChatGoogleGENAI):

    def __init__(self, user: str, password: str, database: str, server: str):
        DatabaseConnect.__init__(self, user, password, database, server)
        ChatGoogleGENAI.__init__(self)

    def get_sql_query(self,table_name: list, user_query: str):

        try:
            table_schema = self.get_multiple_table_schemas(table_name)
            prompt_template = PromptTemplates.query_table()
            prompt = prompt_template.format(table_name=table_name, schema=table_schema,nl_query=user_query)

            model_response = self.llm.invoke(prompt)
            
            text_value = model_response.content[0].get('text', '').replace('\n', ' ').strip()
           
            return text_value
            
        except Exception as e:
            raise e
    
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
            raise e
        
if __name__ == "__main__":

    q = QueryTable(user="root", password="vasu", database="employees", server='localhost')
    query, columns = q.execuete_query(table=["employees","employee_projects","projects"],
                         query="display the names of the employees and projects assigned to them")
    
    print(query)
    llm_answer = read_as_dataframe(data=query, feature_list=columns)
    print(llm_answer)