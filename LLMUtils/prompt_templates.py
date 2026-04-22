
from langchain_core.prompts import PromptTemplate
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
            raise e
        
    
    @classmethod
    def query_table_(cls):
        
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
            raise e