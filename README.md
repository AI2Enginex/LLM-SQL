# 🔍 Querying Database Using LLM (Gemini)

This project demonstrates how to query relational databases using natural language via Google Gemini LLM, enabling non-technical users to extract insights from MySQL and SQL Server without writing SQL code.


# 🚀 Features

✅ Accepts natural language queries from users.

🤖 Translates them into valid SQL queries using Gemini (Google Generative AI).

# 🔗 Executes the generated SQL against:

**MySQL databases.**

**SQL SERVER databases.**

# Microsoft SQL Server databases.

📊 Converts the results into clean pandas DataFrames for easy manipulation or visualization.

🔐 Secure handling of API keys and database credentials.

🧠 Modular and extensible design — supports schema inspection, multiple tables, and advanced query generation.


# 🧠 Tech Stack

Google Gemini (Generative AI)

LangChain (for prompt and model abstraction)

pandas (for data representation)

pyodbc (for SQL Server connection)

mysql-connector-python (for MySQL connection)


# 🏗️ How It Works

1. User Inputs Natural Query

    Example: "Show total sales per month in 2024 for breakfast items"

2. LLM Prompting

    Gemini receives the table schema + user query and returns the corresponding SQL query.

3. SQL Execution

    The app executes the query on the selected database (MySQL or SQL Server).

4. Result Conversion

    The results are returned as a list of tuples, then converted into a readable DataFrame.


# 🧪 Example Queries


| Natural Language Query                   | Description                           |
| ---------------------------------------- | ------------------------------------- |
| *"Show average salary by department."*   | Aggregates data by department         |
| *"List top 5 most ordered pizza types."* | Limits, sorts, joins                  |
| *"What are the monthly sales in 2023?"*  | Uses `YEAR()` and `MONTH()` functions |


# ⚙️ Supported Databases

    ✅ MySQL (via mysql.connector)

    ✅ SQL Server (via pyodbc)

# 📌 Notes

    1. For SQL Server, ensure you have the appropriate ODBC driver installed.

    2. For MySQL, ensure the MySQL service is running and reachable.
