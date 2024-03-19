import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.sql_database import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
#PROMPT
template = """
Based on the table schema below, write a SQL query that would answer user's question:
{schema}

Question: {question}
SQL Query:
"""
prompt = ChatPromptTemplate.from_template(template)

#DB CONNECTION
db_uri = 'YOUR_DB_URI'
db = SQLDatabase.from_uri(db_uri)
# print(db.run("select * from CRM_JEST"))

#CHAIN TO CONNECT SQL WITH LLM
def get_schema(_): #Reason behind passing the _ parameter in function is it suppose to go in RunnablePassthrough and it requires function with 1 parameter.
    return db.get_table_info()

llm = ChatOpenAI() #Define a language modal

##Creating a Chain it will take a user query and return sql query in string 
sql_chain = (
    RunnablePassthrough.assign(schema = get_schema)
    | prompt
    | llm.bind(stop="\nSQL Result:")
    | StrOutputParser()
)

# print(sql_chain.invoke({"question":"How many tickets have been created over the last 2 months"}))

#Complete Chain 
template = """
Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question:{question}
SQL Query: {query}
SQL Response: {response}
"""
prompt = ChatPromptTemplate.from_template(template)

def run_query(query):
    return db.run(query)

full_chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
        schema = get_schema,
        response = lambda vars: run_query(vars['query'])
        )
        | prompt
        | llm
        | StrOutputParser()
)

print(full_chain.invoke({"question":"How many tickets have been created over the last 6 months"}))