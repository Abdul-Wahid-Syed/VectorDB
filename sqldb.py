from langchain import PromptTemplate,LLMChain,SQLDatabase
from langchain.llms.bedrock import Bedrock

llm = Bedrock(
    model_id="anthropic.claude-v2",model_kwargs = {"temperature":1e-10,"max_tokens_to_sample": 8191}
)

db_user = ""
db_password = ""
db_host = ""
db_name = ""
db = SQLDatabase.from_uri(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}/{db_name}")

schema = db.get_table_info(table_names=['insurance'])

template = """Human:You are a PostgreSQL expert. Given an input question, first create a syntactically correct PostgreSQL query to run.
Unless the user specifies in the question a specific number of examples to obtain, query for at most results using the LIMIT clause as per PostgreSQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (").
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use CURRENT_DATE function to get the current date, if the question involves "today".

Question: {input}
Only use the following tables:
'insurance'

Schema of the table:{table_info}

SQL Query: {output} only the query
Assistant:
"""

PROMPT = PromptTemplate(
    input_variables=["input","output","table_info"], template=template
)

db_chain = LLMChain(llm=llm,prompt=PROMPT)

template2 = """Human:
Given an input question and data ,answer the question based on the data
Instruction:
1.Review the Question and data(output of sqlquery from database) carefully
2.Answer the question based on the data in a meaningful sentence,nothing other than that

Question:{input}
Data:{data}
Output Sentence:{output}
Assistant:
"""

prompt2 = PromptTemplate(input_variables=["input","data","output"], template=template2)

llm_chain2 = LLMChain(prompt=prompt2, llm=llm)

question = input("type the question")
def sql_run(question):
    a = db_chain.run({"input":question,"output":"","table_info":schema})
    b = db.run(a)
    response = llm_chain2.run({"input":question,"data":b,"output":""})
    print(response)

sql_run(question)