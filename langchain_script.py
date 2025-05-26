import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Load env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load Excel data
df = pd.read_excel("sap_logs.xlsx")

# LLM setup
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)

# Create agent with dangerous code enabled
agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

# Chat loop
print("Ask your questions (type 'exit' to quit):")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    try:
        result = agent.run(query)
        print("Bot:", result)
    except Exception as e:
        print("Error:", e)
