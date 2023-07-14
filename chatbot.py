from langchain import OpenAI , LLMMathChain
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent , Tool 
from langchain.chat_models import ChatOpenAI 
from constants import serp_api,openai_key
import os 
import chainlit as cl 

os.environ["OPENAI_API_KEY"]=openai_key
os.environ["SERPAPI_API_KEY"]=serp_api

@cl.langchain_factory(use_async=False)
def load():
    llm=ChatOpenAI(temperature=0,streaming=True)
    llm1=OpenAI(temperature=0,streaming=True)
    search=SerpAPIWrapper() 
    llm_math_chain=LLMMathChain.from_llm(llm=llm,verbose=True)
    
    tools= [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current affairs and information",
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run ,
            description="useful for when you need to perform mathematical operations"
        ),
    ]
    return initialize_agent(
        tools,llm1,agent="chat-zero-shot-react-description",verbose=True
    )
    