#%%
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate

from schemas import AgentResponse

#%%

tools = [TavilySearch(max_results=3)]
llm = ChatOpenAI(model="hf.co/MaziyarPanahi/Mistral-Nemo-Instruct-2407-GGUF:Q4_K_M", base_url="http://192.168.1.93:11434/v1", api_key="ollama")

agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
)



#%% 

def main():

    print("Hello from LangChain course on search agents !")

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "search the web for instructions on long-term memory stratgies for LLMs",
                }
            ]
        }
    )
    # Access structured response from the agent
    structured = result.get("structured_response", None)
    print(structured if structured is not None else result)

if __name__ == "__main__":
    main()

# %%
