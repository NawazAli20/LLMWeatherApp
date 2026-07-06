import os

from dotenv import load_dotenv

load_dotenv()
#os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY =os.getenv("OPENWEATHER_API_KEY")

#Get the llm model
from langchain.chat_models import init_chat_model

#llm = init_chat_model(model="llama-3.1-8b-instant", model_provider="Groq", temperature=0.8, max_tokens=1000)
llm = init_chat_model(model="gpt-5.4-mini", model_provider="OpenAI", temperature=0.8, max_tokens=1000)
test_message = llm.invoke("hi").content
print(test_message)


import requests
# getWeather tool 
from langchain.tools import tool

url = "https://api.openweathermap.org/data/2.5/weather?"

@tool
def getWeather(location:str,zipcode:str)->str:
    """Returns weather information based on the city and/or zipcode"""
    params={
        "zip":zipcode,
        "q":location,
        "appid":OPENWEATHER_API_KEY,
        "units":"imperial" #imperial
    }
    try:
        response = requests.get(url,params=params,timeout=10)
        return response.json()
    except:
        return f"Does not have the weather information"

## Message formatting 
from langchain.messages import AIMessage, HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are an helpfull weather assistant, Please provide the weather information in a concise way using the tool call")
]

user_input = input("What is your query: ")
messages.append(HumanMessage(content=user_input))

## bind the tool with the model and append the AI message 

llm_with_tools = llm.bind_tools([getWeather])

aiMessage = llm_with_tools.invoke(messages)
#print("AIMessage:",aiMessage)
messages.append(aiMessage)

# Append tool call message
for tool_call in aiMessage.tool_calls:
    tool_call_message = getWeather.invoke(tool_call)
    messages.append(tool_call_message)

## Respons with tool calls 
final_response = llm_with_tools.invoke(messages)
print("\n..................")
print(final_response.content)
print("\n..................")

##Response with structured output 

from pydantic import BaseModel, Field 

class FormattedOutput(BaseModel):
    day: str 
    low_temperarure:float
    high_tempature:float
    wind:str

class FiveDaysForcast():
    location:str
    forcast:list[FormattedOutput]

llm_with_structured_op = llm.with_structured_output(FormattedOutput)
#llm_with_structured_op = llm.with_structured_output(FiveDaysForcast)

final_reponse = llm_with_structured_op.invoke(messages)
print("...............")
print(final_reponse)
print("...............")