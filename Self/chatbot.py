import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
load_dotenv() 

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
#os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

## model

model = init_chat_model(model="openai/gpt-oss-120b", model_provider="Groq")

##chatbot
from langchain.messages import HumanMessage, SystemMessage, AIMessage
conversions = [
    SystemMessage(content="You are a helpfull chatbot assistant. Please respond concisely and to-the-point")
]
while True:
    user_input = input("User: ")
    if user_input.lower() in ["bye","exit"]:
        print("Goodbye!")
        exit()
    conversions.append(HumanMessage(content=user_input))
    response = model.invoke(conversions)
    conversions.append(AIMessage(content=response.content))
    print("AI:", response.content)