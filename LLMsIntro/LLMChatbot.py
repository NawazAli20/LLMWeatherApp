import os 
from dotenv import load_dotenv
load_dotenv() 
from langchain.chat_models import init_chat_model
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

model=init_chat_model(model="openai/gpt-oss-120b", model_provider="Groq")


## Chatbot 
from langchain.messages import HumanMessage, AIMessage, SystemMessage
conversations = [
    SystemMessage(content="You are an helpful chatbot assistant. Please answer briefly and to the point.")
]
while True:
    user_input = input("User: ")
    if user_input.lower() in ["bye","exit"]:
        print("Goodbye")
        exit()
    conversations.append(HumanMessage(content=user_input))
    response = model.invoke(conversations)
    conversations.append(AIMessage(content=response.content))
    print("AI:", response.content)