from llama_cpp import Llama
from openai import OpenAI, ChatCompletion

from pprint import pprint

client = OpenAI(
    base_url="http://localhost:8080/",
    api_key="local-no-key-required"
)

messages_history=[
    {
        "role": "system",
        "content": "You are a rural worker"
    },
]

def add_user_message(message: str):
    messages_history.append(
        {
            "role": "user",
            "content": message
        }
    )

def print_message_history():
    print("=======================")
    for message in messages_history:
        print(message["role"], ":", message["content"])

def get_message_from_response(response: dict):
    chat_message = response.choices[0].message
    response = {
        "role": chat_message.role,
        "content": chat_message.content
    }
    return response

if __name__ == "__main__":
    while True:
        user_input = input("(Type exit to quit) User: ")
        if user_input.strip() == "exit":
            break
        
        add_user_message(user_input)
        llm_output = client.chat.completions.create(
            model="no_index/Phi-3.5-mini-instruct-q5_k_m.gguf",
            messages_history=messages_history
        )
        messages_history.append(
            get_message_from_response(llm_output)
        )

        print_message_history()
print("Goodbye!")