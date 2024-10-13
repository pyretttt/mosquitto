from llama_cpp import Llama
from pprint import pprint


llm = Llama(
    model_path="no_index//Phi-3.5-mini-instruct-q5_k_m.gguf",
    chat_template="chat_template.default"
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
    return response["choices"][0]["message"]

if __name__ == "__main__":
    while True:
        user_input = input("(Type exit to quit) User: ")
        if user_input.strip() == "exit":
            break
        
        add_user_message(user_input)
        llm_output = llm.create_chat_completion(
            messages=messages_history,
            max_tokens=128
        )
        messages_history.append(
            get_message_from_response(llm_output)
        )

        print_message_history()
print("Goodbye!")