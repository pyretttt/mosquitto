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
        "content": "You are a good math teacher"
    },
]

def add_user_message(message: str):
    messages_history.append(
        {
            "role": "user",
            "content": message
        }
    )

def add_assistant_message(message: str):
    messages_history.append(
        {
            "role": "assistant",
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
            model="no_index/Qwen2.5-7B-Instruct-q5_k_m.gguf",
            messages=messages_history,
            stream=True,
            stream_options={"include_usage": True}
        )
        full_response = ""
        request_usage = None
        finish_reason = None
        for chunk in llm_output:
            if chunk.usage:
                request_usage = chunk.usage
            if len(chunk.choices) < 1:
                continue
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
            latest_chunk_str = chunk.choices[0].delta.content
            if latest_chunk_str is None:
                continue
            full_response += latest_chunk_str
            print(latest_chunk_str, end='', flush=True)

        print()

        print("Stop reason: ", finish_reason)
        print("Usage: ", request_usage)

        add_assistant_message(full_response)
        print_message_history()
print("Goodbye!")