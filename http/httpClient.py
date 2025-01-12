import requests

if __name__ == "__main__":
    r = requests.get("http://127.0.0.1:3000/home")
    
    print(r.text)