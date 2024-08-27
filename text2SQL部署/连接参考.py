import requests
import json

#由于腾讯云服务的设置，每次的服务器启动后host的地址是不一样的
host = "43.133.241.37" #修改这一行为正确IP

def send_message(prompt, history):
    url = "http://{}:6006/".format(host)
    headers = {'Content-Type': 'application/json'}
    data = {
        "prompt": prompt,
        "history": history
    }
    #print(data)
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: Received status code {response.status_code}")
        return None

def main():
    # 初始化历史记录并添加系统消息
    history = []
    
    while True:
        user_input = input("自然语言: ")
        if user_input.lower() == 'quit':
            #print("Exiting conversation.")
            break
        response_data = send_message(user_input, history)
        if response_data:
            print("SQL语句:", response_data['response'])
            print(history)
            # # 维护历史记录，包括用户输入和AI响应
            # history.append({"role": "user", "content": user_input})
            # history.append({"role": "assistant", "content": response_data['response']})

if __name__ == "__main__":
    main()
