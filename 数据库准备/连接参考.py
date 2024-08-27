import requests
import json

# 设置API的URL
url = 'http://47.98.41.200:8000/execute_sql'

# 要发送的SQL语句
sql_query = "SELECT * FROM battery;"  # 请替换为你的实际表名

# 构建POST请求的数据
payload = {
    'sql': sql_query
}

# 设置请求头
headers = {
    'Content-Type': 'application/json'
}

# 发送POST请求
response = requests.post(url, data=json.dumps(payload), headers=headers)

print(response)

# 打印返回结果
if response.status_code == 200:
    print("Success! Here's the result:")
    print(response.json())
else:
    print(f"Failed with status code {response.status_code}:")
    print(response.json())
