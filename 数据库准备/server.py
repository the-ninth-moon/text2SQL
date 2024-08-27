from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pymysql

app = FastAPI()

# 配置MySQL数据库连接
db_config = {
    "host":"47.98.41.200",
    "user":"root",  # 替换为你的MySQL用户名
    "password":"@text2SQL",  # 替换为你的MySQL密码
    "database":"text2sql",
    "charset":'utf8mb4'
}


# 定义接收的SQL请求体结构
class SQLRequest(BaseModel):
    sql: str


# SQL语句格式化和检查函数
def format_and_validate_sql(sql: str):
    # 去除句首的\n字符
    sql = sql.lstrip('\n')

    # 如果句尾没有;则添加
    if not sql.strip().endswith(';'):
        sql = sql.strip() + ';'

    # 检查SQL语句是否包含DELETE或DROP关键字
    forbidden_keywords = ['DELETE', 'DROP']
    if any(keyword in sql.upper() for keyword in forbidden_keywords):
        raise HTTPException(status_code=400, detail="Forbidden SQL operation detected!")

    return sql


# 处理SQL执行的函数
def execute_sql(sql: str):
    connection = pymysql.connect(**db_config)
    try:
        # 连接MySQL数据库
        with connection.cursor() as cursor:
            cursor.execute(sql)
            result = cursor.fetchall()
        connection.commit()
        connection.close()
    except Exception as e:
        connection.close()
        raise HTTPException(status_code=500, detail=str(e))
    return result


# 创建API接口
@app.post("/execute_sql")
async def execute_sql_api(request: SQLRequest):
    # 格式化和检查SQL语句
    formatted_sql = format_and_validate_sql(request.sql)

    # 执行SQL语句
    result = execute_sql(formatted_sql)

    return {"result": result}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
