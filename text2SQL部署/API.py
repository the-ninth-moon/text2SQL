from fastapi import FastAPI, Request
import uvicorn
import json
import datetime
import torch
import os
import torch
from threading import Thread
from typing import Union
from pathlib import Path
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer
)


# 设置设备参数
DEVICE = "cuda"  # 使用CUDA
DEVICE_ID = "0"  # CUDA设备ID，如果未设置则为空
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # 组合CUDA设备信息

# 清理GPU内存函数
def torch_gc():
    if torch.cuda.is_available():  # 检查是否可用CUDA
        with torch.cuda.device(CUDA_DEVICE):  # 指定CUDA设备
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片

# 创建FastAPI应用
app = FastAPI()

# 处理POST请求的端点
@app.post("/")
async def create_item(request: Request):
    global model, tokenizer  # 声明全局变量以便在函数内部使用模型和分词器
    json_post_raw = await request.json()  # 获取POST请求的JSON数据
    json_post = json.dumps(json_post_raw)  # 将JSON数据转换为字符串
    json_post_list = json.loads(json_post)  # 将字符串转换为Python对象
    prompt = json_post_list.get('prompt')  # 获取请求中的提示
    history = json_post_list.get('history')  # 获取请求中的历史记录
    history.append({
  "role": "system",
  "content": "你是一位精通SQL和数据库管理的AI助手。你的任务是将用户提供的自然语言查询转换为相应的SQL查询。以下是数据库的建表信息：\n\n" +
  "CREATE TABLE `data_info_by_city` (\n" +
  "    `city` varchar(80) DEFAULT NULL COMMENT '城市',\n" +
  "    `area` varchar(80) DEFAULT NULL COMMENT '区域',\n" +
  "    `province` varchar(80) DEFAULT NULL COMMENT '省份',\n" +
  "    `cumulative_charging_swapping_times` bigint DEFAULT NULL COMMENT '累计充换电次数',\n" +
  "    `cumulative_charging_times` bigint DEFAULT NULL COMMENT '累计充电次数',\n" +
  "    `cumulative_swapping_times` bigint DEFAULT NULL COMMENT '累计换电次数',\n" +
  "    `daily_real_time_charging_swapping_times` bigint DEFAULT NULL COMMENT '今天实时充换电次数',\n" +
  "    `daily_real_time_charging_times` bigint DEFAULT NULL COMMENT '今天实时充电次数',\n" +
  "    `daily_real_time_swapping_times` bigint DEFAULT NULL COMMENT '今天实时换电次数',\n" +
  "    `battery_safety_rate`  bool DEFAULT NULL COMMENT '电池安全预警占比',\n" +
  "    `battery_non_safety_rate`  bool DEFAULT NULL COMMENT '电池非安全/风险/异常预警占比',\n" +
  "    `reduce_total_mileage` varchar(80) default null comment '节省总里程',\n" +
  "    `reduce_carbon_emissions` varchar(80) default null comment '减少碳排放',\n" +
  "    `battery_riding_mileage` bigint default null comment '电池当天行驶里程',\n" +
  "    `battery_non_safety_times` bigint default null comment '电池当天安全预警次数'\n" +
  ");\n\n" +
  "CREATE TABLE `user_rider_info` (\n" +
  "    `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,\n" +
  "    `city` varchar(80) DEFAULT NULL COMMENT '中文城市名称,如杭州市、北京市',\n" +
  "    `area` varchar(80) DEFAULT NULL COMMENT '中文区域名称，如余杭区、西湖区',\n" +
  "    `province` varchar(80) DEFAULT NULL COMMENT '中文省份名称，如浙江省、江苏省',\n" +
  "    `gender` bigint default null comment '性别 0表示男性，1表示女/女性/女生',\n" +
  "    `age` bigint default null comment '年龄',\n" +
  "    `hd_count` bigint default null comment '换电次数',\n" +
  "    `daily_hd_count` bigint default null comment '每天换电次数'\n" +
  ") comment '骑手表';\n\n" +
  "CREATE TABLE `battery` (\n" +
  "    `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,\n" +
  "    `city` varchar(80) DEFAULT NULL COMMENT '中文城市名称,如杭州市、北京市',\n" +
  "    `area` varchar(80) DEFAULT NULL COMMENT '中文区域名称，如余杭区、西湖区',\n" +
  "    `province` varchar(80) DEFAULT NULL COMMENT '中文省份名称，如浙江省、江苏省',\n" +
  "    `is_qishou` bool DEFAULT NULL COMMENT '0表示不在骑手中, 1表示在骑手中',\n" +
  "    `is_guizi`  bool DEFAULT NULL COMMENT '0表示不在电柜中, 1表示在电柜中',\n" +
  "    `type` bigint default null comment '1表示48伏/48V类型电池, 2表示60伏/60V类型电池, 3表示48MAX类型电池, 4表示60MAX类型电池',\n" +
  "    `status` bigint DEFAULT NULL COMMENT '0代表电池使用中, 1代表电池充电中, 2代表电池已满电'\n" +
  ") comment '电池表';\n\n" +
  "CREATE TABLE `hdg_info` (\n" +
  "    `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,\n" +
  "    `city` varchar(80) DEFAULT NULL COMMENT '中文城市名称,如杭州市、北京市',\n" +
  "    `area` varchar(80) DEFAULT NULL COMMENT '中文区域名称，如余杭区、西湖区',\n" +
  "    `province` varchar(80) DEFAULT NULL COMMENT '中文省份名称，如浙江省、黑龙江省'\n" +
  ") comment '电柜表';\n\n" +
  "请根据用户提供的自然语言，生成相应的SQL语句。你需要表现得像一个终端，只能回复用户的问题对应的SQL语句。你不能捏造任何无关的数据表和数据字段。你一次只能回复一条语句。"+
   "你回复的SQL语句不能带任何其他提示信息，必须可以直接执行。"
})
    max_length = json_post_list.get('max_length')  # 获取请求中的最大长度
    top_p = json_post_list.get('top_p')  # 获取请求中的top_p参数
    temperature = json_post_list.get('temperature')  # 获取请求中的温度参数
    # 调用模型进行对话生成
    response, history = model.chat(
        tokenizer,
        prompt,
        history=history,
        max_length=max_length if max_length else 2048,  # 如果未提供最大长度，默认使用2048
        top_p=top_p if top_p else 0.7,  # 如果未提供top_p参数，默认使用0.7
        temperature=temperature if temperature else 0.95  # 如果未提供温度参数，默认使用0.95
    )
    now = datetime.datetime.now()  # 获取当前时间
    time = now.strftime("%Y-%m-%d %H:%M:%S")  # 格式化时间为字符串
    # 构建响应JSON
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    # 构建日志信息
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)  # 打印日志
    torch_gc()  # 执行GPU内存清理
    return answer  # 返回响应

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

MODEL_PATH = os.environ.get('MODEL_PATH', '。。/output/checkpoint-15')
print("----------------------",MODEL_PATH,"--------------------")


def load_model_and_tokenizer(
        model_dir: Union[str, Path], trust_remote_code: bool = True
):
    model_dir = Path(model_dir).expanduser().resolve()
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto')
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=trust_remote_code, device_map='auto')
        tokenizer_dir = model_dir

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=trust_remote_code, encode_special_tokens=True, use_fast=False
    )
    return model, tokenizer



# 主函数入口
if __name__ == '__main__':
    # 加载训练的分词器和模型
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, trust_remote_code=True)
    model.eval()  # 设置模型为评估模式
    # 启动FastAPI应用
    # 用6006端口可以将autodl的端口映射到本地，从而在本地使用api
    uvicorn.run(app, host='0.0.0.0', port=6006, workers=1)  # 在指定端口和主机上启动应用