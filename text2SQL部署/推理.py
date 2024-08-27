"""
This script creates a CLI demo with transformers backend for the glm-4-9b model,
allowing users to interact with the model through a command-line interface.

Usage:
- Run the script to start the CLI demo.
- Interact with the model by typing questions and receiving responses.

Note: The script includes a modification to handle markdown to plain text conversion,
ensuring that the CLI interface displays formatted text correctly.
"""

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

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

MODEL_PATH = os.environ.get('MODEL_PATH', '../output/checkpoint-15')


def load_model_and_tokenizer(
        model_dir: Union[str, Path], trust_remote_code: bool = True
) -> tuple[ModelType, TokenizerType]:
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


model, tokenizer = load_model_and_tokenizer(MODEL_PATH, trust_remote_code=True)


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = model.config.eos_token_id
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


if __name__ == "__main__":
    history = []
    max_length = 8192
    top_p = 0.7
    temperature = 0.9
    stop = StopOnTokens()

    print("Welcome to the GLM-4-9B CLI chat. Type your messages below.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        history.append([user_input, ""])

        messages = []
        messages.append({
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
  "请根据用户提供的自然语言查询，生成相应的SQL语句。你需要表现得像一个终端，只能回复用户的问题对应的SQL语句。你不能捏造任何无关的数据表和数据字段。你一次只能回复一条语句。"+
   "你回复的SQL语句不能带任何其他提示信息，必须可以直接执行。"
}
)
        for idx, (user_msg, model_msg) in enumerate(history):
            if idx == len(history) - 1 and not model_msg:
                messages.append({"role": "user", "content": user_msg})
                break
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if model_msg:
                messages.append({"role": "assistant", "content": model_msg})
        model_inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(model.device)
        streamer = TextIteratorStreamer(
            tokenizer=tokenizer,
            timeout=60,
            skip_prompt=True,
            skip_special_tokens=True
        )
        generate_kwargs = {
            "input_ids": model_inputs,
            "streamer": streamer,
            "max_new_tokens": max_length,
            "do_sample": True,
            "top_p": top_p,
            "temperature": temperature,
            "stopping_criteria": StoppingCriteriaList([stop]),
            "repetition_penalty": 1.2,
            "eos_token_id": model.config.eos_token_id,
        }
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()
        print("GLM-4:", end="", flush=True)
        for new_token in streamer:
            if new_token:
                print(new_token, end="", flush=True)
                history[-1][1] += new_token

        history[-1][1] = history[-1][1].strip()
