import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(),logging.FileHandler("system.log",encoding="utf-8")]
    )

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.info("开始导入第三方库")

from transformers import AutoTokenizer,AutoModelForCausalLM

logging.info("第三方库导入完成")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# 加载分词器（Tokenizer）
logging.info("开始加载tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
logging.info("tokenizer加载完成")

# 加载模型（Model）
logging.info("开始加载model")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
logging.info("model加载完成")


SAVE_PATH = "Qwen/qwen2.5-1.5b-instruct"

# 保存分词器（Tokenizer）
logging.info("保存tokenizer")
tokenizer.save_pretrained(SAVE_PATH)

# 保存模型（Model）
logging.info("保存model")
model.save_pretrained(SAVE_PATH)


logging.info(f"模型已保存到{SAVE_PATH}")
