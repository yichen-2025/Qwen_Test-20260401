import torch
import logging
import os
from datetime import datetime
import pandas as pd

# 使用镜像网站，不然会很慢
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 禁用符号链接警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(),logging.FileHandler("system.log",encoding="utf-8")]
    )

# 导入transformers前需要设置环境变量↑
# 导入transformers库需要一段时间
logging.info("开始导入第三方库")
begin=datetime.now()

from transformers import AutoTokenizer,AutoModelForCausalLM

end=datetime.now()
logging.info(f"导入第三方库耗时: {end-begin}")


class TestSystem:

    LABEL_NAME="output"     # 标签的名字

    # 系统提示词
    SYSTEM_PROMPT="""
    你是一个专业的网络安全专家，擅长识别DDoS等恶意流量。请根据以下网络流量特征，判断该流量是正常流量还是恶意流量。
    判断标准：
    恶意流量特征：
    - 平均包大小小于10字节或大于1000字节
    - 包长度标准差大于1000
    - 后向包比例大于0.6或小于0.1
    - 流间隔时间标准差大于5000000

    正常流量特征：
    - 平均包大小在10字节到1000字节之间
    - 包长度标准差在0到1000之间
    - 后向包比例在0.1到0.6之间
    - 流间隔时间标准差在0到5000000之间

    示例：
    1.平均包大小为1163.3字节。包长度均值为1057.55字节。包长度标准差为1853.44字节。平均前向段大小为8.67字节。平均后向段大小为1658.14字节。后向包长度均值为1658.14字节。后向包长度标准差为2137.3字节。后向包比例为0.0。SYN包比例为0.0。流间隔时间标准差为430865.8秒。回答：恶意流量
    2.平均包大小为34字节。包长度均值为22.67字节。包长度标准差为14.43字节。平均前向段大小为18.5字节。平均后向段大小为0.0字节。后向包长度均值为0.0字节。后向包长度标准差为0.0字节。后向包比例为0.0。SYN包比例为0.5。流间隔时间标准差为0秒。回答：正常流量
    只回答'正常流量'或'恶意流量'，严禁添加任何解释。
    """

    # 把模型回复的文字映射为0或1
    @staticmethod
    def response_map(x):
        return 0 if "正常" in x else 1 if "恶意" in x else -1
        
    
    def __init__(self,model_file_path,test_file_name_or_path,system_prompt=SYSTEM_PROMPT,label_name=LABEL_NAME):
        """
        params:
            model_file_path: 模型文件路径
            test_file_name_or_path: 测试数据集文件名或路径
            system_prompt: 系统提示词
            label_name: 标签的名字
        """
        self.SYSTEM_PROMPT=system_prompt
        self.LABEL_NAME=label_name
        self.dataset=pd.read_csv(test_file_name_or_path)
        self.result={
            'label':[],
            'response':[],
            'model_response':[],
            'time':[]
        }

        for label in self.dataset[self.LABEL_NAME]:
            self.result['label'].append(self.response_map(label))
        
        logging.info(f"开始加载模型: {model_file_path.split('/')[-1]}")

        self.tokenizer=AutoTokenizer.from_pretrained(model_file_path)
        self.model=AutoModelForCausalLM.from_pretrained(model_file_path)
        
        self.is_cuda=torch.cuda.is_available()  # 检查是否有可用的GPU
        self.device=torch.device("cuda" if self.is_cuda else "cpu")
        self.model=self.model.to(self.device)   # 将模型移动到指定设备

        logging.info(f"模型加载完成: {model_file_path.split('/')[-1]}")
        logging.info(f"模型设备: {self.device}")
        if not self.is_cuda:
            res=input("当前环境不支持GPU，是否继续？(y/n)")
            if res.lower()!="y":
                logging.info("用户取消预测")
                exit(0)
            logging.info("用户确认预测，继续执行")
        
    def send_prompt(self,prompt):
        """
        发送提示词到模型并获取回复
        params:
            prompt: 输入的提示词
        return:
            模型回复
        """
        
        # 把提示词发送给大模型
        messages=[
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        )

        if self.is_cuda:
            inputs=inputs.to(self.device)

        with torch.no_grad():   # 禁用梯度计算，节省内存和计算资源
            outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 对模型的回复进行清理，提取出最后的回复
        if "assistant" in response:
            response=response.split("assistant")[-1].strip()
        
        response=response.replace("<|im_end|>","").replace("<|im_start|>","").strip()

        logging.info(f"模型清理后的回复: {response}")
        
        return response

    def predict(self):
        """
        预测测试数据集中的每个样本
        return:
            预测结果
        """
        self.result['response']=[]
        self.result['model_response']=[]
        self.result['time']=[]


        for i,prompt in enumerate(self.dataset['input']):
            logging.info(f"开始处理第{i}条数据")
            begin=datetime.now()
            model_response=self.send_prompt(prompt)
            self.result['model_response'].append(model_response)
            self.result['response'].append(self.response_map(model_response))
            end=datetime.now()
            logging.info(f"第{i}条数据处理耗时: {end-begin}")
            self.result['time'].append((end-begin).total_seconds())

        logging.info("所有数据处理完成")
        return pd.DataFrame(self.result)


    def execute(self):
        """
        执行测试
        return:
            预测结果
        """
        results=self.predict()

        file_path=f"results/result {datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        logging.info(f"预测结果已保存至: {file_path}")
        results.to_csv(file_path,index=False)
        
        return results
