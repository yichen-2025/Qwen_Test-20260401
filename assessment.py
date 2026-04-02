import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from datetime import datetime

class Assessment:
    def __init__(self,result_file_path):
        """
        params:
            result_file_path: 预测结果文件路径
        """
        self.result=pd.read_csv(result_file_path)
        self.y_test=self.result["label"].values
        self.y_pred=self.result["response"].values

        self.accuracy=accuracy_score(self.y_test,self.y_pred)
        self.precision=precision_score(self.y_test,self.y_pred)
        self.recall=recall_score(self.y_test,self.y_pred)
        self.f1=f1_score(self.y_test,self.y_pred)
        self.cm=confusion_matrix(self.y_test,self.y_pred)

        self.time_mean=self.result["time"].mean()
        self.time_std=self.result["time"].std()

    def __str__(self):
        metrics=self.get_metrics()

        return "\n".join([
            f"{key}:{value}" for key,value in zip(metrics["metric"],metrics["value"])
        ])

    def get_metrics(self):
        """
        获取模型指标（准确率、精确率、召回率、F1值等）
        return:
            模型指标
        """
        metrics_df={
            'metric':['accuracy','precision','recall','f1','time_mean','time_std'],
            'value':[self.accuracy,self.precision,self.recall,self.f1,self.time_mean,self.time_std]
        }
        metrics_df=pd.DataFrame(metrics_df).set_index("metric")
        return metrics_df

    def save_metrics(self):
        """
        保存模型指标到csv文件
        """
        metrics_df=self.get_metrics()
        filename=f"results/metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        metrics_df.to_csv(filename,index=False)
