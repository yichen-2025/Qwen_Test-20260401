

def process_raw_datasets(file_name):
    """
    处理原始数据集
    params:
        file_name: 原始数据集文件名
    """
    from preprocessing import preprocess
    from preprocessing import split_train_test
    from preprocessing import to_json

    df=preprocess(file_name)
    train_df,test_df=split_train_test(df)
    to_json(train_df,file_name.replace(".csv",".json"))

def test_model(model_file_path,test_file_name_or_path):
    """
    测试模型
    params:
        model_file_path: 模型文件路径
        test_file_name_or_path: 测试集文件名或路径
    """

    from system import TestSystem
    system=TestSystem(
        model_file_path=model_file_path,
        test_file_name_or_path=test_file_name_or_path,
    )

    system.execute()

def eval_result(result_file_path):
    """
    评估模型结果
    params:
        result_file_path: 模型结果文件路径
    """
    from assessment import Assessment

    assessment=Assessment(result_file_path)
    assessment.save_metrics()


if __name__ == "__main__":
    # 第一步：处理原始数据集
    # process_raw_datasets(原始数据集路径)
    process_raw_datasets("raw_datasets/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    
    # 第二步：测试模型
    # from preprocessing import shrink
    # 用shrink函数缩小测试集大小
    # test_model(模型路径,测试集路径)
    test_model("Qwen/qwen2.5-1.5b-instruct","test/test 20260401_153000.csv")    # 假设测试集文件名是test 20260401_153000.csv

    # 第三步：评估模型结果
    # eval_result(模型结果文件路径)
    eval_result("results/results 20260401_153000.csv")      # 假设结果文件名是results 20260401_153000.csv
   
    # 第四步：在results目录下查看评估结果
    # 查看results目录下的metrics.csv文件
    # 评估结果包括准确率、精确率、召回率、F1值等
