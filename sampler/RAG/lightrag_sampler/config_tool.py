from itertools import product
import os
from pyDOE import lhs
import json
import pandas as pd
from lightrag.base import QueryParam

def sample_configs_by_lhs(num_samples, config_target_path):
    """基于 LHS 采样数据库配置，并保存到 CSV"""

    configs_info = json.load(open(config_target_path))
    num_params = len(configs_info) # 配置维度
    lhs_sample = lhs(num_params, samples=num_samples)

    all_configs = []
    for i in range(num_samples):
        config = {}
        for j, (key, val) in enumerate(configs_info.items()):
            if val['type'] == 'integer':
                range_width = val['max'] - val['min'] + 1
                config[key] = int(lhs_sample[i][j] * range_width) + val['min']
            elif val['type'] == 'float':
                range_width = val['max'] - val['min']
                config[key] = lhs_sample[i][j] * range_width + val['min']
            elif val['type'] == 'enum':
                possible_values = val['enum_values']
                config[key] = possible_values[int(lhs_sample[i][j] * len(possible_values))]
        all_configs.append(config)
    return all_configs

def evaluate_configs(config_info_path, config_path, fidelity_path):
    all_configs = pd.read_csv(config_path).to_dict()
    configs_info = json.load(open(config_info_path))
    config_num = len(all_configs['llm_model_name'])

    for i in range(config_num):
        query_param = QueryParam()
        init_dict = {}
        llm_dict = {}
        neo4j_dict = {}
        nanodb_dict = {}
        for config_name, config_value_list in all_configs.items():
            if configs_info[config_name]["scope"] == "query":  # 创建参数类设置查询参数
                setattr(query_param, config_name, config_value_list[i])

            elif configs_info[config_name]["scope"] == "init":  # rag初始化参数
                init_dict[config_name] = config_value_list[i]

            elif configs_info[config_name]["scope"] == "llm":  # llm参数（rag初始化参数的一部分）
                real_name = config_name.split(".")[-1]
                llm_dict[real_name] = config_value_list[i]

            elif configs_info[config_name]["scope"] == "nanodb":  # vectordb参数
                nanodb_dict[config_name] = config_value_list[i]

            elif configs_info[config_name]["scope"] == "neo4j":  # neo4j部分关键参数（可能有权限问题）
                real_name = config_name.split(".")[-1]
                neo4j_dict[real_name] = config_value_list[i]
            else:
                print("error config name")
                exit()

        init_dict["llm_model_kwargs"] = {}
        init_dict["llm_model_kwargs"]["options"] = llm_dict
        init_dict["cosine_better_than_threshold"] = nanodb_dict["cosine_threshold"]

        #todo: run rag on this configuration
        # rag init
        # query
        # 统计结果

async def evaluate_configs_async(config_info_path, config_path, fidelity_path):
    """异步评估配置"""
    # 这里可以实现异步评估逻辑
    all_configs = pd.read_csv(config_path).to_dict()
    configs_info = json.load(open(config_info_path))
    config_num = len(all_configs['llm_model_name'])

    for i in range(config_num):
        query_param = QueryParam()
        init_dict = {}
        llm_dict = {}
        neo4j_dict = {}
        nanodb_dict = {}
        for config_name, config_value_list in all_configs.items():
            if configs_info[config_name]["scope"] == "query":  # 创建参数类设置查询参数
                setattr(query_param, config_name, config_value_list[i])

            elif configs_info[config_name]["scope"] == "init":  # rag初始化参数
                init_dict[config_name] = config_value_list[i]

            elif configs_info[config_name]["scope"] == "llm":  # llm参数（rag初始化参数的一部分）
                real_name = config_name.split(".")[-1]
                llm_dict[real_name] = config_value_list[i]

            elif configs_info[config_name]["scope"] == "nanodb":  # vectordb参数
                nanodb_dict[config_name] = config_value_list[i]

            elif configs_info[config_name]["scope"] == "neo4j":  # neo4j部分关键参数（可能有权限问题）
                real_name = config_name.split(".")[-1]
                neo4j_dict[real_name] = config_value_list[i]
            else:
                print("error config name")
                exit()

        init_dict["llm_model_kwargs"] = {}
        init_dict["llm_model_kwargs"]["options"] = llm_dict
        init_dict["cosine_better_than_threshold"] = nanodb_dict["cosine_threshold"]

    



"""
6.9:
1. 更新文档，删除了一些不感兴趣/固定的配置，增加了fidelity，修改了Performance Metrics
2. 按照模板制定了config json描述文件（包含RAG初始化、query_param、neo4j、nanodb、llm_model四部分）
3. 按照模板方法拉丁超立方采样生成了一组配置
4. 按照模板修改了Lightrag neo4j_impl，增加了修改neo4j参数方法（运行cypher修改dbms，未测试）
5. 大致写好了简单的采样方法

todo:

"""













if __name__ == '__main__':
    path = "sampling/configs/LightRAG.json"
    data = sample_configs_by_lhs(4096, path)
    print(data[:10])
    csv_data = pd.DataFrame(data)
    csv_data.to_csv("sampling/LightRAG_LHS.csv", index=False)
    # evaluate_configs(path, "LightRAG_LHS.csv", "fidelity_factors/LightRAG_Fidelity.json")