import os
import pandas as pd
import re

result_path = "sampling/result"
fidelities = {"difficulty": ["easy", "medium", "hard"], "question_type": ["bridge", "comparison"],
              "num": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}
analyze_metrix = ["total_time", "insert_time", "query_time", "precision", "f1_score", "recall"]


def sort_csv_files_by_fidelity(source_dir, fidelity="difficulty"):
    file_pattern = re.compile(r'^(?P<difficulty>[^_]+)_(?P<question_type>[^_]+)_(?P<num>[^_]+)\.json_result\.csv$')
    fidelity_dict = {}
    for filename in os.listdir(source_dir):
        if (match := file_pattern.match(filename)) and filename.endswith('.csv'):
            fidelity_value = match.group(fidelity)

            if fidelity_value in fidelity_dict:
                fidelity_dict[fidelity_value].append(filename)
            else:
                fidelity_dict[fidelity_value] = [filename]

    return fidelity_dict


def merge_csv_files(file_paths):
    """
    合并多个 CSV 文件为一个

    参数:
    file_paths: CSV 文件路径列表
    output_path: 合并后文件的输出路径
    """
    # 读取所有 CSV 文件并存储为 DataFrame 列表
    dfs = []
    for file in file_paths:
        df = pd.read_csv(file)
        dfs.append(df)

    # 合并所有 DataFrame
    merged_df = pd.concat(dfs, ignore_index=True)
    #
    return merged_df


def calculate_statistics(csv_path, columns=None, stats=None):
    """
    计算 CSV 文件中特定列的统计量

    参数:
    csv_path: CSV 文件路径
    columns: 需要计算统计量的列名列表（默认为所有数值列）
    stats: 需要计算的统计量列表（默认为均值、中位数、标准差）

    返回:
    包含统计量的 DataFrame
    """
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)

    # 如果未指定列，默认为所有数值列
    if columns is None:
        columns = df.select_dtypes(include='number').columns.tolist()

    # 如果未指定统计量，默认为均值、中位数和标准差
    if stats is None:
        stats = ['mean', 'median', 'std']

    # 计算统计量
    result = df[columns].agg(stats)

    return result


def merge_fidelity_sensitive_files(result_path):
    for fidelity, fidelity_value in fidelities.items():
        print(f"Fidelity: {fidelity}")
        fidelity_result_path = os.path.join(result_path, fidelity)
        fidelity_dict = sort_csv_files_by_fidelity(result_path, fidelity)
        for fidelity_value, files in fidelity_dict.items():
            print(f"  {fidelity_value}: {len(files)} files")
            dfs = []
            for file in files:
                path = os.path.join(result_path, file)
                df = pd.read_csv(path)
                dfs.append(df)
            merged_df = pd.concat(dfs, ignore_index=True)  # 一个fidelity下的所有文件合并
            output_path = os.path.join(fidelity_result_path, f"{fidelity_value}_merged.csv")
            merged_df.to_csv(output_path, index=False)
            print(f" fidelity: {fidelity} Merged file saved to: {output_path}")


def analyze_fidelity_sensitive(result_path):
    for fidelity, fidelity_value in fidelities.items():
        print(f"Fidelity: {fidelity}")
        fidelity_result_path = os.path.join(result_path, fidelity)
        for file in os.listdir(fidelity_result_path):
            print(f"  Analyzing file: {file}")
            if file.endswith(".csv"):
                path = os.path.join(fidelity_result_path, file)
                result = calculate_statistics(path, analyze_metrix)
                print(result)


analyze_fidelity_sensitive(result_path)



