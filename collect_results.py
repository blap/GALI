import os
import json
import pandas as pd
import re
from collections import OrderedDict

lb_columns = ['narrativeqa','qasper','multifieldqa_en','hotpotqa','2wikimqa','musique','gov_report','qmsum','multi_news','trec','triviaqa','samsum','passage_count','passage_retrieval_en','lcc','repobench-p']

# Coursera	GSM	QuALITY	TOFEL	CodeU	SFiction
le_columns = ['coursera', 'gsm100', 'quality', 'tpo', 'codeU', 'sci_fi']

pg19_columns = [str(1024*i) for i in range(1, 65)]

def generate_excel(file_path, data, column_order):
    # Convert to a DataFrame
    new_data = pd.DataFrame(data)

    # # Convert to a DataFrame
    # # Step 2: Check if the Excel file exists
    # if os.path.exists(file_path):
    #     # Load existing data
    #     existing_data = pd.read_excel(file_path)

    #     # Check for duplicates based on file_path
    #     existing_file_paths = set(existing_data["file_path"])
    #     # print("existing_file_paths ",existing_file_paths)
    #     # print("new_data 111 ",new_data)
    #     new_data = new_data[~new_data["file_path"].isin(existing_file_paths)]
    #     # print("new_data ",new_data)
    #     # Combine e xisting and new data
    #     combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    # else:
        # If no existing file, use the new data
    combined_data = new_data

    # Step 3: Reorder columns if column_order is specified
    if column_order:
        for col in column_order:
            if col not in combined_data.columns:
                combined_data[col] = None  # Add missing columns with None
        combined_data = combined_data[column_order]
    else:
        combined_data = combined_data[list(combined_data.columns)]  # Preserve default column order

    # Step 4: Write the combined data back to the Excel file
    combined_data.to_excel(file_path, index=False)
    print(f"Data appended to {file_path}")

def read_json_and_write_to_excel(base_dir, excel_dir, columns):
    """
    Recursively read JSON files in all subdirectories and write their contents to an Excel file with specified columns.

    Parameters:
        base_dir (str): The root directory to search.
        output_excel_path (str): The path where the Excel file will be saved.
        columns (list): The list of column names to extract from JSON files.
        column_order (list, optional): The order of columns in the Excel file.

    Returns:
        None
    """
    repro_data = []  # List to store data from all JSON files
    dpi_data = []
    # Walk through all directories and files
    print("base_dir ",base_dir)
    params_ket_set = set()
    for root, _, files in os.walk(base_dir):
        if "archive" in root:
            continue

        # print("files ",files)
        for file_name in files:
            if file_name not in ('result.json', 'result_t.json', 'result_o.json'):
                continue
            file_path = os.path.join(root, file_name)
            print(f"Processing file: {file_path}")

            if 'pg19' in base_dir:
                with open(file_path, "r") as file:
                    data = file.read()

                data = data.replace("NaN", "null")
                # print("data ",data)
                json_data = json.loads(data)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
            


                
                    
            # Extract specified columns
            row = {"file_path": file_path}  # Always include the file path
            if "lb/pred" in file_path:
                pre_fix = 'lb'
                for col in columns:
                    row[col] = json_data.get(col, None)  # Use None if key is missing
            elif "le/pred" in file_path:
                pre_fix = 'le'
                for col in columns:
                    row[col] = json_data.get(col, {}).get("LEval_score", None)  # Use None if key is missing
            elif "pg19/pred" in file_path:
                pre_fix = 'pg19'
                for col in columns:
                    row[col] = json_data.get(col, None)  # Use None if key is missing

            if "dpi" in root:
                # load params
                all_items = os.listdir(root)
                # 仅保留文件
                # files = [f for f in all_items if os.path.isfile(f)]
                pattern = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}.json")
                for param_file in all_items:
                    # print("param_file ",param_file)
                    if re.fullmatch(pattern, param_file):
                        found_file = os.path.join(root, param_file)
                        with open(found_file, "r", encoding="utf-8") as f:
                            cfg = json.load(f)
                            if 'params' in cfg:
                                dpi_params = cfg['params']
                                # sorted_keys = sorted(dpi_params.keys())
                                for k,v in dpi_params.items():
                                    row[k] = v
                                    params_ket_set.add(k)
                               
                        break
                dpi_data.append(row)
            else:
                repro_data.append(row)
    sorted_keys = sorted(params_ket_set)
    print(row.keys())
    # Convert to a DataFrame
    repro_data = pd.DataFrame(repro_data)
    output_excel_path = os.path.join(excel_dir,f"{pre_fix}_repro.xlsx")
    generate_excel(output_excel_path, repro_data,["file_path"] + columns)

    dpi_data = pd.DataFrame(dpi_data)
    output_excel_path = os.path.join(excel_dir,f"{pre_fix}_dpi.xlsx")
    generate_excel(output_excel_path, dpi_data,["file_path"] + sorted_keys + columns)


# Example usage
# base_directory = "results/lb/pred"  # Replace with the path to your directory

# read_json_and_write_to_excel(base_directory, columns)

import argparse
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default="all")
    parser.add_argument('--task', type=str, default="longbench")
    return parser.parse_args(args)

#  python collect_results.py --task longbench --exp all
if __name__ == '__main__':
    args = parse_args()
    task = args.task
    exp_dir = args.exp
    if exp_dir == "all":
        if task == 'longbench':
            read_json_and_write_to_excel("results/lb/pred", "results/lb/pred", lb_columns)
        elif task == 'leval':
            read_json_and_write_to_excel("results/le/pred", "results/le/pred", le_columns)
        elif task == 'pg19':
            read_json_and_write_to_excel("results/pg19/pred", "results/pg19/pred", pg19_columns)
            pass
    else:
        if task == 'longbench':
            read_json_and_write_to_excel(exp_dir, "results/lb/pred", lb_columns)
        elif task == 'leval':
            read_json_and_write_to_excel(exp_dir, "results/le/pred", le_columns)
        elif task == 'pg19':
            pass
            read_json_and_write_to_excel(exp_dir, "results/pg19/pred", pg19_columns)
