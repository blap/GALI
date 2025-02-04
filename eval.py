import os
import json
import argparse
import numpy as np
import torch
from leval.Evaluation.auto_eval import *
import re

K2Length = {
    "2k": 2048,
    "4k": 4096,
    "6k": 6144,
    "8k": 8192,
    "16k": 16384,
    "32k": 32768,
}

def find_and_open_timestamped_file(directory):
    """
    Check if any file in the directory is named with a timestamp in the format 'YYYY-MM-DD_HH-MM-SS'.
    If found, open the first matching file.

    Parameters:
        directory (str): Path to the directory to search.

    Returns:
        None
    """
    # Define the regex pattern for timestamp format
    pattern = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}.json")
    
    # Search for timestamp-named files in the directory
    for file_name in os.listdir(directory):
        if pattern.fullmatch(file_name):  # Check if file name matches the pattern
            found_file = os.path.join(directory, file_name)
            print(f"Timestamped file found: {found_file}")
            
            # Open and read the file (for demonstration, open in read mode)
            try:
                with open(found_file, "r", encoding="utf-8") as f:
                    json_content = json.load(f)
                    return json_content
            except Exception as e:
                print(f"Error opening file: {e}")
            
            return None
    print(f"No timestamped files found in {directory}.")
    return None


from longbench.metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

dataset2count = {
    "narrativeqa": 200,
    "qasper": 200,
    "multifieldqa_en": 150,
    "multifieldqa_zh": 200,
    "hotpotqa": 200,
    "2wikimqa": 200,
    "musique": 200,
    "dureader": 200,
    "gov_report": 200,
    "qmsum": 200,
    "multi_news": 200,
    "vcsum": 200,
    "trec": 200,
    "triviaqa": 200,
    "samsum": 200,
    "lsht": 200,
    "passage_retrieval_en": 200,
    "passage_count": 200,
    "passage_retrieval_zh": 200,
    "lcc": 500,
    "repobench-p": 500,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--exp', type=str,default=None)
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
        # if dataset == "passage_count":
        #     print("score ",score, total_score)
    # print("len(predictions) ",len(predictions) )
    return round(100 * total_score / len(predictions), 2)


def longbench_eval(exp_dir):
    scores = dict()
    scores_o = dict()
    scores_t = dict()
    results_root = "results/lb/"
    arg_e = False
    if arg_e:
        path = f"{results_root}pred_e/{exp_dir}/"
    else:
        path = f"{results_root}pred/{exp_dir}/"

    params_tested = find_and_open_timestamped_file(path)
    if params_tested is not None:
        if "ins-8k-attn" in exp_dir:
            exp_name = params_tested['exp_name']
            ori_k = "8k"
            tar_k = "16k"
            treshhold_length_o = K2Length[ori_k]
            treshhold_length_t = K2Length[tar_k]
        else:
            exp_name = params_tested['exp_name']
            ori_k = exp_name.split("-to-")[0].split("-")[-1]
            tar_k = exp_name.split("-to-")[1].split("-")[0]
            treshhold_length_o = K2Length[ori_k]
            treshhold_length_t = K2Length[tar_k]
    else:
        treshhold_length_o = 0
        treshhold_length_t = 0

    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        predictions_t, answers_t, lengths_t = [], [], []
        predictions_o, answers_o, lengths_o = [], [], []
        dataset = filename.split('.')[0]
        if "_ori" in dataset:
            continue
            # dataset = dataset.split("_ori")[0]
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
                if data.get("prefill_len", -1) >= treshhold_length_o:
                    predictions_o.append(data["pred"])
                    answers_o.append(data["answers"])
                if data.get("prefill_len", -1) >= treshhold_length_t:
                    predictions_t.append(data["pred"])
                    answers_t.append(data["answers"])

        if arg_e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)

            if len(predictions_o) > 0:
                score_o = scorer(dataset, predictions_o, answers_o, all_classes)
            else:
                score_o = 0

            if len(predictions_t) > 0:
                score_t = scorer(dataset, predictions_t, answers_t, all_classes)
            else:
                score_t = 0
        if len(predictions) < dataset2count[dataset]:
            scores[dataset] = f"{score}({len(predictions)}/{dataset2count[dataset]})"
        else:
            scores[dataset] = score

        scores_o[dataset] = f"{score_o}({len(predictions_o)}/{dataset2count[dataset]})"
        scores_t[dataset] = f"{score_t}({len(predictions_t)}/{dataset2count[dataset]})"
    if arg_e:
        out_path = f"{results_root}pred_e/{exp_dir}/result.json"
    else:
        out_path = f"{results_root}pred/{exp_dir}/result.json"
        out_path_o = f"{results_root}pred/{exp_dir}/result_o.json"
        out_path_t = f"{results_root}pred/{exp_dir}/result_t.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    with open(out_path_o, "w") as f:
        json.dump(scores_o, f, ensure_ascii=False, indent=4)
    with open(out_path_t, "w") as f:
        json.dump(scores_t, f, ensure_ascii=False, indent=4)


def leval_eval(exp_dir):
    scores = dict()
    scores_o = dict()
    # This script can calulate these metrics
    SUPPORT_METRICS = ["f1", "rouge", "exam"]

    results_root = "results/le/"
    path = f"{results_root}pred/{exp_dir}/"
    all_files = os.listdir(path)
    # search for the prediction key

    params_tested = find_and_open_timestamped_file(path)
    if params_tested is not None:
        exp_name = params_tested['exp_name']
        ori_k = exp_name.split("-to-")[0].split("-")[-1]
        tar_k = exp_name.split("-to-")[1].split("-")[0]
        treshhold_length_o = K2Length[ori_k]
        treshhold_length_t = K2Length[tar_k]
    else:
        treshhold_length_o = 0
        treshhold_length_t = 0

    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        filename = os.path.join(path, filename)
        _, dataset = os.path.split(filename)
        dataset = dataset.split('.')[0]
        pred_data = read_jsonl(filename)
        prediction_key = 0
        if len(pred_data) == 0:
            continue
        for key in pred_data[0]:
            if "pred" in key:
                prediction_key = key
                break
        predictions = []
        predictions_o = []
        references = []
        references_o = []
        if  "topic_retrieval_longchat" in filename:
            references = [[], [], []]
            predictions = [[], [], []]
            references_o = [[],[],[]]
            predictions_o = [[],[],[]]
        elif "sci_fi" in filename:
            references = [[], []]
            predictions = [[], []]
            references_o = [[],[]]
            predictions_o = [[],[]]

        config_name = None

        with_options = False
        for task in with_option_tasks:
            if task in filename:
                with_options = True
                break

        for i,instance in enumerate(pred_data):
            prefill_len = instance.get("prefill_len", -1)
            if instance["evaluation"] not in SUPPORT_METRICS:
                continue
            if with_options:
                references.append([process_gt_mc(instance["gt"])])
                predictions.append(process_output_mc(instance[prediction_key],filename))
                if prefill_len >= treshhold_length_o:
                    references_o.append([process_gt_mc(instance["gt"])])
                    predictions_o.append(process_output_mc(instance[prediction_key],filename))
            elif "gsm" in filename:
                references.append([process_math(instance["gt"])])
                predictions.append(process_math(instance[prediction_key]))
                if prefill_len >= treshhold_length_o:
                    references_o.append([process_math(instance["gt"])])
                    predictions_o.append(process_math(instance[prediction_key]))
            elif "codeU" in filename:
                references.append([process_gt_code(instance["gt"])])
                predictions.append(process_output_code(instance[prediction_key], instance["gt"]))
                if prefill_len >= treshhold_length_o:
                    references_o.append([process_gt_code(instance["gt"])])
                    predictions_o.append(process_output_code(instance[prediction_key], instance["gt"]))
            elif "topic_retrieval_longchat" in filename:
                references[i%3].append([instance["gt"].lower()])
                predictions[i%3].append(instance[prediction_key].lower())
                if prefill_len >= treshhold_length_o:
                    references_o[i%3].append([instance["gt"].lower()])
                    predictions_o[i%3].append(instance[prediction_key].lower())
            elif "sci_fi" in filename:
                loyalty, fact = process_gt_judge(instance["gt"])
                references[0].append([loyalty])
                references[1].append([fact])
                loyalty_pred, fact_pred = process_output_judge(instance[prediction_key])
                predictions[0].append(loyalty_pred)
                predictions[1].append(fact_pred)
                if prefill_len >= treshhold_length_o:
                    references_o[0].append([loyalty])
                    references_o[1].append([fact])
                    predictions_o[0].append(loyalty_pred)
                    predictions_o[1].append(fact_pred)
            else:
                references.append([instance["gt"]])
                predictions.append(instance[prediction_key])
                if prefill_len >= treshhold_length_o:
                    references_o.append([instance["gt"]])
                    predictions_o.append(instance[prediction_key])
            config_name = instance["evaluation"]
        assert config_name is not None

        if config_name in SUPPORT_METRICS:
            print("begin evaluating:", config_name)
            LEval_metric = LEvalMetrics(config_name=config_name)
            LEval_metric_o = LEvalMetrics(config_name=config_name)
            if "topic_retrieval_longchat" in filename:
                output_str = ""
                balance_score = 0
                for i in range(len(predictions)):
                    pred = predictions[i]
                    ref = references[i]
                    metrics = LEval_metric.compute(predictions=pred, references=ref)
                    output_str += f"first {i+1} sentence retrieval score: {metrics}\n"
                    balance_score += metrics["exact_match"]
                    scores[f"{dataset}_{i+1}"] = metrics
                print(output_str[:-1])
                scores[f"{dataset}"] = {"LEval_score": balance_score/3}
                print(f"average score of the 1st/2nd/3rd sentence retrieval: {balance_score/3}")


                output_o_str = ""
                balance_o_score = 0
                for i in range(len(predictions_o)):
                    pred = predictions_o[i]
                    ref = references_o[i]
                    if len(pred) > 0:
                        metrics_o = LEval_metric_o.compute(predictions=pred, references=ref)
                        output_o_str += f"first {i+1} sentence retrieval score: {metrics_o}\n"
                        balance_o_score += metrics["exact_match"]
                    else:
                        metrics_o = {"none":0}
                    scores_o[f"{dataset}_{i+1}"] = metrics
                print(output_o_str[:-1])
                scores_o[f"{dataset}"] = {"LEval_score": balance_o_score/3}
                print(f"average score of the 1st/2nd/3rd sentence retrieval longer than t: {balance_o_score/3}")
            elif "sci_fi" in filename:
                output_str = ""
                balance_score = 0
                for i in range(len(predictions)):
                    pred = predictions[i]
                    ref = references[i]
                    metrics = LEval_metric.compute(predictions=pred, references=ref)
                    if i ==0:
                        output_str += f"loyalty score: {metrics}\n"
                        scores[f"{dataset}_loyalty"] = metrics
                    else:
                        output_str += f"fact score: {metrics}"
                        scores[f"{dataset}_fact"] = metrics
                    
                    balance_score += metrics["exact_match"]
                print(output_str)
                scores[f"{dataset}"] = {"LEval_score": balance_score/2}
                print(f"average score of fact and loyalty: {balance_score/2}")

                output_o_str = ""
                balance_o_score = 0
                for i in range(len(predictions_o)):
                    pred = predictions_o[i]
                    ref = references_o[i]
                    if len(pred) > 0:
                        metrics_o = LEval_metric_o.compute(predictions=pred, references=ref)
                        if i ==0:
                            output_o_str += f"loyalty score: {metrics_o}\n"
                            scores_o[f"{dataset}_loyalty"] = metrics_o
                        else:
                            output_o_str += f"fact score: {metrics_o}"
                            scores_o[f"{dataset}_fact"] = metrics_o
                    else:
                        metrics_o = {"none":0}
                    balance_o_score += metrics["exact_match"]
                print(output_o_str)
                scores_o[f"{dataset}"] = {"LEval_score": balance_o_score/2}
                print(f"average score of fact and loyalty longer than t: {balance_o_score/2}")
            else:
                metrics = LEval_metric.compute(predictions=predictions, references=references)
                print(metrics)
                if len(predictions_o) > 0:
                    metrics_o = LEval_metric_o.compute(predictions=predictions_o, references=references_o)
                else:
                    metrics_o = {"none":0}
                print(metrics_o)
                scores[dataset] = metrics
                scores_o[dataset] = metrics_o
        else:
            print(config_name, "evaluation is not ready")
            input("press enter to continue calculate other metrics")
    out_path = f"{results_root}pred/{exp_dir}/result.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    
    out_path = f"{results_root}pred/{exp_dir}/result_o.json"
    with open(out_path, "w") as f:
        json.dump(scores_o, f, ensure_ascii=False, indent=4)


def pg19_eval(exp_dir):
    results_root = "results/pg19/"
    path = f"{results_root}pred/{exp_dir}/"
    all_files = os.listdir(path)
    # search for the prediction key

    print("Evaluating on:", all_files)
    base = 1024
    max_step = 64
    results = {}
    for i in range(1, max_step+1):
        results[str(base * i)] = []
    log_prob_list = None
    for filename in all_files:
        if not filename.endswith(".pt"):
            continue
    
        file_path = os.path.join(path, filename)
        pred_data = torch.load(file_path)
        for one_pred in pred_data:
            log_prob_list = one_pred['log_prob_list']
            log_prob_list_pt = torch.tensor(log_prob_list)
            for length, ppl_list in results.items():
                length = int(length)
                if length <= len(log_prob_list_pt)+20: # exclude some special tokens
                    ppl = torch.exp(log_prob_list_pt[:length].sum()/length)
                    ppl_list.append(ppl)
        break
    if log_prob_list is None:
        return
    for length, ppl_list in results.items():
        results[length] = torch.mean(torch.tensor(ppl_list)).item()
    
    out_path = f"{results_root}pred/{exp_dir}/result.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def needle_eval(exp_dir):
    results_root = "results/pg19/"
    path = f"{results_root}pred/{exp_dir}/"
    all_files = os.listdir(path)
    # search for the prediction key

    print("Evaluating on:", all_files)
    base = 1024
    max_step = 64
    results = {}
    for i in range(1, max_step+1):
        results[str(base * i)] = []
    log_prob_list = None
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        # should only have one jsonl file
        # _, dataset = os.path.split(filename)
        # dataset = dataset.split('.')[0]
        file_path = os.path.join(path, filename)
        pred_data = read_jsonl(file_path)
        for one_pred in pred_data:
            log_prob_list = one_pred['log_prob_list']
            log_prob_list_pt = torch.tensor(log_prob_list)
            for length, ppl_list in results.items():
                length = int(length)
                ppl = torch.exp(log_prob_list_pt[:length].sum()/length)
                ppl_list.append(ppl)
        break
    if log_prob_list is None:
        return
    for length, ppl_list in results.items():
        results[length] = torch.mean(torch.tensor(ppl_list)).item()
    
    out_path = f"{results_root}pred/{exp_dir}/result.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def process_folders(base_path, target_file="results.json", process_function=None):
    """
    Traverse all folders under a base path. If a folder contains a target file, skip it;
    otherwise, execute a specified function.

    Args:
        base_path (str): The root directory to traverse.
        target_filename (str): The filename to check in each folder (default: "results.json").
        process_function (callable): The function to execute on folders without the target file.
                                     This function should accept the folder path as its argument.
    """
    for entry in os.listdir(base_path):
        subfolder_path = os.path.join(base_path, entry)

        # Check if the entry is a directory
        if os.path.isdir(subfolder_path):
            # Check if the target file exists in the directory
            if not os.path.exists(os.path.join(subfolder_path, target_file)) and "-to-" in subfolder_path:
                print(f"Target file '{target_file}' not found in: {subfolder_path}")
                
                # Execute the process_function if provided
                if process_function:
                    process_function(entry)
            elif not os.path.exists(os.path.join(subfolder_path, target_file)) and "ins-8k-attn" in subfolder_path:
                print(f"Target file '{target_file}' not found in: {subfolder_path}")
                
                # Execute the process_function if provided
                if process_function:
                    process_function(entry)
            else:
                print(f"Skipping subfolder: {subfolder_path} (contains {target_file})")

if __name__ == '__main__':
    args = parse_args()
    task = args.task
    exp_dir = args.exp
    if exp_dir == "all":
        if task == 'longbench':
            process_folders("results/lb/pred", target_file="result.json", process_function=longbench_eval)
        elif task == 'leval':
            process_folders("results/le/pred", target_file="-result.json", process_function=leval_eval)
        elif task == 'pg19':
            process_folders("results/pg19/pred", target_file="result.json", process_function=pg19_eval)
    else:
        if task == 'longbench':
            longbench_eval(exp_dir)
        elif task == 'leval':
            leval_eval(exp_dir)
        elif task == 'pg19':
            pg19_eval(exp_dir)

