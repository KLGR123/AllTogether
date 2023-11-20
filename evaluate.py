import re
import os
import json
import time
import pickle
import random
import logging
import numpy as np
from tqdm import tqdm

import openai
import lxml
from lxml import etree

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import transformers
from datasets import load_dataset
from transformers import pipeline
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
    
from utils import get_tree_repr
from utils import prune_tree


# logger = logging.getLogger(__name__)
OPENAI_API_KEY = "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"
GPT_MODEL_TYPE = "gpt-4"
DATA_PATH = "/mnt/data/liujiarun/projects/Mind2Web"
MODEL_PATH = "/mnt/data/liujiarun/projects/llama2/llama-2-13b-chat-hf"
TOKENIZER_PATH = "/mnt/data/liujiarun/projects/llama2/llama-2-13b-chat-hf"
SCORE_FILE_PATH = "/mnt/data/liujiarun/projects/Mind2Web/model/scores_all_data.pkl"
TEST_SPLIT_FILES = {'test_domain': 'data/test_domain/*.json'} # 'data/test_task/*.json', 'test_website': 'data/test_website/*.json', 'test_domain': 'data/test_domain/*.json'
OUTPUT_PATH = "/mnt/data/liujiarun/projects/Mind2Web/output"
PEFT_MODEL_PATH = "/mnt/data/liujiarun/projects/Mind2Web/training/llama2-13b/checkpoint-300" # llama2-7b-better/checkpoint-400 llama2-7b/checkpoint-550
EVAL_DATASET_PATH = "/mnt/data/liujiarun/projects/Mind2Web/evaluating/balanced_data/test_website"
PREVIOUS_K = 7 
NEG_RATIO = 0.1
NUM_CANDIDATES = 5
MAX_CONTEXT_LEN = 512
MAX_NEW_TOKENS = 50 # 50
TOP_K = 50
TEMPERATURE = 0.1
GPT_RANDOM = 0.7

USE_GPT = False
USE_PEFT = False
LOAD_IN_8BIT = True
IS_TRAIN = False
FORCE_INSTRUCTION = True
OUTPUT_FILE = False
SEPERATE_POS_NEG = False

openai.api_key = OPENAI_API_KEY
openai.proxy = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890'
}


def format_input_multichoice(sample, candidate_ids, gt=-1, previous_k=PREVIOUS_K, keep_html_brackets=False, force_instruction=FORCE_INSTRUCTION):
    dom_tree = lxml.etree.fromstring(sample["cleaned_html"])
    dom_tree = prune_tree(dom_tree, candidate_ids)
    seq_context, id_mapping = get_tree_repr(dom_tree, id_mapping={}, keep_html_brackets=keep_html_brackets)
    candidate_nodes = dom_tree.xpath("//*[@backend_node_id]")

    choices = []
    for idx, node in enumerate(candidate_nodes):
        tree_reprs = get_tree_repr(node, id_mapping=id_mapping, keep_html_brackets=keep_html_brackets)
        choice = [node.attrib["backend_node_id"], " ".join(tree_reprs[0].split()[:10])]
        choices.append(choice)

    seq_input = (
        "\n\nBased on the HTML webpage above, try to complete the following task:\n"
        f"Task: {sample['confirmed_task']}\n"
        f"Previous actions:\n"
    )

    if len(sample["previous_actions"]) > 0:
        for action in sample["previous_actions"][-previous_k:]:
            seq_input += f"{action}\n"
    else:
        seq_input += "None\n"
        
    seq_input += (
        "\nWhat should be the next action? Please select from the following choices "
        "(If the correct action is not in the page above, please select A. 'None of the above'):\n\n"
        "A. None of the above\n"
    )

    for idx, choice in enumerate(choices):
        seq_input += f"{chr(66 + idx)}. {choice[1]}\n"

    if force_instruction:
        seq_input += (
            "\nYour output format should be\n```"
            "<The option you selected (A / B / C...)>.\n"
            "Action: <The action you want to perform, action can be CLICK / TYPE / SELECT>.\n"
            "Value: <Optional. value can be text content or None (if action is CLICK)."
            "```"
        )

    gt = id_mapping.get(gt, -1)
    if gt == -1:
        seq_target = "A."
    else:
        current_action_op = sample["operation"]["op"]
        current_action_value = sample["operation"]["value"]
        seq_target = f"{chr(66 + gt)}.\n" + f"Action: {current_action_op}\n"
        if current_action_op != "CLICK":
            seq_target += f"Value: {current_action_value}"
    return seq_context, seq_input, seq_target, choices


def get_data_split(data_dir, split_file, candidate_results=None, is_train=False):
    dataset = load_dataset(data_dir, data_files=split_file, split="all")
    candidate_scores = None
    candidate_ranks = None

    def flatten_actions(samples):
        outputs = {"website": [], "confirmed_task": [], "annotation_id": [], "previous_actions": [], "action_uid": [],
                    "operation": [], "pos_candidates": [], "neg_candidates": [], "cleaned_html": []}
                    
        num_actions = [len(actions) for actions in samples["actions"]]
        for key in ["website", "confirmed_task", "annotation_id"]:
            for idx, value in enumerate(samples[key]):
                outputs[key] += [value] * num_actions[idx]

        for actions, action_reprs in zip(samples["actions"], samples["action_reprs"]):
            for a_idx, action in enumerate(actions):
                outputs["previous_actions"].append(action_reprs[:a_idx])
                for key in ["action_uid", "operation", "pos_candidates", "neg_candidates", "cleaned_html"]:
                    outputs[key].append(action[key])

        return outputs

    def get_score(sample):
        sample_id = f"{sample['annotation_id']}_{sample['action_uid']}"
        for candidates in [sample["pos_candidates"], sample["neg_candidates"]]:
            for candidate in candidates:
                candidate_id = candidate["backend_node_id"]
                candidate["score"] = candidate_scores[sample_id][candidate_id]
                candidate["rank"] = candidate_ranks[sample_id][candidate_id]

        return {"pos_candidates": sample["pos_candidates"], "neg_candidates": sample["neg_candidates"]}
    
    flatten_dataset = dataset.map(flatten_actions, batched=True, remove_columns=dataset.column_names, batch_size=10, num_proc=4)
    
    if candidate_results is not None:
        candidate_scores = candidate_results["scores"]
        candidate_ranks = candidate_results["ranks"]

    flatten_dataset = flatten_dataset.map(get_score)
    if is_train:
        flatten_dataset = flatten_dataset.filter(lambda x: len(x["pos_candidates"]) > 0)
    return flatten_dataset


class MultiChoiceDataset(Dataset):
    def __init__(self, data, neg_ratio=0.05, num_candidates=5, max_context_len=512, top_k=-1):
        self.data = data
        self.neg_ratio = neg_ratio
        self.num_candidates = num_candidates
        self.max_context_len = max_context_len
        self.top_k = top_k

    def __len__(self):
        return len(self.data) * 10

    def __getitem__(self, idx):
        sample = self.data[idx // 10]
        top_negatives = [c for c in sample["neg_candidates"] if c["rank"] < self.top_k]
        other_negatives = [c for c in sample["neg_candidates"] if c["rank"] >= self.top_k]
        neg_candidates = top_negatives if random.random() < 0.8 and len(top_negatives) > 0 else other_negatives

        if len(sample["pos_candidates"]) > 0 and (random.random() > self.neg_ratio or len(neg_candidates) == 0):
            pos_candidate = random.choice(sample["pos_candidates"])
            neg_candidate = random.sample(neg_candidates, min(len(neg_candidates), self.num_candidates - 1))
            candidate_ids = [pos_candidate["backend_node_id"]] + [c["backend_node_id"] for c in neg_candidate]  
            seq_context, seq_in, seq_out, _ = format_input_multichoice(sample, candidate_ids, gt=pos_candidate["backend_node_id"]) 
        else:
            neg_candidate = random.sample(neg_candidates, min(len(neg_candidates), self.num_candidates), )
            candidate_ids = [c["backend_node_id"] for c in neg_candidate]
            seq_context, seq_in, seq_out, _ = format_input_multichoice(sample, candidate_ids, gt=-1)
        
        model_input = {"inputs": seq_context + seq_in, "labels": seq_out}
        return model_input


def postprocess_action(action):
    action = action.strip()
    selected_option = action[0]
    action = re.search(r"Action: (CLICK|SELECT|TYPE)", action)
    action = action.group(1) if action is not None else ""
    value = re.search(r"Value: (.*)$", action, re.MULTILINE)
    value = value.group(1) if value is not None else ""
    return selected_option, action.strip() + " " + value.strip()


def calculate_f1(pred, label):
    pred = set(pred.strip().split())
    label = set(label.strip().split())

    if len(pred) == 0 and len(label) == 0:
        return 1
    if len(pred) == 0 or len(label) == 0:
        return 0

    tp = len(pred & label)
    fp = len(pred - label)
    fn = len(label - pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if precision == 0 or recall == 0:
        return 0
        
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def evaluate_two_step_model():
    print(f"Use model {MODEL_PATH}")
    tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
    model = LlamaForCausalLM.from_pretrained(MODEL_PATH, load_in_8bit=LOAD_IN_8BIT, device_map='auto', torch_dtype=torch.float16)
    model.eval()

    candidate_results = None
    test_dataset_dict = {}

    with open(SCORE_FILE_PATH, "rb") as f:
        candidate_results = pickle.load(f)

    for test_key, test_split_file in TEST_SPLIT_FILES.items():
        test_data = get_data_split(DATA_PATH, test_split_file, candidate_results=candidate_results, is_train=IS_TRAIN)
        test_dataset_dict[test_key] = MultiChoiceDataset(test_data, neg_ratio=NEG_RATIO, num_candidates=NUM_CANDIDATES, max_context_len=MAX_CONTEXT_LEN)

    with torch.no_grad():
        for test_key, test_dataset in test_dataset_dict.items():
            print(f"Start evaluating for {test_key}. The dataset length is {len(test_dataset.data)}.")

            all_element_acc = []
            all_action_f1 = []
            all_step_sr = []
            all_final_predictions = []
            all_outputs = []
            all_failed_task_set = set()
            all_task_set = set()

            for k in [1, 5, 10, 20, 50]:
                top_k_list = [1 if any([c["rank"] < k for c in sample["pos_candidates"]]) else 0 for sample in test_dataset.data]
                recall_at_k = np.mean(top_k_list)
                print(f"Recall Cap @ {k}: {recall_at_k}")

            with tqdm(total=len(test_dataset.data)) as t:
                for sample in test_dataset.data:
                    all_task_set.add(sample["annotation_id"])
                    final_prediction = None
                    outputs = []

                    pos_candidates = sample["pos_candidates"]
                    pos_candidates = [c for c in pos_candidates if c["rank"] < TOP_K]
                    pos_ids = [c["backend_node_id"] for c in pos_candidates]
                    neg_candidates = sample["neg_candidates"]
                    neg_candidates = [c for c in neg_candidates if c["rank"] < TOP_K]
                    neg_ids = [c["backend_node_id"] for c in neg_candidates]
                    all_candidates = pos_ids + neg_ids
                    random.shuffle(all_candidates)

                    if len(pos_ids) == 0:
                        all_element_acc.append(0)
                        all_action_f1.append(0)
                        all_step_sr.append(0)

                        all_failed_task_set.add(sample["annotation_id"])
                        all_final_predictions.append([f"{sample['annotation_id']}_{sample['action_uid']}", 
                                                        sample["confirmed_task"], sample["website"], "", ""])
                        all_outputs.append([f"{sample['annotation_id']}_{sample['action_uid']}", 
                                            sample["confirmed_task"], sample["website"], []])
                        t.update()
                        continue

                    while len(all_candidates) > 1:
                        candidate_ids = all_candidates[:NUM_CANDIDATES]
                        all_candidates = all_candidates[NUM_CANDIDATES:]
                        seq_context, seq_in, _, choices = format_input_multichoice(sample, candidate_ids)
                        outputs.append([candidate_ids, [seq_context, seq_in, choices], None])

                        model_input = seq_context + seq_in
                        model_input = tokenizer(model_input, return_tensors="pt").to("cuda")

                        output = model.generate(**model_input, max_new_tokens=MAX_NEW_TOKENS)[0]
                        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
                        outputs[-1][-1] = decoded_output[0]
                        print("Llama model output: ", decoded_output)
                        pred_element, pred_action = postprocess_action(decoded_output)

                        if pred_element[0] != "A":
                            try:
                                pred_element_id = choices[ord(pred_element[0]) - ord("B")][0]
                                all_candidates.append(pred_element_id)
                                final_prediction = (pred_element_id, pred_action)
                            except IndexError:
                                print(f"IndexError: {decoded_output}")
                                print(f"Choices: {choices}")
                    
                    all_outputs.append([f"{sample['annotation_id']}_{sample['action_uid']}", outputs])

                    if len(all_candidates) == 0 or final_prediction is None:
                        all_element_acc.append(0)
                        all_action_f1.append(0)
                        all_step_sr.append(0)
                        all_failed_task_set.add(sample["annotation_id"])
                        all_final_predictions.append([f"{sample['annotation_id']}_{sample['action_uid']}", "", ""])
                        print("len(all_candidates) is ", len(all_candidates))
                        print("final_prediction is ", final_prediction)
                    else:
                        pred_action = pred_action.split()
                        pred_action_op = pred_action[0]
                        pred_action_value = pred_action[1] if len(pred_action) > 1 else ""

                        print("pred_action_op:", pred_action_op)
                        print("pred_action_value:", pred_action_value)
                        print("sample_op:", sample["operation"]["op"])
                        print("sample_value:", sample["operation"]["value"])

                        all_element_acc.append(final_prediction[0] in pos_ids)
                        all_true = (pred_action_op==sample["operation"]["op"]) and (pred_action_value==sample["operation"]["value"])
                        all_step_sr.append(all_true)
                        all_failed_task_set.add(sample["annotation_id"]) if not all_true else None
                        _, _, target_out, _ = format_input_multichoice(sample, pos_ids[0], gt=pos_ids[0])
                        _, target_action = postprocess_action(target_out)
                        all_action_f1.append(calculate_f1(final_prediction[1], target_action))
                        all_final_predictions.append([f"{sample['annotation_id']}_{sample['action_uid']}", final_prediction[0], final_prediction[1]])

                    t.set_postfix(element_acc=np.mean(all_element_acc) * 100,
                                  action_f1=np.mean(all_action_f1) * 100,
                                  step_sr=np.mean(all_step_sr) * 100,
                                  final_sr=(1 - (len(all_failed_task_set) / len(all_task_set))) * 100)
                    t.update()

            result = {"element_acc": np.mean(all_element_acc) * 100, "action_f1": np.mean(all_action_f1) * 100,
                      "step_sr": np.mean(all_step_sr) * 100, "final_sr": (1 - (len(all_failed_task_set) / len(all_task_set))) * 100}

            with open(f"{OUTPUT_PATH}/{test_key}_predictions_top{TOP_K}.json", "w") as f:
                json.dump(all_final_predictions, f)
            with open(f"{OUTPUT_PATH}/{test_key}_results_top{TOP_K}.json", "w") as f:
                json.dump(result, f, indent=4)
            with open(f"{OUTPUT_PATH}/{test_key}_outputs_top{TOP_K}.json", "w") as f:
                json.dump(all_outputs, f)

            print(f"Result for {test_key}: {result}.")
    print(f"Finish Evaluating.")


def postprocess_output(output):
    if "None" in str(output):
        pos_id, op, value = "", "", ""
    else:
        output_lines = output.split("\n")
        output_lines = [line for line in output_lines if line]
        print(output_lines)
        for line in output_lines:
            line = re.sub(r"(?<![\'\"])(CLICK|TYPE|SELECT)(?![\'\"])", r"'\1'", line)
            try:
                pos_id, op, value = eval(line)
                break
            except Exception as e:
                number_list = re.findall(r'\d+', line)
                pos_id = str(number_list[0]) if number_list else ""
                op = "CLICK" if "CLICK" in line else "TYPE" if "TYPE" in line else "SELECT" if "SELECT" in line else ""
                value = line.lower().split("value")[-1] if "value" in line.lower() else ""
                if pos_id and op:
                    break
                continue
    
    pos_id = "" if pos_id and not op else pos_id
    return pos_id, op, value


def evaluate():
    if not USE_GPT:
        print(f"Use model {MODEL_PATH}")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb_config, device_map="auto",
                                                        trust_remote_code=True, use_auth_token=True)
        
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        if USE_PEFT:
            model = PeftModel.from_pretrained(base_model, PEFT_MODEL_PATH)
        else:
            model = base_model

        model.eval()

    element_success_set = set()
    element_all_set = set()

    outputs_list = []
    acc_list = []
    pos_acc_list = []
    neg_acc_list = []
    pos_op_acc_list = []
    f1_score_list = []
    pos_element_dist = []

    with torch.no_grad():
        for filename in os.listdir(EVAL_DATASET_PATH):
            samples = []
            with open(os.path.join(EVAL_DATASET_PATH, filename), "r") as file:
                print("Evaluating datafile", filename)
                for line in file:
                    if (USE_GPT and random.random() > GPT_RANDOM) or not USE_GPT:
                        samples.append(json.loads(line))

            with tqdm(total=len(samples)) as pbar:
                for sample in samples:
                    try:
                        setting_prompt = (
                            "Here is a web intelligence agent that interacts with webpage environments. "
                            "It can get the next ACTION based on the user's TASK_GOAL, a piece of current HTML_CHUNK and a historical ACTION_HISTORY sequence recording the actions that have been performed. \n\nThe current HTML_CHUNK is: \n```\n"
                        )
                        post_setting_prompt = (
                            f"""The TASK_GOAL is: 'You are at website {sample["website"]}. {sample["confirmed_task"]}'. \n"""
                            f"""The ACTION_HISTORY trajectory is: {str(sample["previous_actions"])}. \n\n"""
                        )
                        instruction_prompt = (
                            "Given HTML_CHUNK and TASK_GOAL and ACTION_HISTORY, What the web agent need to do is select an element from this HTML code to interact with, thus bringing the goal closer to completion. \n"
                            "It should first determine if there are any elements in HTML_CHUNK that need to be interacted with, and if not, its output should be 'None'.\n"
                            "However, if there is an element to interact with (such as clicking or typing or selecting), the agent needs to output an ACTION. \nAn ACTION is a list with following format: \n"
                            "[<backend_node_id>, <operation>, <value>]\n"
                            "<backend_node_id> should be the chosen element's backend_node_id in HTML.\n"
                            "<operation> should be 'CLICK' or 'TYPE' or 'SELECT' for interacting with the chosen element.\n"
                            "<value> should be '' if <operation> is 'CLICK', or it should be the specific text content for typing or selecting.\n\n"
                            "Now based on all the information as above, the web agent's output is: " 
                        )
                        instruction_prompt_1 = (
                            "Given HTML_CHUNK and TASK_GOAL and ACTION_HISTORY, What the web agent need to do is select an element from this HTML code to interact with, thus bringing the goal closer to completion. \n"
                            "It should first determine if there are any elements in HTML_CHUNK that need to be interacted with (relevant element), if not, its output should be 'None', if there are elements to interact, then output 'Yes'. \n"
                            "Now based on all the information as above, the web agent's output should be: (None or Yes) " 
                        )
                        instruction_prompt_2 = (
                            "Given HTML_CHUNK and TASK_GOAL and ACTION_HISTORY, What the web agent need to do is select an element from this HTML code to interact with, thus bringing the goal closer to completion. \n"
                            "There is a most relevant element to interact with (clicking or typing or selecting), and the agent needs to output an ACTION for that element. \nAn ACTION is a list with following format: \n"
                            "[<backend_node_id>, <operation>, <value>]\n"
                            "<backend_node_id> should be the chosen element's backend_node_id in HTML.\n"
                            "<operation> should be 'CLICK' or 'TYPE' or 'SELECT' for interacting with the chosen element.\n"
                            "<value> should be '' if <operation> is 'CLICK', or it should be the specific text content for typing or selecting.\n\n"
                            "Now based on all the information as above, the web agent's output should be: (ACTION list) " 
                        )

                        gt_backend_node_id = sample["backend_node_id"]
                        gt_op = sample["op"]
                        gt_value = sample["value"]

                        prompt = setting_prompt

                        def get_elements_around_number(string, number, sibling=50):
                            string_list = string.split('\n')
                            index = -1
                            for i, element in enumerate(string_list):
                                if str(number) in element:
                                    index = i
                                    break
                            if index != -1:
                                start_index = max(0, index - sibling)
                                end_index = min(len(string_list) - 1, index + sibling)
                                new_string = '\n'.join(string_list[start_index:end_index+1])
                                return new_string
                            return None

                        result = get_elements_around_number(sample["cleaned_html"], gt_backend_node_id)
                        prompt += result + "\n```\n\n"
                        # prompt += sample["cleaned_html"] + "\n```\n\n"
                        prompt += (post_setting_prompt + instruction_prompt)
                        # prompt = f"<s>[INST] {prompt} [/INST]"

                        if not USE_GPT:
                            prompt = tokenizer(prompt, return_tensors="pt").to("cuda")
                            output = tokenizer.decode(model.generate(**prompt, max_new_tokens=MAX_NEW_TOKENS)[0], skip_special_tokens=True)
                            output = output.split("the web agent's output is: ")[-1][:100]
                        else:
                            output = openai.ChatCompletion.create(model=GPT_MODEL_TYPE, messages=[{"role": "assistant", "content": prompt}],
                                                                    max_tokens=MAX_NEW_TOKENS,
                                                                    temperature=TEMPERATURE)
                            output = output["choices"][0]["message"]["content"]
                            time.sleep(1.5)

                        print(output)

                        pos_id, op, value = postprocess_output(output)
                        print("OUTPUT:", (pos_id, op, value), "GT:", (gt_backend_node_id, gt_op, gt_value))
                        is_correct = 1 if (pos_id == gt_backend_node_id and op == gt_op and value == gt_value) else 0
                        acc_list.append(is_correct)

                        if gt_backend_node_id:
                            pos_acc_list.append(is_correct)
                            pos_op_acc_list.append(1) if (op == gt_op) and gt_op else 0
                            distance = abs(int(gt_backend_node_id) - int(pos_id)) if pos_id else int(gt_backend_node_id)
                            pos_element_dist.append(distance)
                            if gt_value:
                                f1_score_list.append(calculate_f1(value, gt_value))
                        else:                    
                            neg_acc_list.append(is_correct)

                        sample_id = sample["confirmed_task"] + " " + str(sample["previous_actions"])
                        outputs_list.append({sample_id: [output, gt_backend_node_id, gt_op, gt_value]})

                        element_all_set.add(sample_id)
                        if is_correct:
                            element_success_set.add(sample_id)
                    
                        pbar.set_postfix(total_acc=np.mean(acc_list)*100, f1_score=np.mean(f1_score_list)*100, pos_acc=np.mean(pos_acc_list)*100,
                                         neg_acc=np.mean(neg_acc_list)*100, pos_op_acc=np.mean(pos_op_acc_list)*100, pos_element_distance=np.mean(pos_element_dist)*100)
                        pbar.update()

                    except Exception as e:
                        print(e)
                        continue
    
        results = {"total_acc": np.mean(acc_list)*100, "f1_score": np.mean(f1_score_list)*100, "pos_acc": np.mean(pos_acc_list)*100,
                   "neg_acc": np.mean(neg_acc_list)*100, "pos_op_acc": np.mean(pos_op_acc_list)*100, 
                   "pos_element_distance": np.mean(pos_element_dist)*100, "mind2web_element_acc": (len(element_success_set) / len(element_all_set))*100}
        
        if not USE_GPT:
            if OUTPUT_FILE:
                with open(f"{OUTPUT_PATH}/{MODEL_PATH.split('/')[-1]}_{EVAL_DATASET_PATH.split('/')[-1]}_{PEFT_MODEL_PATH.split('/')[-1]}_outputs.json", "w") as f:
                    for output in outputs_list:
                        json.dump(output, f)
            with open(f"{OUTPUT_PATH}/{MODEL_PATH.split('/')[-1]}_{EVAL_DATASET_PATH.split('/')[-1]}_{PEFT_MODEL_PATH.split('/')[-1]}_results.json", "w") as f:
                json.dump(results, f, indent=4)
        else:
            with open(f"{OUTPUT_PATH}/{GPT_MODEL_TYPE}_{EVAL_DATASET_PATH.split('/')[-1]}_results.json", "w") as f:
                json.dump(results, f, indent=4)

    print(f"Finish Evaluating.")


if __name__ == "__main__":
    evaluate()
    # print(postprocess_output("['6551', 'CLICK', '']\n\nHaving obtained the ACTION, the web browser's next action is: [CLICK'].\n\nNow"))