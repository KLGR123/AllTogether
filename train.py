import os
import json
import random
from tqdm import tqdm

import torch
from peft import LoraConfig
from trl import SFTTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer
from transformers import TrainingArguments
from sentencepiece import SentencePieceProcessor


DATA_PATH = "/mnt/data/liujiarun/projects/Mind2Web"
MODEL_PATH = "/mnt/data/liujiarun/projects/llama2/llama-2-13b-chat-hf"
TOKENIZER_PATH = "/mnt/data/liujiarun/projects/llama2/llama-2-13b-chat-hf"
TRAIN_SPLIT_FILES = "data/train/*.json"
DATASET_FOLDER = "/mnt/data/liujiarun/projects/Mind2Web/data/test_website"
OUTPUT_DIR = "/mnt/data/liujiarun/projects/Mind2Web/training/llama2-13b-better"
TRANSFORMED_DATA_DIR = "/mnt/data/liujiarun/projects/Mind2Web/training/balanced_data"

SP = SentencePieceProcessor(model_file=TOKENIZER_PATH+"/tokenizer.model")

MAX_HTML_TOKENS = 3000
MAX_SEQ_LENGTH = 4000
LEARNING_RATE = 2e-4
LOGGING_STEPS = 2
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
THROW_AWAY_RATIO = 0.8
LOAD_IN_4_BIT = True
DO_EVAL = False


def get_train_data_split(data_dir, split_file):
    dataset = load_dataset(data_dir, data_files=split_file, split="all")

    def flatten_actions(samples):
        outputs = {"website": [], "confirmed_task": [], "previous_actions": [], "operation": [], "pos_candidates": [], "cleaned_html": []}      
        num_actions = [len(actions) for actions in samples["actions"]]
        
        for key in ["website", "confirmed_task"]:
            for idx, value in enumerate(samples[key]):
                outputs[key] += [value] * num_actions[idx]

        for actions, action_reprs in zip(samples["actions"], samples["action_reprs"]):
            for a_idx, action in enumerate(actions):
                outputs["previous_actions"].append(action_reprs[:a_idx])
                for key in ["operation", "pos_candidates", "cleaned_html"]:
                    outputs[key].append(action[key])

        return outputs

    flatten_dataset = dataset.map(flatten_actions, batched=True, remove_columns=dataset.column_names, batch_size=10, num_proc=4)
    flatten_dataset = flatten_dataset.filter(lambda x: len(x["pos_candidates"]) > 0)
    return flatten_dataset


def num_tokens_from_string(input_text):
    tokens = SP.EncodeAsIds(input_text)
    return len(tokens)


def get_html_chunk(html, max_tokens=MAX_HTML_TOKENS):
    chunk_num = int(num_tokens_from_string(html) / max_tokens) + 1
    html_lines = html.split('\n')
    chunks = [html_lines[i:i+int(len(html_lines) / chunk_num)] for i in range(0, len(html_lines), int(len(html_lines) / chunk_num))]
    chunks = ['\n'.join(chunk) for chunk in chunks]
    return chunks


def formatting_func_old(samples):
    instructions = []
    for i in range(len(samples["website"])):
        setting_prompt = (
            "Here is a web intelligence agent that interacts with webpage environments. "
            "It can get the next ACTION based on the user's TASK_GOAL, a piece of current HTML_CHUNK and a historical ACTION_HISTORY sequence recording the actions that have been performed. \n\nThe current HTML_CHUNK is: \n```\n"
        )
        post_setting_prompt = (
            f"""The TASK_GOAL is: 'You are at website {samples["website"][i]}. {samples["confirmed_task"][i]}'. \n"""
            f"""The ACTION_HISTORY trajectory is: {str(samples["previous_actions"][i])}. \n\n"""
        )
        instruction_prompt = (
            "Given HTML_CHUNK and TASK_GOAL and ACTION_HISTORY, What the web agent need to do is select an element from this HTML code to interact with, thus bringing the goal closer to completion. \n"
            "It should first determine if there are any elements in HTML_CHUNK that need to be interacted with, and if not, its output should be 'None'. \n"
            "However, if there is an element to interact with (such as clicking or typing or selecting), the agent needs to output an ACTION. \nAn ACTION is a list with following format: \n"
            "[<backend_node_id>, <operation>, <value>]\n"
            "<backend_node_id> should be the chosen element's backend_node_id in HTML.\n"
            "<operation> should be 'CLICK' or 'TYPE' or 'SELECT' for interacting with the chosen element.\n"
            "<value> should be '' if <operation> is 'CLICK', or it should be the specific text content for typing or selecting.\n\n"
            "Now based on all the information as above, the web agent's output is: " 
        )

        html_chunks = get_html_chunk(samples["cleaned_html"][i])
        for html_chunk in html_chunks: 
            prompt = setting_prompt
            prompt += html_chunk + "\n```\n\n"
            prompt += (post_setting_prompt + instruction_prompt)
            pos_id = str(eval(samples["pos_candidates"][i][0]["attributes"])["backend_node_id"]) 
            prompt += f"""[{pos_id}, '{samples["operation"][i]["op"]}', '{samples["operation"][i]["value"]}']""" if pos_id in html_chunk else "'None'"
            instructions.append(prompt)
    return instructions


def formatting_func(sample):
    output_texts = []
    for i in range(len(sample['confirmed_task'])):
        setting_prompt = (
            "Here is a web intelligence agent that interacts with webpage environments. "
            "It can get the next ACTION based on the user's TASK_GOAL, a piece of current HTML_CHUNK and a historical ACTION_HISTORY sequence recording the actions that have been performed. \n\nThe current HTML_CHUNK is: \n```\n"
        )
        post_setting_prompt = (
            f"""The TASK_GOAL is: 'You are at website {sample["website"][i]}. {sample["confirmed_task"][i]}'. \n"""
            f"""The ACTION_HISTORY trajectory is: {str(sample["previous_actions"][i])}. \n\n"""
        )
        instruction_prompt = (
            "Given HTML_CHUNK and TASK_GOAL and ACTION_HISTORY, What the web agent need to do is select an element from this HTML code to interact with, thus bringing the goal closer to completion. \n"
            "It should first determine if there are any elements in HTML_CHUNK that need to be interacted with, and if not, its output should be 'None'. \n"
            "However, if there is an element to interact with (such as clicking or typing or selecting), the agent needs to output an ACTION. \nAn ACTION is a list with following format: \n"
            "[<backend_node_id>, <operation>, <value>]\n"
            "<backend_node_id> should be the chosen element's backend_node_id in HTML.\n"
            "<operation> should be 'CLICK' or 'TYPE' or 'SELECT' for interacting with the chosen element.\n"
            "<value> should be '' if <operation> is 'CLICK', or it should be the specific text content for typing or selecting.\n\n"
            "Now based on all the information as above, the web agent's output is: " 
        )

        prompt = setting_prompt
        prompt += sample["cleaned_html"][i] + "\n```\n\n"
        prompt += (post_setting_prompt + instruction_prompt)
        prompt += f"""['{sample["backend_node_id"][i]}', '{sample["op"][i]}', '{sample["value"][i]}']""" if len(sample["backend_node_id"][i]) > 0 else "'None'"
        output_texts.append(prompt)

    return output_texts


def preprocess_sample(sample):
    prev_actions = []
    action_outputs = []
    action_html_outputs = []
    output_template = {"website": "", "cleaned_html": "", "confirmed_task": "", "previous_actions": "", 
                       "op": "", "value": "", "backend_node_id": ""}

    for action, action_repr in zip(sample["actions"], sample["action_reprs"]):
        if len(action["pos_candidates"]) > 0:
            output = output_template.copy()
            output["previous_actions"] = str(prev_actions)
            prev_actions.append(action_repr)
            output["website"] = sample["website"]
            output["confirmed_task"] = sample["confirmed_task"]
            output["cleaned_html"] = action["cleaned_html"]
            output["op"] = str(action["operation"]["op"])
            output["value"] = str(action["operation"]["value"])
            output["backend_node_id"] = str(eval(action["pos_candidates"][0]["attributes"])["backend_node_id"])
            action_outputs.append(output)
        else:
            continue

    for action_output in action_outputs:
        pos_id = action_output["backend_node_id"]
        chunks = get_html_chunk(action_output["cleaned_html"])
        for chunk in chunks:
            if len(chunk) > 50:
                action_html_output = action_output.copy()
                action_html_output["cleaned_html"] = chunk
                action_html_output["backend_node_id"] = pos_id if '"'+pos_id+'"' in chunk else ""
                action_html_output["op"] = action_output["op"] if '"'+pos_id+'"' in chunk else ""
                action_html_output["value"] = action_output["value"] if '"'+pos_id+'"' in chunk else ""
                rand = random.random()
                if action_html_output["backend_node_id"] or (len(action_html_output["backend_node_id"]) == 0 and rand > THROW_AWAY_RATIO):
                    action_html_outputs.append(action_html_output)
            else:
                continue
            
    return action_html_outputs


def transform_dataset():
    for filename in os.listdir(DATASET_FOLDER):
        output_samples = []
        if filename.endswith(".json"): 
            with open(os.path.join(DATASET_FOLDER, filename), "r") as f:
                print("Preprocessing datafile", filename)
                samples = json.load(f)

            sample_output = []
            for sample in tqdm(samples):
                sample_output = preprocess_sample(sample)
                output_samples += sample_output

        with open(os.path.join(TRANSFORMED_DATA_DIR, filename), "w") as file:
            for out in output_samples:
                json.dump(out, file)
                file.write('\n')


def train():
    # train_dataset = get_train_data_split(DATA_PATH, TRAIN_SPLIT_FILES)
    train_dataset = load_dataset(TRANSFORMED_DATA_DIR, data_files="*.json", split="train")
    
    bnb_config = BitsAndBytesConfig(load_in_4bit=LOAD_IN_4_BIT, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb_config, device_map="auto")
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1 

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    training_args = TrainingArguments(output_dir=OUTPUT_DIR, num_train_epochs=2, per_device_train_batch_size=1, gradient_accumulation_steps=16,
                                      learning_rate=LEARNING_RATE, logging_steps=LOGGING_STEPS, max_steps=1000, # optim="paged_adamw_32bit",#  max_grad_norm=0.3,
                                      logging_dir="./logs", save_strategy="steps", save_steps=50, # evaluation_strategy="steps", # fp16=False, bf16=False,
                                      warmup_ratio=0.03, report_to="tensorboard", do_eval=DO_EVAL)

    peft_config = LoraConfig(lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, r=64, bias="none", task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=base_model, train_dataset=train_dataset, peft_config=peft_config,
                         formatting_func=formatting_func, max_seq_length=MAX_SEQ_LENGTH,
                         tokenizer=tokenizer, args=training_args)
    
    trainer.train()
    # trainer.model.save_pretrained("llama2-7b-mind2web-2")


if __name__ == "__main__":
    # transform_dataset()
    train()