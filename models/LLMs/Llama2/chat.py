import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"

dataname = 'chemistry'
data = torch.load(f"../../../datasets/pt/{dataname}.pt")
raw_texts = data.raw_texts
start_index = 0
index_list = list(range(start_index, len(raw_texts)))

toolkit_path = "../toolkit/Forced-choice.json"
with open(toolkit_path, 'r') as f:
    prompt_cfg = json.load(f)

label_set = "1 = cs.LG, 2 = cs.CV, ..."

is_multiturn = "rounds" in prompt_cfg


def build_single_turn_prompt(cfg, text, label_set):
    user_msg = cfg["user"].format(
        title=text,
        journal=getattr(data, 'journal', [''])[0] if hasattr(data, 'journal') else '',
        authors=getattr(data, 'authors', [''])[0] if hasattr(data, 'authors') else '',
        label_set=label_set,
    )
    system = cfg.get("system", "")
    if system:
        return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user_msg} [/INST]"
    else:
        return f"<s>[INST] {user_msg} [/INST]"


def build_multiturn_prompt(cfg, text, label_set):
    rounds = cfg["rounds"]
    prompts = []
    history = "<s>"
    for i, round_cfg in enumerate(rounds):
        user_msg = round_cfg["user"].format(
            title=text,
            abstract=getattr(data, 'abstract', [''])[0] if hasattr(data, 'abstract') else '',
            journal=getattr(data, 'journal', [''])[0] if hasattr(data, 'journal') else '',
            authors=getattr(data, 'authors', [''])[0] if hasattr(data, 'authors') else '',
            keywords=getattr(data, 'keywords', [''])[0] if hasattr(data, 'keywords') else '',
            label_set=label_set,
        )
        history += f"[INST] {user_msg} [/INST]"
        prompts.append(history)
        history += " {reply} </s><s>"
    return prompts


batch_size = 8

model_name = "./data/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
tokenizer.pad_token = tokenizer.bos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={0: "40GB", 1: "40GB", 2: "40GB"},
    offload_folder="offload",
)
model.config.gradient_checkpointing = True
model.eval()


def generate_batch(batch_prompts):
    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        use_cache=True,
    )
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]


data_loader = DataLoader(
    list(zip(raw_texts, index_list)),
    batch_size=batch_size,
    sampler=SequentialSampler(list(zip(raw_texts, index_list)))
)

for batch in tqdm(data_loader):
    text_batch, index_batch = batch[0], batch[1]

    if not is_multiturn:
        batch_prompts = [build_single_turn_prompt(prompt_cfg, text, label_set) for text in text_batch]
        answers = generate_batch(batch_prompts)
    else:
        num_rounds = len(prompt_cfg["rounds"])
        histories = [build_multiturn_prompt(prompt_cfg, text, label_set) for text in text_batch]
        final_answers = [""] * len(text_batch)
        round_replies = [""] * len(text_batch)

        for r in range(num_rounds):
            batch_prompts = [histories[i][r].format(reply=round_replies[i]) if r > 0
                             else histories[i][r]
                             for i in range(len(text_batch))]
            replies = generate_batch(batch_prompts)
            for i, reply in enumerate(replies):
                round_replies[i] = reply
                if r + 1 < num_rounds:
                    histories[i][r + 1] = histories[i][r + 1].replace("{reply}", reply, 1)
            final_answers = replies
        answers = final_answers

    for idx, answer in zip(index_batch, answers):
        os.makedirs(f"llama_response/{dataname}", exist_ok=True)
        with open(f"llama_response/{dataname}/{idx}.json", 'w') as f:
            json.dump({"answer": answer}, f)

    torch.cuda.empty_cache()
