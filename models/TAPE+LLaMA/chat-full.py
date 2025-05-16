import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"


# 数据加载
dataname = 'chemistry_fullhomo' #<class 'dict'>
# dataname = 'arxiv' #<class 'torch_geometric.data.data.Data'>
data = torch.load(f"../../../datasets/{dataname}.pt")
print(type(data)) 
raw_texts = data.raw_texts
start_index = 110000  # Or any index where you want to start processing
# Modify the index_list to start from the specified start_index
index_list = list(range(start_index, len(raw_texts)))
# index_list = list(range(len(raw_texts)))

# 选择 prompt
if dataname == 'cora':
    prompt = "\n Question: Which of the following sub-categories of AI does this paper belong to: Case Based, Genetic Algorithms, Neural Networks, Probabilistic Methods, Reinforcement Learning, Rule Learning, Theory? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: "
elif dataname == 'pubmed':
    prompt = "\n Question: Does the paper involve any cases of Type 1 diabetes, Type 2 diabetes, or Experimentally induced diabetes? Please give one or more answers of either Type 1 diabetes, Type 2 diabetes, or Experimentally induced diabetes; if multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, give a detailed explanation with quotes from the text explaining why it is related to the chosen option. \n \n Answer: "
elif dataname == 'arxiv':
    prompt = "\n Question: Which arXiv CS subcategory does this paper belong to? Give 5 likely arXiv CS sub-categories as a comma-separated list ordered from most to least likely, in the form “cs.XX”, and provide your reasoning. \n \n Answer:"
elif dataname == 'citeseer':
    prompt = "\n Question: Which of the following sub-categories of CS does this paper belong to: Agents, ML (Machine Learning), IR (Information Retrieval), DB (Databases), HCI (Human-Computer Interaction), AI (Artificial Intelligence)? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: "
elif dataname == 'wikics':
    prompt = "\n Question: Which of the following sub-categories of CS does this Wikipedia page belong to: Computational Linguistics, Databases, Operating Systems, Computer Architecture, Computer Security, Internet Protocols, Computer File Systems, Distributed Computing Architecture, Web Technology, Programming Language Topics? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: "
elif dataname == 'instagram':
    prompt = "\n Question: Which of the following categories does this user on Instagram belong to:  Normal Users, Commercial Users? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: "
elif dataname == 'reddit':
    prompt = "\n Question: Which of the following categories does this user on Reddit belong to:  Normal Users, Popular Users? If multiple options apply, provide a comma-separated list ordered from most to least related, then for each choice you gave, explain how it is present in the text.\n\nAnswer: "
elif dataname == 'chemistry_fullhomo':
    prompt = "\n Question: Which of the following categories does this paper belong to: ENGINEERING,MATERIALSSCIENCE,PHYSICS,CHEMISTRY,COMPUTERSCIENCE,MEDICINE,AGRICULTURE,MATHEMATICS,PUBLIC,GEOSCIENCES,EDUCATION,DENTISTRY,RADIOLOGY,HUMANITIES,ELECTROCHEMISTRY,NANOSCIENCE&NANOTECHNOLOGY,ENVIRONMENTALSCIENCES,ENERGY&FUELS,METALLURGY&METALLURGICALENGINEERING,GREEN&SUSTAINABLESCIENCE&TECHNOLOGY,WATERRESOURCES,POLYMERSCIENCE,BIOPHYSICS,BIOTECHNOLOGY&APPLIEDMICROBIOLOGY,INSTRUMENTS&INSTRUMENTATION,MULTIDISCIPLINARYSCIENCES,BIOCHEMISTRY&MOLECULARBIOLOGY,CRYSTALLOGRAPHY,OPTICS,SPECTROSCOPY,BIOCHEMICALRESEARCHMETHODS,FOODSCIENCE&TECHNOLOGY,ACOUSTICS,TOXICOLOGY,THERMODYNAMICS,METEOROLOGY&ATMOSPHERICSCIENCES,MINERALOGY,BIOLOGY,NUCLEARSCIENCE&TECHNOLOGY,MICROSCOPY,PHARMACOLOGY&PHARMACY,AGRICULTURALENGINEERING,MECHANICS,CONSTRUCTION&BUILDINGTECHNOLOGY,MINING&MINERALPROCESSING,MARINE&FRESHWATERBIOLOGY,QUANTUMSCIENCE&TECHNOLOGY,LIMNOLOGY,MICROBIOLOGY,NUTRITION&DIETETICS,GEOCHEMISTRY&GEOPHYSICS,ENVIRONMENTALSTUDIES,PLANTSCIENCES,MATHEMATICAL&COMPUTATIONALBIOLOGY,AGRONOMY,ENDOCRINOLOGY&METABOLISM,TRANSPORTATIONSCIENCE&TECHNOLOGY,SOILSCIENCE,CELLBIOLOGY,ONCOLOGY,GENETICS&HEREDITY,FORESTRY,INFECTIOUSDISEASES,IMMUNOLOGY,MATHEMATICS,ARCHAEOLOGY,AUTOMATION&CONTROLSYSTEMS,ASTRONOMY&ASTROPHYSICS,ECOLOGY,ART,DERMATOLOGY,TRANSPLANTATION,HORTICULTURE,VIROLOGY,PHYSIOLOGY,EVOLUTIONARYBIOLOGY,MEDICALINFORMATICS,ALLERGY,ENTOMOLOGY,GASTROENTEROLOGY&HEPATOLOGY,ROBOTICS,SURGERY,ANTHROPOLOGY,OCEANOGRAPHY,VETERINARYSCIENCES,NEUROSCIENCES,INFORMATIONSCIENCE&LIBRARYSCIENCE,ANATOMY&MORPHOLOGY,INTEGRATIVE&COMPLEMENTARYMEDICINE,INTERNATIONALRELATIONS,STATISTICS&PROBABILITY,LOGIC,MYCOLOGY,PARASITOLOGY,ECONOMICS,ARCHITECTURE,TRANSPORTATION,MEDICALLABORATORYTECHNOLOGY,UROLOGY&NEPHROLOGY,ZOOLOGY,CLINICALNEUROLOGY,CELL&TISSUEENGINEERING,OPHTHALMOLOGY,IMAGINGSCIENCE&PHOTOGRAPHICTECHNOLOGY,TELECOMMUNICATIONS,FISHERIES,NOTHING"

# 批次大小，可根据 GPU 显存再调大
batch_size = 8

# DataLoader
# data_loader = DataLoader(
#     list(zip(raw_texts, index_list)),
#     batch_size=batch_size,
#     sampler=SequentialSampler(list(zip(raw_texts, index_list)))
# )
data_loader = DataLoader(
    # list(zip(raw_texts, index_list)),
    list(zip(raw_texts[start_index:], index_list)),
    batch_size=batch_size,
    sampler=SequentialSampler(list(zip(raw_texts, index_list))),
    num_workers=4,  # Increase number of workers to speed up data loading
    pin_memory=True,  # Helps in faster data loading
)


# 初始化 tokenizer
model_name = "./data/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
tokenizer.pad_token = tokenizer.bos_token

# 加载模型并并行化
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto",  # 自动把模型层分布到 GPU3/GPU4
    max_memory={0: "40GB", 1: "40GB",2: "40GB"},  # 明确每卡显存上限
    # max_memory={0: "60GB"},
    offload_folder="offload",  # 添加 offload_folder，允许将部分模型参数移到 CPU 上
)
model.config.gradient_checkpointing = True
model.eval()  # 推理模式

# 推理
for batch in tqdm(data_loader):
    text_batch, index_batch = batch[0], batch[1]
    batch_prompts = [text + prompt for text in text_batch]
    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)


    # DataParallel 下要用 module.generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,    # 限制生成长度，越短越快
        do_sample=False,      # 关闭采样，用最简单的 greedy
        use_cache=True,       # 启用缓存加速
    )

    answers = [
        tokenizer.decode(output, skip_special_tokens=True)
        for output in outputs
    ]



    # 保存结果
    for idx, answer in zip(index_batch, answers):
        os.makedirs(f"llama_response/{dataname}", exist_ok=True)
        with open(f"llama_response/{dataname}/{idx}.json", 'w') as f:
            json.dump({"answer": answer}, f)
    # Clear GPU memory after each batch
    torch.cuda.empty_cache()  # This frees up unused GPU memory
