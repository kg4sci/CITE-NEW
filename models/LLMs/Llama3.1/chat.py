import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataname = 'chemistry'
data = torch.load(f"../../../datasets/pt/{dataname}.pt")
raw_texts = data.raw_texts
start_index = 0
index_list = list(range(start_index, len(raw_texts)))

toolkit_path = "" # path/to/toolkit/Forced-choice.json
with open(toolkit_path, 'r') as f:
    prompt_cfg = json.load(f)

label_set = "1. ENGINEERING; 2. MATERIALSSCIENCE; 3. PHYSICS; 4. CHEMISTRY; 5. COMPUTERSCIENCE; 6. MEDICINE; 7. AGRICULTURE; 8. MATHEMATICS; 9. PUBLIC; 10. GEOSCIENCES; 11. EDUCATION; 12. DENTISTRY; 13. RADIOLOGY; 14. HUMANITIES; 15. ELECTROCHEMISTRY; 16. NANOSCIENCE&NANOTECHNOLOGY; 17. ENVIRONMENTALSCIENCES; 18. ENERGY&FUELS; 19. METALLURGY&METALLURGICALENGINEERING; 20. GREEN&SUSTAINABLESCIENCE&TECHNOLOGY; 21. WATERRESOURCES; 22. POLYMERSCIENCE; 23. BIOPHYSICS; 24. BIOTECHNOLOGY&APPLIEDMICROBIOLOGY; 25. INSTRUMENTS&INSTRUMENTATION; 26. MULTIDISCIPLINARYSCIENCES; 27. BIOCHEMISTRY&MOLECULARBIOLOGY; 28. CRYSTALLOGRAPHY; 29. OPTICS; 30. SPECTROSCOPY; 31. BIOCHEMICALRESEARCHMETHODS; 32. FOODSCIENCE&TECHNOLOGY; 33. ACOUSTICS; 34. TOXICOLOGY; 35. THERMODYNAMICS; 36. METEOROLOGY&ATMOSPHERICSCIENCES; 37. MINERALOGY; 38. BIOLOGY; 39. NUCLEARSCIENCE&TECHNOLOGY; 40. MICROSCOPY; 41. PHARMACOLOGY&PHARMACY; 42. AGRICULTURALENGINEERING; 43. MECHANICS; 44. CONSTRUCTION&BUILDINGTECHNOLOGY; 45. MINING&MINERALPROCESSING; 46. MARINE&FRESHWATERBIOLOGY; 47. QUANTUMSCIENCE&TECHNOLOGY; 48. LIMNOLOGY; 49. MICROBIOLOGY; 50. NUTRITION&DIETETICS; 51. GEOCHEMISTRY&GEOPHYSICS; 52. ENVIRONMENTALSTUDIES; 53. PLANTSCIENCES; 54. MATHEMATICAL&COMPUTATIONALBIOLOGY; 55. AGRONOMY; 56. ENDOCRINOLOGY&METABOLISM; 57. TRANSPORTATIONSCIENCE&TECHNOLOGY; 58. SOILSCIENCE; 59. CELLBIOLOGY; 60. ONCOLOGY; 61. GENETICS&HEREDITY; 62. FORESTRY; 63. INFECTIOUSDISEASES; 64. IMMUNOLOGY; 65. MATHEMATICS; 66. ARCHAEOLOGY; 67. AUTOMATION&CONTROLSYSTEMS; 68. ASTRONOMY&ASTROPHYSICS; 69. ECOLOGY; 70. ART; 71. DERMATOLOGY; 72. TRANSPLANTATION; 73. HORTICULTURE; 74. VIROLOGY; 75. PHYSIOLOGY; 76. EVOLUTIONARYBIOLOGY; 77. MEDICALINFORMATICS; 78. ALLERGY; 79. ENTOLOGY; 80. GASTROENTEROLOGY&HEPATOLOGY; 81. ROBOTICS; 82. SURGERY; 83. ANTHROPOLOGY; 84. OCEANOGRAPHY; 85. VETERINARYSCIENCES; 86. NEUROSCIENCES; 87. INFORMATIONSCIENCE&LIBRARYSCIENCE; 88. ANATOMY&MORPHOLOGY; 89. INTEGRATIVE&COMPLEMENTARYMEDICINE; 90. INTERNATIONALRELATIONS; 91. STATISTICS&PROBABILITY; 92. LOGIC; 93. MYCOLOGY; 94. PARASITOLOGY; 95. ECONOMICS; 96. ARCHITECTURE; 97. TRANSPORTATION; 98. MEDICALLABORATORYTECHNOLOGY; 99. UROLOGY&NEPHROLOGY; 100. ZOOLOGY; 101. CLINICALNEUROLOGY; 102. CELL&TISSUEENGINEERING; 103. OPHTHALMOLOGY; 104. IMAGINGSCIENCE&PHOTOGRAPHICTECHNOLOGY; 105. TELECOMMUNICATIONS; 106. FISHERIES; 107. NOTHING; "

is_multiturn = "rounds" in prompt_cfg

batch_size = 8
model_name = ""  # /path/to/Llama-3.1-8B-Instruct
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"
tokenizer.model_max_length = 16000
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={0: "40GB"},
    offload_folder="offload",
)
model.config.gradient_checkpointing = True
model.eval()


def build_prompt_text(cfg, text, label_set, is_multiturn):
    if not is_multiturn:
        user_msg = cfg["user"].format(
            title=text,
            journal=getattr(data, 'journal', [''])[0] if hasattr(data, 'journal') else '',
            authors=getattr(data, 'authors', [''])[0] if hasattr(data, 'authors') else '',
            label_set=label_set,
        )
        system_msg = cfg.get("system", "")
        if system_msg:
            return f"SYSTEM: {system_msg}\nUSER: {user_msg}\nASSISTANT:"
        return f"USER: {user_msg}\nASSISTANT:"
    else:
        parts = []
        system_msg = cfg.get("system", "")
        if system_msg:
            parts.append(f"SYSTEM: {system_msg}")
        for idx, round_cfg in enumerate(cfg["rounds"]):
            user_msg = round_cfg["user"].format(
                title=text,
                abstract=getattr(data, 'abstract', [''])[0] if hasattr(data, 'abstract') else '',
                journal=getattr(data, 'journal', [''])[0] if hasattr(data, 'journal') else '',
                authors=getattr(data, 'authors', [''])[0] if hasattr(data, 'authors') else '',
                keywords=getattr(data, 'keywords', [''])[0] if hasattr(data, 'keywords') else '',
                label_set=label_set,
            )
            parts.append(f"USER: {user_msg}")
            if idx < len(cfg["rounds"]) - 1:
                parts.append("ASSISTANT: ")
        parts.append("ASSISTANT:")
        return "\n".join(parts)


def run_inference(prompts):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        do_sample=False,
        use_cache=True,
    )

    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


# DataLoader
pairs = list(zip(raw_texts, index_list))
data_loader = DataLoader(
    pairs,
    batch_size=batch_size,
    sampler=SequentialSampler(pairs),
    num_workers=4,
    pin_memory=True,
)

os.makedirs(f"responses/{dataname}", exist_ok=True)

for batch in tqdm(data_loader):
    text_batch, index_batch = batch[0], batch[1]

    prompts = [build_prompt_text(prompt_cfg, text, label_set, is_multiturn) for text in text_batch]
    answers = run_inference(prompts)

    for idx, answer in zip(index_batch, answers):
        with open(f"responses/{dataname}/{idx}.json", 'w') as f:
            json.dump({"id": int(idx), "answer": answer}, f)

    torch.cuda.empty_cache()
