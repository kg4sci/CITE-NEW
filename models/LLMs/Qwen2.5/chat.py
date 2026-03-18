import os
import json
import torch
import requests
from tqdm import tqdm

dataname = 'chemistry'
data = torch.load(f"../../../datasets/pt/{dataname}.pt")
raw_texts = data.raw_texts
start_index = 0
index_list = list(range(start_index, len(raw_texts)))

toolkit_path = ""# path/to/toolkit/Forced-choice.json
with open(toolkit_path, 'r') as f:
    prompt_cfg = json.load(f)

label_set = "1. ENGINEERING; 2. MATERIALSSCIENCE; 3. PHYSICS; 4. CHEMISTRY; 5. COMPUTERSCIENCE; 6. MEDICINE; 7. AGRICULTURE; 8. MATHEMATICS; 9. PUBLIC; 10. GEOSCIENCES; 11. EDUCATION; 12. DENTISTRY; 13. RADIOLOGY; 14. HUMANITIES; 15. ELECTROCHEMISTRY; 16. NANOSCIENCE&NANOTECHNOLOGY; 17. ENVIRONMENTALSCIENCES; 18. ENERGY&FUELS; 19. METALLURGY&METALLURGICALENGINEERING; 20. GREEN&SUSTAINABLESCIENCE&TECHNOLOGY; 21. WATERRESOURCES; 22. POLYMERSCIENCE; 23. BIOPHYSICS; 24. BIOTECHNOLOGY&APPLIEDMICROBIOLOGY; 25. INSTRUMENTS&INSTRUMENTATION; 26. MULTIDISCIPLINARYSCIENCES; 27. BIOCHEMISTRY&MOLECULARBIOLOGY; 28. CRYSTALLOGRAPHY; 29. OPTICS; 30. SPECTROSCOPY; 31. BIOCHEMICALRESEARCHMETHODS; 32. FOODSCIENCE&TECHNOLOGY; 33. ACOUSTICS; 34. TOXICOLOGY; 35. THERMODYNAMICS; 36. METEOROLOGY&ATMOSPHERICSCIENCES; 37. MINERALOGY; 38. BIOLOGY; 39. NUCLEARSCIENCE&TECHNOLOGY; 40. MICROSCOPY; 41. PHARMACOLOGY&PHARMACY; 42. AGRICULTURALENGINEERING; 43. MECHANICS; 44. CONSTRUCTION&BUILDINGTECHNOLOGY; 45. MINING&MINERALPROCESSING; 46. MARINE&FRESHWATERBIOLOGY; 47. QUANTUMSCIENCE&TECHNOLOGY; 48. LIMNOLOGY; 49. MICROBIOLOGY; 50. NUTRITION&DIETETICS; 51. GEOCHEMISTRY&GEOPHYSICS; 52. ENVIRONMENTALSTUDIES; 53. PLANTSCIENCES; 54. MATHEMATICAL&COMPUTATIONALBIOLOGY; 55. AGRONOMY; 56. ENDOCRINOLOGY&METABOLISM; 57. TRANSPORTATIONSCIENCE&TECHNOLOGY; 58. SOILSCIENCE; 59. CELLBIOLOGY; 60. ONCOLOGY; 61. GENETICS&HEREDITY; 62. FORESTRY; 63. INFECTIOUSDISEASES; 64. IMMUNOLOGY; 65. MATHEMATICS; 66. ARCHAEOLOGY; 67. AUTOMATION&CONTROLSYSTEMS; 68. ASTRONOMY&ASTROPHYSICS; 69. ECOLOGY; 70. ART; 71. DERMATOLOGY; 72. TRANSPLANTATION; 73. HORTICULTURE; 74. VIROLOGY; 75. PHYSIOLOGY; 76. EVOLUTIONARYBIOLOGY; 77. MEDICALINFORMATICS; 78. ALLERGY; 79. ENTOLOGY; 80. GASTROENTEROLOGY&HEPATOLOGY; 81. ROBOTICS; 82. SURGERY; 83. ANTHROPOLOGY; 84. OCEANOGRAPHY; 85. VETERINARYSCIENCES; 86. NEUROSCIENCES; 87. INFORMATIONSCIENCE&LIBRARYSCIENCE; 88. ANATOMY&MORPHOLOGY; 89. INTEGRATIVE&COMPLEMENTARYMEDICINE; 90. INTERNATIONALRELATIONS; 91. STATISTICS&PROBABILITY; 92. LOGIC; 93. MYCOLOGY; 94. PARASITOLOGY; 95. ECONOMICS; 96. ARCHITECTURE; 97. TRANSPORTATION; 98. MEDICALLABORATORYTECHNOLOGY; 99. UROLOGY&NEPHROLOGY; 100. ZOOLOGY; 101. CLINICALNEUROLOGY; 102. CELL&TISSUEENGINEERING; 103. OPHTHALMOLOGY; 104. IMAGINGSCIENCE&PHOTOGRAPHICTECHNOLOGY; 105. TELECOMMUNICATIONS; 106. FISHERIES; 107. NOTHING; "

is_multiturn = "rounds" in prompt_cfg

API_URL = os.environ.get("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1") + "/chat/completions"
API_KEY = os.environ.get("QWEN_API_KEY", "your-api-key")
API_MODEL = "qwen2.5-72b-instruct"
API_TIMEOUT = 60


def build_single_turn_messages(cfg, text, label_set):
    user_msg = cfg["user"].format(
        title=text,
        journal=getattr(data, 'journal', [''])[0] if hasattr(data, 'journal') else '',
        authors=getattr(data, 'authors', [''])[0] if hasattr(data, 'authors') else '',
        label_set=label_set,
    )
    messages = []
    if cfg.get("system"):
        messages.append({"role": "system", "content": cfg["system"]})
    messages.append({"role": "user", "content": user_msg})
    return messages

def call_api(messages):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "model": API_MODEL,
        "messages": messages,
        "max_tokens": 64,
        "temperature": 0.0,
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=API_TIMEOUT)
        if response.status_code == 200:
            rj = response.json()
            if "choices" in rj and rj["choices"]:
                return rj["choices"][0]["message"]["content"].strip()
            else:
                raise KeyError(f"Unexpected API response format: {json.dumps(rj, ensure_ascii=False)}")
        else:
            raise RuntimeError(f"API HTTP {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Error while calling API: {e}")
        return ""


os.makedirs(f"responses/{dataname}", exist_ok=True)

for idx, text in tqdm(zip(index_list, raw_texts), total=len(index_list)):
    try:
        if not is_multiturn:
            messages = build_single_turn_messages(prompt_cfg, text, label_set)
            answer = call_api(messages)
        else:
            messages = []
            answer = ""
            for round_cfg in prompt_cfg["rounds"]:
                user_msg = round_cfg["user"].format(
                    title=text,
                    abstract=getattr(data, 'abstract', [''])[0] if hasattr(data, 'abstract') else '',
                    journal=getattr(data, 'journal', [''])[0] if hasattr(data, 'journal') else '',
                    authors=getattr(data, 'authors', [''])[0] if hasattr(data, 'authors') else '',
                    keywords=getattr(data, 'keywords', [''])[0] if hasattr(data, 'keywords') else '',
                    label_set=label_set,
                )
                messages.append({"role": "user", "content": user_msg})
                answer = call_api(messages)
                messages.append({"role": "assistant", "content": answer})

        with open(f"responses/{dataname}/{idx}.json", 'w') as f:
            json.dump({"id": int(idx), "answer": answer}, f)

    except Exception as e:
        print(f"Error at index {idx}: {e}")
        with open(f"responses/{dataname}/{idx}.json", 'w') as f:
            json.dump({"id": int(idx), "answer": "", "error": str(e)}, f)
