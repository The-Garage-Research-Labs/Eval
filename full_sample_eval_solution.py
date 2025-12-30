import os
from dotenv import load_dotenv

load_dotenv()
# move to the directory where src is located
os.chdir('/home/abdo/PAPER/Eval/src')

# Setting an API key for NVIDIA 
os.environ['NVIDIA_API_KEY'] = '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'

from html_eval import Experiment
from html_eval.configs.experiment_config import ExperimentConfig
from html_eval.configs.dataset_config import SWDEConfig , WebSrcConfig
from html_eval.configs.pipeline_config import RerankerPipelineConfig , RerankerPreprocessorConfig , RerankerExtractorConfig , RerankerPostprocessorConfig
from html_eval.configs.llm_client_config import LLMClientConfig


TOTAL_GPU_UTIL = 0.9
GEN_GPU_UTIL = 0.2

import multiprocessing as mp
    
# The goal is to answer the following question (Does the reranker provide any kind of help?)
# Things we need to iterate on
# SWDE verticals / WEDBSRC
# Disable Method (chunking/reranker)
######################################## CONSTANTS
# EXPERIMENT_NAME = "pruner_best_model"
# EXPERIMENT_NAME = "pruner_trying"
EXPERIMENT_NAME = "pruner_abalationv2_final"

SWDE_DOMAINS = {
    "auto": 17923,
    "university": 16705,
    "camera": 5258,
    "book": 20000,
    "job": 20000,
    "nbaplayer": 4405,
    "movie": 20000,
    "restaurant": 20000
}
SWDE_SAMPLES = 500
WEBSRC_TOTAL = 50000
WEBSRC_SAMPLES = 5000
BATCH_SIZE = 100
SEED = 42
USE_PRUNER = True  
########################################## CONFIG

dataset_configs = []

for dom , val in SWDE_DOMAINS.items():
    dataset_configs.append(SWDEConfig(
        local_dir="/home/abdo/PAPER/Eval/data/swde/hf_SWDE",
        indices=list(range(0,val,int(val/SWDE_SAMPLES))),  # Use a subset of the dataset for
        domain=dom,
        batch_size=BATCH_SIZE
    ))



dataset_configs.append(WebSrcConfig(
    html_source_path='/home/abdo/PAPER/Eval/data/websrc/hf_websrc_dev/dev/dev_html_content.jsonl',
    data_source_path='/home/abdo/PAPER/Eval/data/websrc/hf_websrc_dev/dev/dev_dataset.jsonl',
    indices= list(range(0,WEBSRC_TOTAL,int(WEBSRC_TOTAL/WEBSRC_SAMPLES))),
    batch_size=BATCH_SIZE
))
print(f"Created {len(dataset_configs)} of dataset configs.")

reranker_preprocessor_config = RerankerPreprocessorConfig(
    attr_cutoff_len=5,
    # chunk_size=100000000, #500 is best after finetuning
    chunk_size=2000, 
    fetch_workers=mp.cpu_count(),
    cpu_workers=mp.cpu_count()
)

# llm_client_config = LLMClientConfig(
#     # model_name='google/gemma-3n-e4b-it',
#     # model_name='google/gemma-3n-e2b-it',
#     model_name='qwen/qwen3-coder-480b-a35b-instruct',
#     # model_name='mistralai/mistral-large-3-675b-instruct-2512',
#     # model_name='numind/NuExtract-1.5',
#     # model_name="moonshotai/kimi-k2-instruct-0905",
#     # model_name='openai/gpt-oss-120b',
#     # model_name='qwen/qwen3-235b-a22b',
# )

llm_client_config = LLMClientConfig(
        # model_name="openai/gpt-oss-120b",
        llm_source='vllm',
        model_name='Qwen/Qwen3-0.6B',
        seed=SEED,
        # api_key="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",
        engine_args={
            "gpu_memory_utilization": 0.7,
            "max_model_len": 8196,
            # "enforce_eager": True,
        },
        # temperature=1.0,
        lora_modules={
            # "pruner": "abdo-Mansour/Pruner_Adaptor_Qwen_3_r64_n",
            "pruner": "abdo-Mansour/Pruner_Adaptor_Qwen_3_FINAL",
            "extractor": "abdo-Mansour/Extractor_Adaptor_Qwen3_Final"
        },
) 

reranker_extractor_config = RerankerExtractorConfig(
    same_llm_config= True,
    llm_config=llm_client_config,
    llm_pruner_config=LLMClientConfig(
        # model_name='google/gemma-3n-e4b-it',
        model_name='qwen/qwen3-coder-480b-a35b-instruct',
        # model_name="moonshotai/kimi-k2-instruct-0905",
        # model_name='nvidia/nemotron-3-nano-30b-a3b',
        # model_name='mistralai/mistral-large-3-675b-instruct-2512',
        # model_name='openai/gpt-oss-120b',
        # model_name='qwen/qwen3-235b-a22b',
    ),
#     llm_pruner_config=LLMClientConfig(
#         # model_name="openai/gpt-oss-120b",
#         llm_source='vllm',
#         model_name='Qwen/Qwen3-0.6B',
#         seed=SEED,
#         # api_key="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",
#         engine_args={
#             "gpu_memory_utilization": 0.7,
#             "max_model_len": 6096,
#             # "enforce_eager": True,
#         },
#         # temperature=0.5,
#         lora_modules={
#             # "pruner": "abdo-Mansour/Pruner_Adaptor_Qwen_3_r64_n",
#             "pruner": "abdo-Mansour/Pruner_Adaptor_Qwen_3_FINAL",
#             "extractor": "abdo-Mansour/Extractor_Adaptor_Qwen3_Final"
#         },
# ) ,
    query_generation_prompt_template="""
    You are a highly precise Context-Aware Question Answering engine. Your sole task is to extract the answer to the User Query based ONLY on the provided Context.
    
    User Query:
    {query}
    
    Context:
    {content}
    
    INSTRUCTIONS:
    1. Answer the query using ONLY information found in the Context. Do not use outside knowledge.
    2. If the answer is not present in the Context, set the value to null.
    3. Your output must be valid, parseable JSON.
    4. Provide concise answers without additional commentary.
    5. If the query is boolean, respond with yes or no.
    6. Choose the most relevant information if multiple answers exist.
    7. THE ANSWER IS PRETTY SIMPLE JUST OUTPUT IT.

    OUTPUT FORMAT:
    REASONING: "The reasoning behind the answer"
    {{"answer": "The extracted text or synthesized answer"}}
    
    OUTPUT THE ANSWER INSIDE A JSON STRUCTURE
    OUTPUT THE ANSWER INSIDE A JSON STRUCTURE
    OUTPUT THE ANSWER INSIDE A JSON STRUCTURE
    OUTPUT THE ANSWER INSIDE A JSON STRUCTURE
    OUTPUT THE ANSWER INSIDE A JSON STRUCTURE
    OUTPUT THE ANSWER INSIDE A JSON STRUCTURE
    OUTPUT THE ANSWER INSIDE A JSON STRUCTURE
    OUTPUT THE ANSWER INSIDE A JSON STRUCTURE

""",
    schema_generation_prompt_template="""
    You are an expert Data Extraction and ETL agent. Your task is to parse the provided HTML content and extract specific data points to populate a target JSON schema.
    
    Target Schema Structure:
    {query}
    
    HTML Content:
    {content}
    
    RULES:
    1. Extract exact substrings from the text content of the HTML. Do not invent data.
    2. Ignore HTML tags, attributes, and styles; extract only the visible text value.
    3. If a specific field from the schema is not found in the content, set its value to null.
    4. Ensure the output strictly follows the keys defined in the "Target Schema Structure".
    5. Your output MUST be exactly as shown in the HTML.
    6. Be concise and avoid adding any extra information outside the schema.
    
    OUTPUT JSON:
""",
    classification_prompt_template= (
            "You are a precision HTML content reranker. Your task is to evaluate HTML chunks "
            "for their potential to populate a given schema with meaningful data.\n\n"
            "## Core Objective:\n"
            "Score HTML content based on its likelihood to contain extractable information "
            "that matches the target schema requirements.\n\n"
            "## Instructions:\n"
            "1. Content Analysis: Examine the HTML chunk's text content, attributes, and semantic structure\n"
            "2. Schema Mapping: Assess how well the content aligns with schema field requirements\n"
            "3. Information Density: Evaluate the quantity and quality of extractable data\n"
            "4. Relevance Scoring: Assign a binary relevance score based on extraction potential\n"
        ),
    reranker_quantization=None,
    llm_pruner_prompt="""
You are a Smart and Clever Context Selector. Your task is to filter a list of HTML chunks, keeping ONLY the ones relevant to the provided Query/Schema and any necessary context to answer the query.

Query/Schema:
{query}

**INSTRUCTIONS:**

1.  **Analyze the Query:** Determine exactly what data is being requested. It could be specific content (prices, dates), structural elements (menu items, footers), or broad sections.
2.  **Select Relevant Chunks:** Identify chunks that contain:
    *   The **Direct Answer** (values, text, list items).
    *   Essential **Labels/Context** (e.g., the text "Price:" next to "$10.00").
    *   **Atomic Containers** (tables, lists) that hold the requested data.
3. **Select Context Carefully:** Only include chunks that are necessary to understand or locate the answer. Avoid including unrelated sections.
4.  **Discard Noise:** Remove any chunks that do not contribute to answering *this specific query*.
5.  **Handle Missing Data:** If no chunks contain the requested information, return an empty list `[]`.
6. **Include Supporting Context:** When relevant, include chunks that provide necessary context to understand or locate the answer, even if they don't contain the direct answer themselves.
7. **Table Handling:** If the query relates to tabular data, prioritize chunks that represent entire rows or columns relevant to the schema.
8. **Flow**: Ensure the selected chunks form a coherent context for answering the query.

**Content:**
{content}

**Response Format:**
Output ONLY a valid JSON list of indices.
Example: [1, 4, 12] or []
""",
    disable_reranker=USE_PRUNER,
    use_llm_pruner=USE_PRUNER,
    # reranker_gpu_memory_utilization=TOTAL_GPU_UTIL - GEN_GPU_UTIL,
)



########################################## RUN
full_results = {}
page_level_f1 = 0
token_level_f1 = 0

# 1. Define Pipeline Config ONCE outside the loop.
# We want to keep the model loaded in memory, so we shouldn't recreate this config.
reranker_postprocessor_config = RerankerPostprocessorConfig(
    exact_extraction=True
)

reranker_pipeline_config = RerankerPipelineConfig(
    preprocessor_config=reranker_preprocessor_config,
    extractor_config=reranker_extractor_config,
    postprocessor_config=reranker_postprocessor_config,
    # disable_method=method_status 
)

# Variable to store the single persistent experiment instance
exp = None

for i, data_cfg in enumerate(dataset_configs):
    # Determine the experiment/output name based on the dataset
    name = "swde_" if isinstance(data_cfg, SWDEConfig) else "websrc"
    name += data_cfg.domain if isinstance(data_cfg, SWDEConfig) else ""
    
    print(f"\n[Manager] Preparing to run: {name}")

    if exp is None:
        # --- FIRST ITERATION: INITIALIZE ---
        # Create the Experiment object for the first time.
        # This triggers the heavy lifting (loading VLLM/LLM into GPU).
        exp_config = ExperimentConfig(
            experiment_name=name,
            seed=SEED,
            pipeline_config=reranker_pipeline_config,
            dataset_config=data_cfg,
            output_dir=name
        )
        exp = Experiment(exp_config, resume=False)
    else:
        # --- SUBSEQUENT ITERATIONS: SWAP ---
        # Swap the dataset and output directory.
        # This keeps the Pipeline (and VLLM) alive and just changes the data flow.
        exp.swap_dataset(new_dataset_config=data_cfg, new_experiment_name=name, new_output_dir=name)
        
        # Optional: Update internal experiment name if your Tracker (WandB) uses it
        exp._config.experiment_name = name 

    # Run the experiment on the current dataset
    try:
        results = exp.run()
        full_results[name] = results

        # Aggregate metrics
        if 'page_level_f1' in results["results"]:
            page_level_f1 += results['results']['page_level_f1']['f1']
        elif 'token_f1' in results["results"]:
            token_level_f1 += results['results']['token_f1']['f1']

    except Exception as e:
        print(f"[Error] Failed to run experiment for {name}: {e}")
        import traceback
        traceback.print_exc()

    # # --- MEMORY MANAGEMENT ---
    # # We DO NOT delete 'exp' because we want to reuse the LLM.
    # # We only run garbage collection to clear previous dataset samples from RAM.
    # import gc
    
    # # Clear Python garbage (dataset samples, intermediate strings)
    # gc.collect()



print("======== Final Results ========")
print(f"Average Page Level F1: {page_level_f1 / len(SWDE_DOMAINS.keys())}")
print(f"Average Token Level F1: {token_level_f1}")

full_results['average_page_level_f1'] = page_level_f1 / len(SWDE_DOMAINS.keys())
full_results['average_token_level_f1'] = token_level_f1 
# save full results to a json file
import json
with open(f"{EXPERIMENT_NAME}.json", "w") as f:
    json.dump(full_results, f, indent=4)
