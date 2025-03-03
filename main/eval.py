import infinity_embedded as infinity
from infinity_embedded.common import ConflictType
from transformers import AutoModel,AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import pandas as pd
from tqdm import tqdm
import time as t
import logging
import os
import streamlit as st
from PIL import Image
from contextlib import contextmanager
import re

from utils import paper_only_author,paper_only_content,paper_author_content,text_to_image
from utils import chunk_hybid_search,chunk_hybid_search_with_paper,chunk_filter_paper,add_citations
from llm_api import deepseek_judge_rag,deepseek_daily_qa,kimi_decompose, deepseek_chat,deepseek_chat_refine,zhipu_eval
from llm_api import deepseek_judge_pc,kimi_paper_col,deepseek_rewrite,zhipu_image_query
from llm_api import deepseek_single_query,deepseek_sub_queries,deepseek_multi_step_queries

from datasets import Dataset 
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity
from ragas import evaluate


# 下面是整体的流程
# 检索流程
def retrieve(query, model, db_object):
    """
    输入：用户的查询，向量模型，数据库，检索结果的保存名
    输出：问题类型，检索结果
    """
    # 改写query
    try:
        rewrite = deepseek_rewrite(query,[])
        rewrite = json.loads(rewrite).get("professional_query","")
        logging.info(f"正在改写查询，改写后的查询：{rewrite}")
        logging.info(f"改写成功，使用改写后的查询执行chunk混合检索和以文搜图!")
        res,image_res,table_res = chunk_hybid_search(db_object, model, rewrite, 6, 0.7) # 返回完整的chunk，图，表df
        
    except Exception as e:
        print(f"改写失败，使用原始查询执行chunk混合检索和以文搜图!\n报错：{e}")
        res,image_res,table_res = chunk_hybid_search(db_object, model, query, 6, 0.7) # df

    #image_res_1 = text_to_image(db_object, query, model) # 以文搜图，一张
    #image_res = pd.concat([image_res,image_res_1]) # 合并

    
    logging.info(f"保存检索结果完成，共计{len(res)}条记录")
    
    return res,image_res

# 生成流程
def generate(query,res,image_res,db_object):
    """
    输入：用户的查询，问题类型，检索结果, 子问题（对于非单一问题）,chunk相关的图片表结果，表格表结果
    输出：生成的回复
    """
    
    logging.info("开始生成回复！")
    context = res['restored_content'].tolist() # 上下文
    # 检索论文数据作为参考文献来源
    paper_uuids = res["paper_uuid"].drop_duplicates().tolist() # 论文id
    paper_res = chunk_filter_paper(db_object,paper_uuids)
    # 根据论文id合并论文表和chunk表，以chunk表为基准
    merged_res = pd.merge(res, paper_res[['paper_uuid', 'title', 'author', 'file_path']], on="paper_uuid", how="left")
    merged_res['reference'] = merged_res['title'].astype(str) + ', ' + merged_res['author'].astype(str) # 添加对应的论文参考信息
    # 组织成检索结果1,2,3等
    logging.info(f"组织上下文为['检索结果1':content]的格式。")
    contexts = []
    for idx,cont in enumerate(context):
        contexts.append(f"检索结果{idx+1}: {cont}")
    # 找到chunk的图片和表格，数据库检索
    images = image_res["image_path"].tolist()
    # tables = table_res["table_content"].tolist()
    # 图文的，模型不行，不用
    # res = zhipu_image_query(query,contexts,image)
    # 根据上下文信息的无引用回复,纯文本的
    res = deepseek_chat(query, contexts, messages)

     # 定义正则表达式检测公式和表格
    formula_pattern = re.compile(r'\$[^$]*\$|\$\$[^$]*\$\$')  # 检测$...$或$$...$$
    table_pattern = re.compile(r'\|.*\|')  # 检测包含|的行，通常用于表格
    # 检查是否包含公式或表格
    if formula_pattern.search(res) or table_pattern.search(res):
        # 润色，使md正常
        res = deepseek_chat_refine(res)
    
    # 添加引用                                       # merged_res["restored_content"]
    # final_answer, out_reference = add_citations(res, merged_res["chunk_content"].tolist(),merged_res["reference"].tolist(), rerank_model)
    messages.append({"role": "user", "content": query})
    messages.append({"role": "assistant","content": res})
    
    return res, [], []






if __name__ == "__main__":

    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained('autodl-tmp/models/jina-clip-v2', trust_remote_code=True)
    model = model.to(device)
    model.eval()

    rerank_model = AutoModelForSequenceClassification.from_pretrained('autodl-tmp/models/jina-reranker-v2-base-multilingual', trust_remote_code=True)
    rerank_model = rerank_model.to(device)
    rerank_model.eval()

    infinity_object = infinity.connect("/root/RAG/database", config_path="/root/RAG/infinity_config/infinity_conf.toml")
    db_object = infinity_object.get_database("paper")

    
    # 加载模型，大模型和向量模型
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com", api_key="sk-bf21fcc37a07487ea72fb7f5aa82ad18"))
    evaluator_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="autodl-tmp/models/bce-embedding-base_v1"))
    # 指标
    metrics = [
        LLMContextRecall(llm=evaluator_llm), 
        FactualCorrectness(llm=evaluator_llm), 
        Faithfulness(llm=evaluator_llm),
        SemanticSimilarity(embeddings=evaluator_embeddings)
        ]
    
    # 评估
    # 读取问题文件
    with open('RAG/run/questions.txt', 'r', encoding='utf-8') as file:
        questions = file.readlines()
    # 去除每个问题的换行符
    questions = [q.strip() for q in questions]
    answers = [] # rag的回复
    contexts = [] # 上下文
    ground_truths = [] # 真实的人工回复

    for question in tqdm(questions):
        # 维护历史会话
        system_prompt = """
            你是一个专门为论文提问设计的助手，擅长根据检索到的上下文内容回答用户的问题。
            你的回答应当基于提供的上下文，确保准确性和相关性。
            避免过度解释或脱离实际的内容，尽量简洁明了，聚焦于用户提问的具体问题。
            """
        messages=[
                {"role": "system", "content": system_prompt}
                ]

        # 执行检索流程
        res,image_res = retrieve(question, model, db_object)
        # 执行生成流程
        answer, _, _ = generate(question,res,image_res,db_object)
        # 大模型的回复
        ground_truth = zhipu_eval(question)

        answers.append(answer)
        contexts.append(res["chunk_content"].tolist())
        ground_truths.append(ground_truth)

        


    # To dict
    data = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths
    }

    # Convert dict to dataset
    eval_dataset = Dataset.from_dict(data)
    # 评估
    results = evaluate(dataset=eval_dataset, metrics=metrics)
    df = results.to_pandas()

    df.to_csv("RAG/evalation/eavl.csv", index=False, encoding="utf-8")



