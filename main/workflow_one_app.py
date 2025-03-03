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
from llm_api import deepseek_judge_rag,deepseek_daily_qa,kimi_decompose, deepseek_chat,deepseek_chat_refine
from llm_api import deepseek_judge_pc,kimi_paper_col,deepseek_rewrite,zhipu_image_query
from llm_api import deepseek_single_query,deepseek_sub_queries,deepseek_multi_step_queries

@contextmanager
def timer():
    start = t.time()
    yield lambda: t.time() - start


@st.cache_resource
def init():
    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained('autodl-tmp/models/jina-clip-v2', trust_remote_code=True)
    model = model.to(device)
    model.eval()

    rerank_model = AutoModelForSequenceClassification.from_pretrained('autodl-tmp/models/jina-reranker-v2-base-multilingual', trust_remote_code=True)
    rerank_model = rerank_model.to(device)
    rerank_model.eval()

    return model,rerank_model



# 下面是整体的流程
# 检索流程
def retrieve(query, model, db_object,file, messages):
    """
    输入：用户的查询，向量模型，数据库，检索结果的保存名
    输出：问题类型，检索结果
    """
    # 改写query
    try:
        rewrite = deepseek_rewrite(query, messages)
        rewrite = json.loads(rewrite).get("professional_query","")
        logging.info(f"正在改写查询，改写后的查询：{rewrite}")
        logging.info(f"改写成功，使用改写后的查询执行chunk混合检索和以文搜图!")
        res,image_res,table_res = chunk_hybid_search(db_object, model, rewrite, 50, 0.7) # 返回完整的chunk，图，表df
        
    except Exception as e:
        print("改写失败")
        logging.info(f"改写失败，使用原始查询执行chunk混合检索和以文搜图!\n报错：{e}")
        res,image_res,table_res = chunk_hybid_search(db_object, model, query, 50, 0.7) # df

    #image_res_1 = text_to_image(db_object, query, model) # 以文搜图，一张
    #image_res = pd.concat([image_res,image_res_1]) # 合并
    
    # 保存检索的结果，设置为日志同名的csv
    retrieved_file_name = os.path.join("RAG/run/log", f'{file}')
    res.to_csv(f'{retrieved_file_name}.csv', index=False)
    
    logging.info(f"保存检索结果完成，共计{len(res)}条记录，文件是：{retrieved_file_name}")
    
    return res,image_res

# 生成流程
def generate(query,res,rerank_model,db_object,image_res, messages):
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
    final_answer, out_reference = add_citations(res, merged_res["chunk_content"].tolist(),merged_res["reference"].tolist(), rerank_model, 0.7)
    messages.append({"role": "user", "content": query})
    messages.append({"role": "assistant","content": res})
    
    return final_answer, out_reference, images

    
           

# 初始化日志（可以在加载模型和数据库时配置）
def init_logging(file_name="app_log"):
    log_file_path = os.path.join("RAG/run/log", f'{file_name}.log')
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file_path,
        filemode='w'
    )
    logger = logging.getLogger(__name__)
    logger.info("日志系统初始化完成。")
    return logger

@st.cache_resource
def get_database():
    print("初始化数据库连接")
    infinity_object = infinity.connect("/root/RAG/database", config_path="/root/RAG/infinity_config/infinity_conf.toml")
    
    return infinity_object


# Streamlit 应用
def main():

    st.set_page_config(page_title="多模态RAG学术问答系统", layout="wide")
    st.title("多模态学术问答系统")
    
    # 初始化日志
    file = "test_1"
    logger = init_logging(file)

    # 初始化模型
    with st.spinner("正在加载模型..."):
        model, rerank_model = init()

    # 初始化数据库
    with st.spinner("正在连接数据库..."):
        infinity_object = get_database()
        db_object = infinity_object.get_database("paper")

    # 初始化Session State中的messages和history
    if 'messages' not in st.session_state:
        system_prompt = """
            你是一个专门为论文提问设计的助手，擅长根据检索到的上下文内容回答用户的问题。
            你的回答应当基于提供的上下文，确保准确性和相关性。
            避免过度解释或脱离实际的内容，尽量简洁明了，聚焦于用户提问的具体问题。
        """
        st.session_state.messages = [{"role": "system", "content": system_prompt}]
        st.session_state.history = []  # 用于存储历史记录

    # 维护历史会话长度
    if len(st.session_state.messages) >= 7:  # 例如，限制最大消息数为6
        st.session_state.messages = st.session_state.messages[:1] + st.session_state.messages[-6:]

    # 侧边栏：显示 messages 列表
    with st.sidebar:
        st.header("🔄 Messages")
        if st.session_state.messages[1:]:
            for idx, msg in enumerate(st.session_state.messages[1:], 1):
                st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")
        else:
            st.write("暂无消息。")

    # 主界面
    st.write("请输入您的查询问题，然后点击“提交”按钮获取回答、参考文献和相关图片。")

    query = st.text_input("💬 查询问题", "事理图谱的原理是什么?")

    if st.button("提交"):
        if query.strip() == "":
            st.warning("请输入有效的查询问题。")
        else:
            with st.spinner("系统处理中..."):
                # 显示进度条
                progress_bar = st.progress(0)
                # 执行检索
                progress_bar.progress(10)
                with timer() as get_retrieve_time:
                    res, image_res = retrieve(query, model, db_object, file, st.session_state.messages)
                    retrieve_time = get_retrieve_time()
                progress_bar.progress(50)

                # 执行生成
                with timer() as get_generate_time:
                    answer, reference, images = generate(query, res, rerank_model, db_object, image_res, st.session_state.messages)
                    generate_time = get_generate_time()
                progress_bar.progress(100)
                t.sleep(0.5)  # 确保进度条显示到100%

                # 关闭进度条
                progress_bar.empty()

            # 更新历史记录
            st.session_state.history.append({
                    "query": query,
                    "answer": answer,
                    "reference": reference,
                    "images": images
                })

            # 使用 st.chat_message 显示对话（从 history 中获取带引用的回答）
            st.markdown("### 🗨️ 当前会话")
            for record in st.session_state.history:
                with st.chat_message("user"):
                    st.write(record["query"])
                with st.chat_message("assistant"):
                    st.write(record['answer'])

            st.subheader("📚 参考文献")
            if reference:
                st.markdown("\n\n".join(reference))
            else:
                st.write("无参考文献。")

            st.subheader("🖼️ 相关图片")
            if images:
                cols = st.columns(3)
                for idx, img_path in enumerate(images):
                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        cols[idx % 3].image(img, use_container_width=True)
                    else:
                        cols[idx % 3].write(f"图片路径不存在：{img_path}")
            else:
                st.write("无相关图片。")


            # 显示耗时信息
            st.markdown(f"**检索耗时**: {retrieve_time:.2f} 秒")
            st.markdown(f"**生成耗时**: {generate_time:.2f} 秒")
            st.markdown("---")  # 分隔线

    
    

if __name__ == "__main__":

    main()