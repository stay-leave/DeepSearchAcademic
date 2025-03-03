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

from utils import paper_only_author,paper_only_content,paper_author_content,text_to_image
from utils import chunk_hybid_search,chunk_hybid_search_with_paper,chunk_filter_paper,add_citations
from llm_api import deepseek_judge_rag,deepseek_daily_qa,kimi_decompose
from llm_api import deepseek_judge_pc,kimi_paper_col,deepseek_rewrite,zhipu_image_query
from llm_api import deepseek_single_query,deepseek_sub_queries,deepseek_multi_step_queries

@contextmanager
def timer():
    start = t.time()
    yield lambda: t.time() - start


@st.cache_resource
def init():
    # 初始化数据库连接
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained('autodl-tmp/models/jina-clip-v2', trust_remote_code=True)
    model = model.to(device)
    model.eval()

    rerank_model = AutoModelForSequenceClassification.from_pretrained('autodl-tmp/models/jina-reranker-v2-base-multilingual', trust_remote_code=True)
    rerank_model = rerank_model.to(device)
    rerank_model.eval()

    return model,rerank_model

# 单独逻辑函数
# 单独处理论文级别检索的流程
def process_paper_query(query, model, db_object):
    # 提取是针对作者还是其他内容的查询
    t.sleep(1.2) # 防止并发过多
    paper_col = kimi_paper_col(query)
    paper_col = json.loads(paper_col)
    # 检查作者和内容字段是否存在并有效
    authors = paper_col.get("author", [])
    contents = paper_col.get("content", [])
    if authors and contents: # 如果同时包含作者和内容，则进行混合检索
        logging.info("执行混合检索：根据作者和内容检索论文")
        res = paper_author_content(db_object, model, authors, contents)
    elif authors: # 如果只有作者，执行作者检索
        logging.info("执行作者检索：根据作者检索论文")
        res = paper_only_author(db_object, authors)
    elif contents: # 如果只有内容，执行内容检索
        logging.info("执行内容检索：根据内容检索论文")
        res = paper_only_content(db_object, model, contents)
    else:
        logging.info("无法进行检索：没有提供有效的作者或内容")
        columns = ["paper_uuid", "title", "author", "file_path", "_score"]
        res = pd.DataFrame(columns=columns)

    return res

# 单一问题的检索处理逻辑，用于多个子问题复用
def process_single_query(query, model, db_object):
    logging.info(f"正在进入单一问题的处理流程，问题：{query}")
    # 判断是 论文查询 OR 细节查询
    t.sleep(1.2) # 防止并发过多
    judge_pc = deepseek_judge_pc(query)
    judge_pc = json.loads(judge_pc)
    logging.info(f"判断单一问题是查询论文还是查询细节，结果是{judge_pc}")
    # 是论文查询
    if judge_pc["is_paper_query"]:
        logging.info("是查询论文问题！")
        res = process_paper_query(query, model, db_object)

    else: # 内容检索
        logging.info(f"是内容检索问题！")
        # 改写query
        t.sleep(1.2) # 防止并发过多
        rewrite = deepseek_rewrite(query)
        rewrite = json.loads(rewrite).get("professional_query","")
        logging.info(f"正在改写查询，改写后的查询：{rewrite}")
        if rewrite: # 改写成功，执行混合检索
            logging.info(f"改写成功，使用改写后的查询执行chunk混合检索!")
            res,image_res,table_res = chunk_hybid_search(db_object, model, rewrite) # df
        else:
            logging.info(f"改写失败，使用原始查询执行chunk混合检索!")
            res,image_res,table_res = chunk_hybid_search(db_object, model, query) # df
    
    return res,image_res,table_res

# 单一问题的检索处理逻辑，探索版，多模态
def process_single_query_one(query, model, db_object):
    logging.info(f"正在进入单一问题的处理流程，问题：{query}")
    # 判断是 论文查询 OR 细节查询
    t.sleep(1.2) # 防止并发过多
    judge_pc = deepseek_judge_pc(query)
    judge_pc = json.loads(judge_pc)
    logging.info(f"判断单一问题是查询论文还是查询细节，结果是{judge_pc}")
    # 是论文查询
    if judge_pc["is_paper_query"]:
        logging.info("是查询论文问题！")
        res = process_paper_query(query, model, db_object)

    else: # 内容检索
        logging.info(f"是内容检索问题！")
        # 改写query
        t.sleep(1.2) # 防止并发过多
        rewrite = deepseek_rewrite(query)
        rewrite = json.loads(rewrite).get("professional_query","")
        logging.info(f"正在改写查询，改写后的查询：{rewrite}")
        if rewrite: # 改写成功，执行混合检索
            logging.info(f"改写成功，使用改写后的查询执行chunk混合检索和以文搜图!")
            res,image_res,table_res = chunk_hybid_search(db_object, model, rewrite) # 返回完整的chunk，图，表df
            image_res_1 = text_to_image(db_object, query, model) # 以文搜图，一张
            image_res = pd.concat([image_res,image_res_1]) # 把检索到的图放后面，以文搜图不如文本
        else:
            logging.info(f"改写失败，使用原始查询执行chunk混合检索和以文搜图!")
            res,image_res,table_res = chunk_hybid_search(db_object, model, query) # df
            image_res_1 = text_to_image(db_object, query, model) # 以文搜图，一张
            image_res = pd.concat([image_res,image_res_1]) # 合并
    
    return res,image_res,table_res


# 组织单个问题的上下文，生成回答。用于多个问题复用的
def generate_single_query(query, res, rerank_model,db_object):
    # 如果是论文级别，上下文就是论文名称+作者。不用加引用
    if 'title' in res.columns:
        # logging.info(f"是查询论文的问题。使用['title, author', 'title, author']的格式组织上下文。")
        # res['context'] = res['title'].astype(str) + ', ' + res['author'].astype(str)
        # context = res['context'].tolist() # 上下文
        logging.info("是查询论文的问题。直接以['title', 'author', 'file_path']格式返回Markdown格式的内容。")
        # 构建Markdown格式的字符串
        md_content = "| 论文标题 | 作者 | 文件路径 |\n| --- | --- | --- |\n"
        for index, row in res.iterrows():
            md_content += f"| {row['title']} | {row['author']} | {row['file_path']} |\n"
        return md_content
    else: # chunk级别，使用完整的chunk。如果是多模态大模型，则将图片也输入
        # 为了加引用，必须有论文信息，此时的res里面没有
        logging.info(f"是查询内容的问题。使用['restored_content']的格式组织上下文。")
        context = res['restored_content'].tolist() # 上下文
        # 检索论文数据作为参考文献来源
        paper_uuids = res["paper_uuid"].drop_duplicates().tolist() # 论文id
        paper_res = chunk_filter_paper(db_object,paper_uuids)
        # 根据论文id合并论文表和chunk表，以chunk表为基准
        merged_res = pd.merge(res, paper_res[['paper_uuid', 'title', 'author', 'file_path']], on="paper_uuid", how="left")
        # ["number"] = [i for i in range(1,len(merged_res)+1)] # 为每个记录添加序号
        merged_res['reference'] = merged_res['title'].astype(str) + ', ' + merged_res['author'].astype(str) # 添加对应的论文参考信息
        # 给两个list，分别是chunk内容list 和 论文参考list
        
        # 组织成检索结果1,2,3等
        logging.info(f"组织上下文为['检索结果1':content]的格式。")
        contexts = []
        for idx,cont in enumerate(context):
            contexts.append(f"检索结果{idx+1}: {cont}")
        # 根据上下文信息的无引用回复
        res = deepseek_single_query(query, contexts)
        final_answer, out_reference = add_citations(res, merged_res["restored_content"].tolist(),merged_res["reference"].tolist(), rerank_model)

        # 找到chunk的图片和表格，数据库检索
        


    return final_answer, out_reference

# 组织单个问题的上下文，生成回答，专为单一问题的，加多模态的回复
def generate_single_query_one(query, res, image_res, table_res, rerank_model,db_object):
    # 如果是论文级别，上下文就是论文名称+作者。不用加引用
    if 'title' in res.columns:
        # logging.info(f"是查询论文的问题。使用['title, author', 'title, author']的格式组织上下文。")
        # res['context'] = res['title'].astype(str) + ', ' + res['author'].astype(str)
        # context = res['context'].tolist() # 上下文
        logging.info("是查询论文的问题。直接以['title', 'author', 'file_path']格式返回Markdown格式的内容。")
        # 构建Markdown格式的字符串
        md_content = "| 论文标题 | 作者 | 文件路径 |\n| --- | --- | --- |\n"
        for index, row in res.iterrows():
            md_content += f"| {row['title']} | {row['author']} | {row['file_path']} |\n"
        return md_content
    else: # chunk级别，使用完整的chunk。如果是多模态大模型，则将图片也输入
        # 为了加引用，必须有论文信息，此时的res里面没有
        logging.info(f"是查询内容的问题。使用['restored_content']的格式组织上下文。")
        context = res['restored_content'].tolist() # 上下文
        # 检索论文数据作为参考文献来源
        paper_uuids = res["paper_uuid"].drop_duplicates().tolist() # 论文id
        paper_res = chunk_filter_paper(db_object,paper_uuids)
        # 根据论文id合并论文表和chunk表，以chunk表为基准
        merged_res = pd.merge(res, paper_res[['paper_uuid', 'title', 'author', 'file_path']], on="paper_uuid", how="left")
        # ["number"] = [i for i in range(1,len(merged_res)+1)] # 为每个记录添加序号
        merged_res['reference'] = merged_res['title'].astype(str) + ', ' + merged_res['author'].astype(str) # 添加对应的论文参考信息
        # 给两个list，分别是chunk内容list 和 论文参考list
        
        # 组织成检索结果1,2,3等
        logging.info(f"组织上下文为['检索结果1':content]的格式。")
        contexts = []
        for idx,cont in enumerate(context):
            contexts.append(f"检索结果{idx+1}: {cont}")

        # 找到chunk的图片和表格，数据库检索
        images = image_res["image_path"].tolist()
        tables = table_res["table_content"].tolist()
        # 得到第一张图片的路径
        image = images[0]

        # 图文的，模型不行，不用
        # res = zhipu_image_query(query,contexts,image)
        # 根据上下文信息的无引用回复,纯文本的
        res = deepseek_single_query(query, contexts)
        # 添加引用
        final_answer, out_reference = add_citations(res, merged_res["restored_content"].tolist(),merged_res["reference"].tolist(), rerank_model)

    return final_answer, out_reference, images



# 下面是整体的流程
# 检索流程
def retrieve(query, model, db_object,file):
    """
    输入：用户的查询，向量模型，数据库，检索结果的保存名
    输出：问题类型，检索结果，子问题列表（对于非单一问题）
    """

    flag = "" # 标记是什么类型的问题
    sub_queries = [] # 初始化子问题列表为空
    # 判断是论文RAG查询，还是日常会话
    #judge_rag = deepseek_judge_rag(query)
    #judge_rag = json.loads(judge_rag).get("query_type","")
    #logging.info(f'判断是RAG查询，还是日常会话。结果是{judge_rag}')


    # 进入rag流程，统一先检索
    logging.info("进入RAG！")
    # 判断是否分解
    # t.sleep(1.2) # 防止并发过多
    # decompose = kimi_decompose(query)
    decompose = {"single_query": ""}  # 统一单一问题
    decompose = json.loads(decompose)
    logging.info(f'正在尝试分解问题，将被归类为 single_query, sub_queries, multi_step_queries，然后根据不同的类别回答。结果是{decompose}')
    # 如果是单一问题，判断查询是否是论文还是其他内容
    if "single_query" in decompose:
        logging.info("是single_query，进入单一问题处理逻辑！")
        flag = "single_query" # 标记
        # 执行单一问题处理逻辑
        res,image_res,table_res = process_single_query_one(query, model, db_object)
    elif "sub_queries" in decompose:
        logging.info("是sub_queries，进入多个独立子问题处理逻辑！")
        flag = "sub_queries" # 标记
        # 获取子问题列表
        sub_queries = decompose["sub_queries"] # list
        logging.info(f"sub_queries:{sub_queries}")
        res = {}  # 字典来存储每个子问题的检索结果
        logging.info(f"正在处理多个独立子问题")
        for sub_query in tqdm(sub_queries, desc="正在处理多个独立子问题："):
            res_sub,image_res,table_res = process_single_query(sub_query, model, db_object)
            t.sleep(1.2) # 防止并发过多
            res[sub_query] = res_sub  # 将每个子问题的结果放入字典
        logging.info(f"处理多个独立子问题完成，得到的检索结果是dict")
    elif "multi_step_queries" in decompose:
        logging.info("是multi_step_queries，进入层级多跳子问题处理逻辑！") # 只考虑需要在指定论文范围内查询细节的多跳
        flag = "multi_step_queries" # 标记
        res = {}  # 字典来存储每个子问题的检索结果
        # 获取子问题列表
        sub_queries = decompose["multi_step_queries"] # list
        logging.info(f"multi_step_queries:{sub_queries}")
        first_query = sub_queries.pop(0) # 弹出第一个子问题
        # 检查第一个问题是否是查论文的
        # 判断是 论文查询 OR 细节查询
        t.sleep(1.2) # 防止并发过多
        judge_pc = deepseek_judge_pc(first_query)
        judge_pc = json.loads(judge_pc)
        logging.info(f"检查第一个问题是否是查询论文，结果是:{judge_pc}")
        # 是论文查询
        if judge_pc["is_paper_query"]:
            logging.info(f"第一个问题是查询论文问题！")
            paper_res = process_paper_query(query, model, db_object) # 论文级别检索结果
            paper_uuids = paper_res["paper_uuid"].to_list() # 论文id，用于使细节检索在限定论文里
            logging.info(f"得到论文范围paper_uuids: {paper_uuids}")
            t.sleep(1.2) # 防止并发过多
            logging.info(f"正在处理论文限定下的多个子问题")
            for sub_query in tqdm(sub_queries, desc="正在处理多个多跳子问题："):
                res_sub = chunk_hybid_search_with_paper(db_object,model,query,paper_uuids)
                t.sleep(1.2) # 防止并发过多
                res[sub_query] = res_sub  # 将每个子问题的结果放入字典
            logging.info(f"处理论文限定下的多个子问题完成，得到的检索结果是dict")
        else:
            logging.info(f"第一个问题是查询内容问题！转为多个独立子问题处理逻辑")
            sub_queries.insert(0,first_query) # 把弹出的第一个子问题还原
            logging.info(f"正在处理多个独立子问题")
            for sub_query in tqdm(sub_queries, desc="正在处理多个子问题："):
                res_sub,image_res,table_res = process_single_query(sub_query, model, db_object)
                t.sleep(1.2) # 防止并发过多
                res[sub_query] = res_sub  # 将每个子问题的结果放入字典
            logging.info(f"处理多个独立子问题完成，得到的检索结果是dict")

    # 保存检索的结果，设置为日志同名的csv
    retrieved_file_name = os.path.join("RAG/run/log", f'{file}')
    if isinstance(res, pd.DataFrame):
        # 如果res是DataFrame，直接保存
        res.to_csv(f'{retrieved_file_name}.csv', index=False)
    if isinstance(res, dict):
        res_serializable = {}
        for key, value in res.items():
            res_serializable[key] = value.to_dict(orient='records') # 将df转换为json先
        # 正常序列化保存
        with open(f'{retrieved_file_name}.json', 'w') as f:
            json.dump(res_serializable, f, indent=4)
    logging.info(f"保存检索结果完成，共计{len(res)}条记录，文件是：{retrieved_file_name}")
    
    return flag,res,sub_queries,image_res,table_res

# 生成流程
def generate(query,flag,res,sub_queries, rerank_model,db_object,image_res,table_res):
    """
    输入：用户的查询，问题类型，检索结果, 子问题（对于非单一问题）,chunk相关的图片表结果，表格表结果
    输出：生成的回复
    """
    # 检索完毕，得到res，有df或dict两种，开始生成
    
    try:
        #re_query = kimi_rewrite(query) # 改写问题。
        #query = json.loads(re_query).get("professional_query","")
        logging.info(f"为生成改写查询，改写后: {query}")
    except:
        pass
        logging.info(f"为生成改写查询失败，使用原始查询")
    # 处理单一问题,res是df
    if flag == "single_query":
        logging.info("开始生成single_query的回复！")
        answer, reference, images = generate_single_query_one(query, res, image_res, table_res, rerank_model,db_object)
        return answer, reference, images
        # answer += "\n\n" + reference + "\n\n" + images
        logging.info("得到single_query的回复！")
    # 处理多个独立子问题,res是个字典
    elif flag == "sub_queries":
        logging.info("开始生成sub_queries的回复！遍历每个子问题和对应检索结果，一一回答单个问题。然后把问题和答案作为上下文。")
        # 遍历子问题，找到对应的df
        context = [] # 上下文是每个子问题的回答，要单独加一个函数总结
        for index,sub_query in enumerate(sub_queries):
            logging.info(f"正在生成子问题{sub_query}的回复！")
            ans_sub,reference = generate_single_query(sub_query, res[sub_query], rerank_model,db_object) # 每个子问题都回答
            context.append(
                {
                f"子问题_{index+1}":sub_query,
                "答案":ans_sub
                }
                )
            t.sleep(1.2) # 防止并发过多
        answer = deepseek_sub_queries(query, context)
        logging.info("得到sub_queries的回复！")
    # 多跳子问题,res是字典
    elif flag == "multi_step_queries":
        logging.info("开始生成multi_step_queries的回复！直接执行sub_queries的生成逻辑。但在生成总结的时候执行另外的提示词。是递进关系。")
        # 遍历子问题，找到对应的df
        # 初始化一个空字符串，用来存储逐步的上下文
        full_context = ""
        for index,sub_query in enumerate(sub_queries):
            logging.info(f"正在生成子问题{sub_query}的回复！")
            ans_sub,reference = generate_single_query(sub_query, res[sub_query], rerank_model,db_object) # 每个子问题都回答
            # 在full_context中添加当前子问题的回答
            full_context += f"问题{index+1}: {sub_query}\n"
            full_context += f"答案{index+1}: {ans_sub}\n"
            # 记录当前子问题的上下文，以便下一个子问题使用
            if index < len(sub_queries) - 1:
                full_context += f"下一个问题需要基于当前问题的答案作为前置条件。"
            t.sleep(1.2) # 防止并发过多
        answer = deepseek_multi_step_queries(query, full_context)
        logging.info("得到multi_step_queries的回复！")

    logging.info(f"回复:{answer}")
    
    return answer,[],[]
           

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
    st.title("多模态RAG学术问答系统")
    # 日志配置（可选）
    file = "test_1"
    logger = init_logging(file)

    model,rerank_model = init()
    # infinity_object = infinity.connect("/root/RAG/database", config_path="/root/RAG/infinity_config/infinity_conf.toml")
    infinity_object = get_database()
    db_object = infinity_object.get_database("paper")
    

    st.write("请输入您的查询问题，然后点击“提交”按钮获取回答、参考文献和相关图片。")

    
    query = st.text_input("查询问题", "事理图谱的原理是什么?")
        
    if st.button("提交"):
        # 执行RAG流程
        with timer() as get_retrieve_time:
            flag, res, sub_queries, image_res, table_res = retrieve(query, model, db_object,file)
            retrieve_time = get_retrieve_time()
        with timer() as get_generate_time:
            answer, reference, images = generate(query, flag, res, sub_queries, rerank_model, db_object, image_res, table_res)
            generate_time = get_generate_time()
        # 分离回答、参考文献和图片
        st.subheader("回答")
        st.write(answer)

        st.subheader("参考文献")
        st.markdown("\n\n".join(reference))

        st.subheader("相关图片")
        for img_path in images:
            if os.path.exists(img_path):
                img = Image.open(img_path)
                st.image(img, caption=os.path.basename(img_path), use_container_width=True)
            else:
                st.write(f"图片路径不存在：{img_path}")

        # 显示耗时信息
        st.markdown(f"**检索耗时**: {retrieve_time:.2f} 秒")
        st.markdown(f"**生成耗时**: {generate_time:.2f} 秒")
        st.markdown("---")  # 分隔线
        

if __name__ == "__main__":

    main()