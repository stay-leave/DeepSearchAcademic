from infinity_embedded.common import ConflictType
from infinity_embedded.index import IndexInfo, IndexType
import infinity_embedded as infinity
# 可能是Python的sdk是这个，要从这引入
# 下面是从本地文件夹引入的
# from infinity.common import ConflictType
# from infinity.index import IndexInfo, IndexType
import pandas as pd
import os
import ast
import numpy as np
from tqdm import tqdm
import time as t
from transformers import AutoModel
import torch
import uuid



def init_database(db_name="paper"):
    """
    返回数据库服务端实例
    """
    # 连接数据库服务端，本地文件夹，必须是绝对路径
    infinity_object = infinity.connect("/root/DeepSearchAcademic/database", config_path="/root/DeepSearchAcademic/infinity_config/infinity_conf.toml")
    # 删除数据库，如果有
    infinity_object.drop_database(db_name, conflict_type = ConflictType.Ignore)
    # 创建数据库
    infinity_object.create_database(db_name)

    return infinity_object


def create_tables(infinity_object, db_name="paper"):
    """
    函数作用：
    在指定数据库中创建用于存储论文及其章节、块、图片、表格、公式数据的表结构。
    输入参数：
        infinity_object: 数据库实例对象，提供 get_database() 和 create_table() 方法
        db_name: 数据库名称，默认为 "paper"
    返回：
        table_paper, table_image等: 已创建的论文表和图片表等的实例
    """

    # 按数据库名获取数据库对象
    db_object = infinity_object.get_database(db_name)

    # 以下为数据库的操作：
    # 将论文相关的数据分为多张表进行存储：主表(论文表)、章节表、块表、以及存放图片、表格、公式的子表。
    # 主表：存储论文的基本信息(标题、作者、摘要、关键词等)
    # 章节表：存储论文的章节信息(章节标题、内容)
    # 块表：存储章节分块后的文本(用于更细粒度的检索或处理)
    # 图片表、表格表、公式表：存储对应的多媒体内容，关联到具体的章节和块。

    # 创建论文表(paper表)
    # 存储每篇论文的元数据信息，如论文ID、标题、作者、摘要、关键词、文件路径和摘要向量嵌入等
    table_paper = db_object.create_table(
        "table_paper", 
        {
            "paper_uuid": {"type": "varchar"}, # 主键，用于唯一标识一篇论文
            "title": {"type": "varchar"},       # 论文标题
            "author": {"type": "varchar"},      # 作者信息
            "abstract": {"type": "varchar"},    # 摘要文本
            "abstract_embed": {"type": "vector,1024,float"}, # 摘要的向量表示（嵌入向量）
            "key_words": {"type": "varchar"},   # 关键词
            "references": {"type": "varchar"},   # 参考文献
            "file_path": {"type": "varchar"}    # 原始文件路径
        }
    )

    # 创建章节表(chapter表)
    # 每条记录对应论文的一个章节，包括章节ID、顺序、标题、内容和向量嵌入
    table_chapter = db_object.create_table(
        "table_chapter", 
        {
            "paper_uuid": {"type": "varchar"},    # 对应 paper 表的 paper_uuid，以关联到论文
            "chapter_uuid": {"type": "varchar"},  # 主键，唯一标识该章节
            "chapter_order": {"type": "int"},     # 章节在论文中的顺序
            "chapter_header": {"type": "varchar"},# 章节标题
            "chapter_content": {"type": "varchar"},# 章节内容文本
            "chapter_embed": {"type": "vector,1024,float"} # 章节内容的向量嵌入
        }
    )

    # 创建块表(chunk表)
    # 将章节进一步分块，每块是一条记录。存储块ID、块内容和向量嵌入
    table_chunk = db_object.create_table(
        "table_chunk", 
        {
            "paper_uuid": {"type": "varchar"},   # 与论文关联
            "chapter_uuid": {"type": "varchar"}, # 与章节关联
            "chunk_uuid": {"type": "varchar"},   # 主键，唯一标识该块
            "chunk_order": {"type": "int"},      # 块在章节中的顺序
            "chunk_content": {"type": "varchar"},# 块内容
            "chunk_embed": {"type": "vector,1024,float"} # 块内容的向量嵌入
        }
    )

    # 创建图片表(image表)
    # 每条记录对应一张图片，与特定块或章节关联，用于存储图片路径和向量表示
    table_image = db_object.create_table(
        "table_image", 
        {
            "paper_uuid": {"type": "varchar", "default": ""},    # 与论文关联
            "chapter_uuid": {"type": "varchar", "default": ""},  # 与章节关联
            "chunk_uuid": {"type": "varchar", "default": ""},    # 与块关联
            "image_uuid": {"type": "varchar"},    # 主键，唯一标识该图片
            "images_placeholder": {"type": "varchar"}, # 标识符
            "image_path": {"type": "varchar"},    # 图片文件路径或URL
            "image_embed": {"type": "vector,1024,float"} # 图片向量表示（如图像特征嵌入）
        }
    )

    # 创建表格表(table表)
    # 每条记录对应论文中的一个表格
    table_table = db_object.create_table(
        "table_table", 
        {
            "paper_uuid": {"type": "varchar", "default": ""},   # 与论文关联
            "chapter_uuid": {"type": "varchar", "default": ""}, # 与章节关联
            "chunk_uuid": {"type": "varchar", "default": ""},   # 与块关联
            "table_uuid": {"type": "varchar"},   # 主键，唯一标识该表格
            "chunk_tables_placeholder": {"type": "varchar"},
            "table_content": {"type": "varchar"} # 表格内容（文本化的表格数据）
        }
    )

    # 创建公式表(formula表)
    # 每条记录对应一条公式
    table_formula = db_object.create_table(
        "table_formula", 
        {
            "paper_uuid": {"type": "varchar", "default": ""},    # 与论文关联
            "chapter_uuid": {"type": "varchar", "default": ""},  # 与章节关联
            "chunk_uuid": {"type": "varchar", "default": ""},    # 与块关联
            "formula_uuid": {"type": "varchar"},  # 主键，唯一标识该公式
            "chunk_formulas_placeholder": {"type": "varchar"},
            "formula_content": {"type": "varchar"}# 公式内容（可能存储公式的LaTeX表示或文本描述）
        }
    )

    # 函数返回创建的论文表和图片表等
    return table_paper, table_chapter, table_chunk, table_image, table_table, table_formula

    
def create_index(table_object, index_name, index_info):
    """
    输入：表实例，索引名称，索引信息。全文索引需要建表完成后
    输出：索引
    """

    table_object.create_index(index_name, index_info, conflict_type = ConflictType.Error)

def insert_embed_indexs(table_paper, table_chapter, table_chunk, table_image):
    # 给数据表创建不同的索引
    # table_paper，需要在摘要上设置向量和全文索引，在关键词,题名,作者上设置全文索引
    embed_index_info = IndexInfo(
        "abstract_embed", # 在该字段上建立向量索引
        IndexType.Hnsw,
        {"M":"16", "ef_construction":"50", "metric":"cosine"}
    )
    create_index(table_paper, "abstract_embed_index", embed_index_info)

    # table_chapter 在章节内容上设置向量和全文索引，在标题上设置全文索引
    embed_index_info = IndexInfo(
        "chapter_embed",
        IndexType.Hnsw,
        {"M":"16", "ef_construction":"50", "metric":"cosine"}
    )
    create_index(table_chapter, "chapter_embed_index", embed_index_info)
    
    # table_chunk 在chunk内容上设置向量和全文索引
    embed_index_info = IndexInfo(
        "chunk_embed",
        IndexType.Hnsw,
        {"M":"16", "ef_construction":"50", "metric":"cosine"}
    )
    create_index(table_chunk, "chunk_embed_index", embed_index_info)

    # table_image 在image向量上设置向量索引
    embed_index_info = IndexInfo(
        "image_embed",
        IndexType.Hnsw,
        {"M":"16", "ef_construction":"50", "metric":"cosine"}
    )
    create_index(table_image, "image_embed_index", embed_index_info)

    print("创建向量索引完成!")



def insert_data(table_object, records, unique_key):
    """
    输入：表实例，待插入数据，[dict]格式。要对记录去重
    
    假设records中的每个dict都有"image_path"作为唯一标识符（可根据实际需求调整）
    """
    print(f"正在去重{table_object}的数据!")
    seen = set()
    unique_records = []
    
    for record in records:
        # 主键名
        key = record[unique_key]
        if key not in seen:
            seen.add(key)
            unique_records.append(record)
        # 若已在seen中，说明重复，跳过不添加

    # 插入去重后的unique_records
    table_object.insert(unique_records)
    print(f"插入 {len(unique_records)} 条记录到 {table_object} 成功! （原有 {len(records)} 条，去重后 {len(unique_records)} 条）")

    


# 入库，在线嵌入
def input_data(file_path):
    """
    输入：分块好的数据,csv
    输出：存入数据库
    """
    # 初始化数据库服务端，创建数据库，返回数据库服务端实例
    infinity_object = init_database()
    # 选择数据库，创建六张数据表
    table_paper, table_chapter, table_chunk, table_image, table_table, table_formula = create_tables(infinity_object)
    # 在数据表上创建向量索引
    insert_embed_indexs(table_paper, table_chapter, table_chunk, table_image)

    ### 创建嵌入,新的数据
    data_paper = [] # table_paper数据
    data_chapter = [] # table_chapter数据
    data_chunk = [] # table_chunk数据
    data_image = [] # table_image数据
    data_table = [] # table_table数据
    data_formula = [] # table_formula数据

    df = pd.read_csv(file_path) # 待入库的数据
    # 遍历 DataFrame 的每一行
    for index in tqdm(range(len(df)), desc="Processing rows"):
        row = df.iloc[index] # 使用 iloc 选择当前行的字段值
        paper_uuid = row["paper_uuid"]
        title = row["title"]
        author = row["author"]
        abstract = row["abstract"]
        abstract_embed = model.encode_text(abstract).squeeze() # 嵌入
        key_words = row["keywords"]
        references = row["references"]
        file_path = row["file_path"]
        # table_paper数据  有重复的
        data_paper.append({
            "paper_uuid":paper_uuid,
            "title": title,
            "author": author,
            "abstract": abstract,
            "abstract_embed": abstract_embed,
            "key_words": key_words,
            "references": references,
            "file_path": file_path
        })

        # 父目录，用作图片路径
        base_dir = os.path.dirname(file_path)

        chapter_uuid = row["chapter_uuid"]
        chapter_order = row["chapter_order"]
        chapter_header = row["chapter_header"]
        chapter_content = row["chapter_content"]
        chapter_embed = model.encode_text(chapter_content).squeeze() # 嵌入
        # table_chapter数据 有重复的
        data_chapter.append({
            "paper_uuid": paper_uuid,
            "chapter_uuid": chapter_uuid,
            "chapter_order": chapter_order,
            "chapter_header": chapter_header,
            "chapter_content": chapter_content,
            "chapter_embed": chapter_embed
        })
        
        chunk_uuid = row["chunk_uuid"]
        chunk_order = row["chunk_order"]
        chunk_content = row["chunk_content"]
        chunk_embed = model.encode_text(chunk_content).squeeze() # 嵌入
        # table_chunk数据
        data_chunk.append({
            "paper_uuid": paper_uuid,
            "chapter_uuid": chapter_uuid,
            "chunk_uuid": chunk_uuid,
            "chunk_order": chunk_order,
            "chunk_content": chunk_content,
            "chunk_embed": chunk_embed
        })
        
        # 块级别的多媒体数据 
        chunk_images = row["chunk_images"] # 图片路径实际内容
        chunk_tables = row["chunk_tables"]
        chunk_formulas = row["chunk_formulas"]
        chunk_images_placeholders = row["chunk_images_placeholders"] # 占位符
        chunk_tables_placeholders = row["chunk_tables_placeholders"]
        chunk_formulas_placeholders = row["chunk_formulas_placeholders"]
        
        # data_image 数据
        chunk_images_list = ast.literal_eval(chunk_images) # 如果不为空，则同时遍历内容和占位符
        chunk_images_placeholders_list = ast.literal_eval(chunk_images_placeholders)
        for img, images_placeholder in zip(chunk_images_list, chunk_images_placeholders_list):
            image_uuid = str(uuid.uuid4())
            image_path_full = os.path.join(base_dir, img)
            image_embed = model.encode_image(image_path_full).squeeze()
            record = {
                "paper_uuid": paper_uuid,
                "chapter_uuid": chapter_uuid,
                "chunk_uuid": chunk_uuid,
                "image_uuid": image_uuid,
                "images_placeholder": images_placeholder, # 占位符
                "image_path": image_path_full,
                "image_embed": image_embed
            }
            data_image.append(record)
            
        # data_table 数据
        chunk_tables_list = ast.literal_eval(chunk_tables)
        chunk_tables_placeholders_list = ast.literal_eval(chunk_tables_placeholders)
        for t,chunk_tables_placeholder in zip(chunk_tables_list, chunk_tables_placeholders_list):
            table_uuid = str(uuid.uuid4())
            record = {
                "paper_uuid": paper_uuid,
                "chapter_uuid": chapter_uuid,
                "chunk_uuid": chunk_uuid,
                "table_uuid": table_uuid,
                "chunk_tables_placeholder": chunk_tables_placeholder,
                "table_content": t
            }
            data_table.append(record)

        # data_formula数据
        chunk_formulas_list = ast.literal_eval(chunk_formulas)
        chunk_formulas_placeholders_list = ast.literal_eval(chunk_formulas_placeholders)
        
        for f,chunk_formulas_placeholder in zip(chunk_formulas_list,chunk_formulas_placeholders_list):
            formula_uuid = str(uuid.uuid4())
            record = {
                "paper_uuid": paper_uuid,
                "chapter_uuid": chapter_uuid,
                "chunk_uuid": chunk_uuid,
                "formula_uuid": formula_uuid,
                "chunk_formulas_placeholder": chunk_formulas_placeholder,
                "formula_content": f
            }
            data_formula.append(record)
            
    
    # 存入表
    insert_data(table_paper, data_paper,"paper_uuid")
    insert_data(table_chapter, data_chapter,"chapter_uuid")
    insert_data(table_chunk, data_chunk,"chunk_uuid")
    insert_data(table_image, data_image,"image_uuid")
    insert_data(table_table, data_table,"table_uuid")
    insert_data(table_formula, data_formula,"formula_uuid")

    # table_paper的全文索引-标题
    full_index_info = IndexInfo(
        "title",
        IndexType.FullText,
        {"ANALYZER": "chinese"}
    )
    create_index(table_paper, "title_full_index", full_index_info)
    # table_paper的全文索引-作者
    full_index_info = IndexInfo(
        "author",
        IndexType.FullText,
        {"ANALYZER": "chinese"}
    )
    create_index(table_paper, "author_full_index", full_index_info)
    # table_paper的全文索引-关键词
    full_index_info = IndexInfo(
        "key_words",
        IndexType.FullText,
        {"ANALYZER": "chinese"}
    )
    create_index(table_paper, "key_words_full_index", full_index_info)
    # table_paper的全文索引-摘要
    full_index_info = IndexInfo(
        "abstract",
        IndexType.FullText,
        {"ANALYZER": "chinese"}
    )
    create_index(table_paper, "abstract_full_index", full_index_info)

    # table_chapter的全文索引-章节标题
    full_index_info = IndexInfo(
        "chapter_header",
        IndexType.FullText,
        {"ANALYZER": "chinese"}
    )
    create_index(table_chapter, "chapter_header_full_index", full_index_info)
    # table_chapter的全文索引-内容
    full_index_info = IndexInfo(
        "chapter_content",
        IndexType.FullText,
        {"ANALYZER": "chinese"}
    )
    create_index(table_chapter, "chapter_content_full_index", full_index_info)

    # table_chunk的全文索引-内容
    full_index_info = IndexInfo(
        "chunk_content",
        IndexType.FullText,
        {"ANALYZER": "chinese"}
    )
    create_index(table_chunk, "chunk_content_full_index", full_index_info)

    print("存储完毕，索引建立完毕！")


# 测试用的查询
def query():
    # 查询
    infinity_object = infinity.connect("/root/DeepSearchAcademic/database/test", config_path="/root/DeepSearchAcademic/infinity_config/infinity_conf.toml")
    db_object = infinity_object.get_database("paper")
    table_paper = db_object.get_table("table_paper")
    print(table_paper.output(["*"]).to_pl())
    table_image = db_object.get_table("table_image")
    print(table_image.output(["*"]).to_pl())

    # 全文检索
    matching_text = "高校舆情"
    topn = 3
    res = table_paper.output(["*"]).match_text('abstract', matching_text, topn).to_pl()
    print(res)

    print("向量检索")
    # 向量检索
    query_embed = model.encode_text(matching_text,task='retrieval.query')
    res = table_paper.output(["*"]).match_dense(
        "abstract_embed", query_embed, "float", "cosine", 3,
        {"threshold": "0.5"}
        ).to_pl()
    print(res)



if __name__ == "__main__":

    
    
    """
    # 连接数据库服务端，本地文件夹，必须是绝对路径
    infinity_object = infinity.connect("/root/DeepSearchAcademic/database/test", config_path="/root/DeepSearchAcademic/infinity_config/infinity_conf.toml")
    # 选择数据库，选择数据表
    # 按名称索引到指定数据库
    db_object = infinity_object.get_database("lhd")
    res = db_object.list_tables()
    print(res.table_names)
    res = db_object.show_table('table_text')
    print(res)
    # 数据表
    table_text = db_object.get_table("table_text")

    print(table_text)"""



    # v2  向量模型初始化
    model = AutoModel.from_pretrained('/root/autodl-tmp/models/jina-clip-v2', trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 模块化后的运行命令
    file_path = "DeepSearchAcademic/middel_data/12.13.csv"
    input_data(file_path)
    
    # query()
    




