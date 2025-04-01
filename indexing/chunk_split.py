from langchain_text_splitters import MarkdownHeaderTextSplitter,RecursiveCharacterTextSplitter
import re
from pathlib import Path
import uuid
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import json
from langchain_experimental.text_splitter import SemanticChunker as LCSemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from harvesttext import HarvestText

# 分句
def split_sentences(text):
    """
    使用正则表达式对中文文本进行分句。
    """
    sentences = re.split(r'(。|！|\!|？|\?)', text)  # 按句号、问号、感叹号分句
    sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]  # 合并标点
    if sentences and sentences[-1] not in '。！？':
        sentences.append(sentences[-1])  # 保证最后一条完整
    return [s.strip() for s in sentences if s.strip()]

def harves_abs(text):
    # 抽取式摘要，大模型失败时使用
    try:
        sentences = split_sentences(text)
        res = ht.get_summary(sentences, topK=2, avoid_repeat=True)
        return "".join(res)
    except:
        return "" # 如果失败，就返回空
# 生成chunk的摘要
def deepseek(user_prompt):
    
    system_prompt = """你是一个专业的文本总结助手。你的任务是读取用户提供的一段文本并为其生成简明扼要的中文摘要。要求如下：

    1. 不要在摘要中使用诸如“这段文本的摘要是”或“总结”这类导语。
    2. 直接给出文本的主要内容。
    3. 如果摘要有latex公式，请你渲染。
    4. 最终只返回一个 JSON 对象，键为 "summary"，值为你的摘要字符串。如：{"summary": "这里放摘要内容"}

    请在用户提供文本后直接输出 JSON 格式的摘要结果。
    """

    client = OpenAI(api_key="sk-bf21fcc37a07487ea72fb7f5aa82ad18", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        temperature=0
    )
    
    return response.choices[0].message.content

def generate_abs(doc):
    # 小于200个字就没必要摘要
    if len(doc) < 200:
        return doc
    # 生成每个chunk的摘要
    for _ in range(3):
        res = deepseek(doc)
        try:
            result = json.loads(res)
            # 获取摘要内容
            summary = result.get("summary", "")
            return summary # 
        except:
            print("获取摘要失败，重试")
            continue
    # 尝试3次都失败，使用抽取式摘要
    return harves_abs(doc)
 
# 提取公式，表格，图片等数据
def extract_latexs(content):
    """
    提取 markdown 中的 LaTeX 公式。

    参数:
    content (str): Markdown 格式的文本内容

    返回:
    list: 提取出的 LaTeX 公式列表
    """
    # 正则表达式匹配 $$...$$ 之间的公式
    # pattern = r'(\$\$(.*?)\$\$)'
    # latexs = re.findall(pattern, content, re.DOTALL)
    # return [latex[0] for latex in latexs]  # 保留 $$ 符号

    # 非贪婪匹配
    pattern = r'(?<=\$\$)(.*?)(?=\$\$)'
    matches = re.findall(pattern, content, flags=re.DOTALL)
    # 为了保留$$符号，将其重新加回去
    formulas = [f"$${m}$$" for m in matches]
    return formulas

def extract_tables(content):
    """
    提取 markdown 中的表格。

    参数:
    content (str): Markdown 格式的文本内容

    返回:
    list: 提取出的表格内容
    """
    # 假设表格是 HTML 格式（例如 <body>...</body>），你可能需要调整正则表达式以更准确匹配
    pattern = r'<body.*?>(.*?)</body>'
    tables = re.findall(pattern, content, re.DOTALL)
    return tables  # 返回匹配的表格内容

def extract_images(content):
    """
    提取 markdown 中的图片。

    参数:
    content (str): Markdown 格式的文本内容

    返回:
    list: 提取出的图片链接列表
    """
    # 正则表达式匹配 ![alt text](image_url)
    pattern = r'!\[.*?\]\((.*?)\)'
    images = re.findall(pattern, content)
    return images  # 返回图片链接

# 提取论文元数据
def extract_author(title, content):
    """
    从 Markdown 内容中提取作者信息。
    假设作者信息在标题之后的若干非空行，且在摘要之前。

    参数:
    title (str): 文档的标题，例如 "# 基于BERT的金融文本情感分析模型"
    content (str): Markdown 格式的文本内容

    返回:
    str 或 None: 提取的作者信息，若未找到则返回 None
    """
    lines = content.split('\n')
    title_found = False
    authors = []
    
    # 编译正则表达式，匹配以一个或多个 # 开头，后跟标题内容
    title_pattern = re.compile(r'^#+\s+' + re.escape(title.lstrip('# ').strip()))
    
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if not title_found:
            # 检查当前行是否为标题
            if title_pattern.match(stripped_line):
                title_found = True
            continue  # 继续检查下一行
        else:
            # 检查是否遇到摘要
            if stripped_line.startswith('摘要') or stripped_line.startswith('【摘要】'):
                break  # 如果遇到摘要，停止查找
            # 检查当前行是否为非空行
            if stripped_line:
                authors.append(stripped_line)
    
    if authors:
        # 将多行作者信息合并为一个字符串，使用空格或其他分隔符
        return ' '.join(authors)
    else:
        return None

def extract_abstract_and_keywords(content):
    """
    从 Markdown 内容中提取摘要和关键词。
    假设摘要在“摘要：”之后，关键词在“关键词：”之后，
    并且关键词不包括“中图分类号”等后续信息。

    参数:
    content (str): Markdown 格式的文本内容

    返回:
    tuple: (摘要, 关键词)
    """
    # 提取摘要
    abstract_pattern = r'(摘要[：:]|【摘要】)\s*([\s\S]*?)(?=\n关键词[：:]|【关键词】|$)'
    abstract_match = re.search(abstract_pattern, content)
    abstract = abstract_match.group(2).strip() if abstract_match else None
    
    # 提取关键词
    keywords_pattern = r'(关键词[：:]|【关键词】)\s*([\s\S]*?)(?=\n|中图分类号[：:]|文献标志码|$)'
    keywords_match = re.search(keywords_pattern, content)
    keywords = keywords_match.group(2).strip() if keywords_match else None
    
    # 清理关键词，去除可能的引号或多余空格
    if keywords:
        # 去除前后的引号（全角或半角）
        keywords = keywords.strip('“”"')
        # 分割关键词，去除“中图分类号”等后续信息
        keywords = re.split(r'[；;]\s*中图分类号[：:]', keywords)[0]
    
    return abstract, keywords

# 规避英文元数据
def is_english(text, threshold=0.6):
    # 统计英文字符的数量。判断是否是英文摘要等。应该检测第二个，倒数第一个
    english_chars = re.findall(r'[a-zA-Z]', text)
    english_ratio = len(english_chars) / len(text) if len(text) > 0 else 0
    # 如果英文字符的比例大于设定的阈值，则判定为英文文本
    return english_ratio >= threshold

# 启用，替换文本中的表格等为占位符
def replace_modalities(content):
    """
    将原文中的公式、图片、表格替换为占位符。
    占位符格式：
    [FORMULA_1], [FORMULA_2], ...
    [IMAGE_1], [IMAGE_2], ...
    [TABLE_1], [TABLE_2], ...
    
    返回值：
    (cleaned_content, formulas, images, tables)
    cleaned_content: 替换后的文本
    formulas: 提取的公式列表
    images: 提取的图片链接列表
    tables: 提取的表格原文列表
    """
    # 先提取
    formulas = extract_latexs(content)
    images = extract_images(content)
    tables = extract_tables(content)

    # 替换公式
    # 使用一次性替换的方式，依次替换匹配到的公式为 [FORMULA_n]
    for idx, _ in enumerate(formulas, start=1):
        # 使用非贪婪匹配来确保匹配对应的$$...$$
        content = re.sub(r'\$\$.*?\$\$', f'[FORMULA_{idx}]', content, count=1, flags=re.DOTALL)
    
    # 替换图片
    for idx, _ in enumerate(images, start=1):
        content = re.sub(r'!\[.*?\]\(.*?\)', f'[IMAGE_{idx}]', content, count=1)

    # 替换表格
    for idx, _ in enumerate(tables, start=1):
        content = re.sub(r'<html.*?>.*?</html>', f'[TABLE_{idx}]', content, count=1, flags=re.DOTALL)

    # 去除多余空格与换行
    # content = re.sub(r'\s+', ' ', content).strip()

    return content, formulas, images, tables

# 清洗行级别空白
def clean_lines(page_content):
    # 首先，把多余的空白如连续空格、制表符换成单个空格或直接移除行尾空格等
    page_content = re.sub(r'[ \t]+', ' ', page_content)  # 将连续空格压缩成一个空格（保留必要空格）
    page_content = re.sub(r'[ \t]+\n', '\n', page_content)  # 去掉行尾空格
    # 按行分割
    lines = page_content.split('\n')
    # 去掉纯空行（包括只有空格的行）
    clean_lines = [line for line in lines if line.strip() != '']
    # 重新合并为单个字符串
    return '\n'.join(clean_lines)

# 处理小数点和英文句号
# 定义英文到中文标点符号的映射
PUNCTUATION_MAPPING = {
    '.': '。',
    ',': '，',
    ':': '：',
    ';': '；',
    '!': '！',
    '?': '？',
}

def convert_punctuation(text):
    """
    将文本中的英文标点符号转换为中文标点符号。
    """
    # 创建一个正则表达式模式，匹配所有需要转换的英文标点符号
    pattern = re.compile('|'.join(re.escape(k) for k in PUNCTUATION_MAPPING.keys()))
    
    # 使用子函数进行替换
    converted_text = pattern.sub(lambda x: PUNCTUATION_MAPPING[x.group()], text)
    
    return converted_text

def replace_decimal_points(text):
    """
    将文本中的小数点和点后跟英文字母的点替换为占位符。
    仅替换位于数字之间的点，或点后跟字母的点，或点位于字母之间的点，
    或点位于字母和数字之间的点，避免替换其他情况的点。
    """
    DECIMAL_PLACEHOLDER = "<DECIMAL_POINT>"
    # 使用前向和后向肯定断言，确保点位于数字之间或点后跟字母
    pattern = r'\.(?![\u4e00-\u9fff])|(?<![\u4e00-\u9fff])\.'
    
    return re.sub(pattern, DECIMAL_PLACEHOLDER, text)

def restore_decimal_points(text):
    """
    将占位符替换回小数点。
    """
    return text.replace("<DECIMAL_POINT>", '.')


def merge_separators(chunks, separators):
    """
    合并以标点符号开头的块到前一个块的末尾。
    确保最后一个块的末尾是。
    """
    merged_chunks = []
    punctuation_set = set(separators)
    chinese_punctuations = {'。', '！', '？', '，', '；', '：', '“', '”', '‘', '’', '（', '）', '【', '】', '《', '》', '——'}
    for chunk in chunks:
        if merged_chunks:
            # 检查当前块是否以任何一个分隔符开头
            first_char = chunk[0]
            if first_char in punctuation_set:
                # 将分隔符移动到前一个块的末尾
                merged_chunks[-1] += first_char
                # 去除当前块的第一个字符
                chunk = chunk[1:]
        merged_chunks.append(chunk)

    # 检查最后一个块是否以标点符号结尾
    if merged_chunks:
        last_chunk = merged_chunks[-1]
        if last_chunk:
            last_char = last_chunk[-1]
            # 如果最后一个字符不是中文标点符号
            if last_char == '.' or last_char not in chinese_punctuations:
                # 移除英文句号并替换为中文句号
                if last_char == '.':
                    last_chunk = last_chunk[:-1] + '。'
                else:
                    # 如果没有标点符号，直接添加中文句号
                    last_chunk += '。'
                merged_chunks[-1] = last_chunk

    return merged_chunks

# 处理单个论文为list的记录
def process_paper(file_path):
    # 读取 markdown 文件
    # file_path = "DeepSearchAcademic/parsed_file/融合多特征和注意力机制的多模态情感分析模型_吕学强/ocr/融合多特征和注意力机制的多模态情感分析模型_吕学强.md"
    with open(file_path, 'r', encoding="utf-8") as file:
        content = file.read()

    # 配置 Markdown 分割器，依据标题层级进行分块
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3")
    ]
    # md分块
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    # 分块结果
    md_header_splits = markdown_splitter.split_text(content)

    # 正则表达式用于查找占位符，关联块和多媒体内容
    PLACEHOLDER_REGEX = re.compile(r'\[(FORMULA|IMAGE|TABLE)_(\d+)\]')

    # 为每个细分的chunk生成记录，汇总为一个paper的
    records = []
    # 标志位，用于第一次读取标题
    flag = True
    references = "" # 参考文献
    title_uuid = str(uuid.uuid4()) # 论文的uuid
    # 遍历每个章节
    for chapter_order,i in enumerate(md_header_splits):
        chapter_uuid = str(uuid.uuid4()) # 章节的id
        # 检查该章节内容是否只有一行，避免杂项干扰，尤其是第一行是杂项
        if '\n' not in i.page_content:
            print("只有一行，跳过")  # 规避杂项
            continue
        # 默认论文元数据是在第一个分块，也就是论文题目和引言中间
        if flag: # 论文元数据
            print("提取元数据")
            flag = False # 不再提取
            # 提取标题
            title = i.metadata["Header 1"] # 元数据
            # print(f"标题: {title}")
            # 提取作者信息
            author = extract_author(i.metadata["Header 1"], i.page_content)
            # print(f"作者: {author}")
            # 提取摘要和关键词
            abstract, keywords = extract_abstract_and_keywords(i.page_content)
            # print(f"摘要: {abstract}")
            # print(f"关键词: {keywords}")
            continue # 不进行后续处理
            
        # 判断是否为参考文献部分。可能有多个都是，所以要累加
        if "参考" in i.metadata["Header 1"] or "References" in i.metadata["Header 1"]:
            reference = i.page_content
            references += reference
            # print(f"参考文献: {references}")
            print("获取参考文献！")
            continue # 不参与后续
                    
        # 不要这些数据
        if i.metadata["Header 1"] in ["作者贡献声明：","利益冲突声明：","基金项目：","支撑数据：","致谢：","作者简介:","本文引用格式"]:
            print("跳过无关元数据")
            continue
        
        # 章节
        # 章节文本，查看是否是全英文，如果是，抛弃
        if is_english(i.page_content):
            print("跳过英文元数据")
            continue

        # 章节文本清洗里面的空行
        page_content = i.page_content
        page_content = clean_lines(page_content)
        # 提取章节中的文本，图片，表格，公式，文本里有图片等的占位符。
        text_content, formulas, images, tables = replace_modalities(page_content)

        # 将文本按行切分
        lines = text_content.split('\n')
        # 去掉开头的所有标题行（以 # 开头）
        while lines and lines[0].strip().startswith('#'):
            lines.pop(0)
        docs = ''.join(lines) # 保持紧凑
        # 统一中英文重要标点符号
        # 预处理：替换小数点为占位符
        text_content = replace_decimal_points(docs)
        # 预处理：转换英文标点符号为中文标点符号
        text_content = convert_punctuation(text_content)
        # 恢复小数点
        text_content = restore_decimal_points(text_content)
        
        # 对单个章节进行分块，规避占位符
        docs = text_splitter.split_text(text_content)
        # 处理分块后的标点在前面的问题，和每章最后一块没句号
        docs = merge_separators(docs, ["。", "！", "？", "；"])
        
        for doc_id, doc in enumerate(docs):
            # 移除所有类型的空白字符，包括不可见字符。为空则不处理
            if re.sub(r'\s+', '', doc) == "":
                continue
            
            # 查找当前块中包含的所有占位符
            placeholders = PLACEHOLDER_REGEX.findall(doc)
            # 初始化当前块的多媒体内容
            chunk_formulas = []
            chunk_images = []
            chunk_tables = []
            # 初始化块级别的占位符计数
            formula_counter = 1
            image_counter = 1
            table_counter = 1
            # 创建章节全局占位符到块级占位符的映射
            placeholder_mapping = {}
            # 遍历每个占位符
            for media_type, global_index_str in placeholders:
                global_index = int(global_index_str) - 1  # 转换为0基索引
                global_placeholder = f"[{media_type}_{global_index_str}]"

                if media_type == "FORMULA":
                    if global_index < len(formulas):
                        formula = formulas[global_index]
                        chunk_formulas.append(formula)
                        # 分配块级占位符
                        chunk_placeholder = f"[FORMULA_{formula_counter}]"
                        placeholder_mapping[global_placeholder] = chunk_placeholder
                        formula_counter += 1
                elif media_type == "IMAGE":
                    if global_index < len(images):
                        image = images[global_index]
                        chunk_images.append(image)
                        # 分配块级占位符
                        chunk_placeholder = f"[IMAGE_{image_counter}]"
                        placeholder_mapping[global_placeholder] = chunk_placeholder
                        image_counter += 1
                elif media_type == "TABLE":
                    if global_index < len(tables):
                        table = tables[global_index]
                        chunk_tables.append(table)
                        # 分配块级占位符
                        chunk_placeholder = f"[TABLE_{table_counter}]"
                        placeholder_mapping[global_placeholder] = chunk_placeholder
                        table_counter += 1
            
            # 替换全局占位符为块级占位符
            for global_placeholder, chunk_placeholder in placeholder_mapping.items():
                doc = doc.replace(global_placeholder, chunk_placeholder)

            # 保存每个二次切分的chunk的记录
            record = {
                'chunk_uuid': str(uuid.uuid4()), # chunk的id # 字符串，主键
                # 论文元数据
                'paper_uuid': title_uuid, # 论文的id # 字符串
                'title': title, # 字符串
                'author': author, # 字符串
                'abstract': abstract, # 字符串
                'keywords': keywords, # 字符串
                'file_path': file_path, # 文件路径,字符串 

                # 章节元数据
                'chapter_uuid': chapter_uuid, # 章节的id # 字符串
                'chapter_order': chapter_order+1, # 章节的内部顺序
                'chapter_header': i.metadata, # 章节标题，字典
                'chapter_content': text_content, # 章节内容
                'chapter_images': images, # 所属章节的图片路径
                'chapter_formulas': formulas, # 公式，列表
                'chapter_tables': tables, # 表格，列表
                 
                # chunk元数据
                'chunk_order': doc_id+1, # 块在章节中的顺序，从1开始
                'chunk_content': doc, # 文本内容，不包括图片,表格,公式   字符串
                # 'chunk_abs': generate_abs(doc), # 文本内容的摘要， 字符串
                'chunk_images': chunk_images,  # 块对应的图片列表
                'chunk_images_placeholders': [f"[IMAGE_{idx+1}]" for idx in range(len(chunk_images))],  # 块级占位符
                'chunk_formulas': chunk_formulas,  # 块对应的公式列表
                'chunk_formulas_placeholders': [f"[FORMULA_{idx+1}]" for idx in range(len(chunk_formulas))],  # 块级占位符
                'chunk_tables': chunk_tables,  # 块对应的表格列表
                'chunk_tables_placeholders': [f"[TABLE_{idx+1}]" for idx in range(len(chunk_tables))],  # 块级占位符
                }
            records.append(record)

    # 添加参考文献元数据
    for record in records:
        record['references'] = references

    return records


if __name__ == "__main__":
    # 语义分块
    # model_name = "autodl-tmp/models/bce-embedding-base_v1"
    # embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    # text_splitter = LCSemanticChunker(embedding_model, breakpoint_threshold_type="gradient")


    ht = HarvestText()

    # 递归分块
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "。",     # 中文句号
            "！",     # 中文感叹号
            "？",     # 中文问号
            "；",     # 中文分号
            "",       # 最后回退策略
        ],
        chunk_size=500,
        chunk_overlap=0,  # 添加适当的重叠
        keep_separator=True,
        is_separator_regex=False,
        # length_function=len,
    )
    
    
    # 所有论文的文件夹
    folder = 'DeepSearchAcademic/parsed_file'
    folder_path = Path(folder)
    # 子文件夹的名称
    subfolders = [subfolder.name for subfolder in folder_path.iterdir() if subfolder.is_dir()]

    data = [] # 所有的数据
    for sub in tqdm(subfolders):
        subfolder_path = folder_path / sub  # 构建子文件夹路径
        md_file_path = subfolder_path / "ocr" / f"{sub}.md"  # 构建同名 .md 文件路径
        # 单个文件分块后的结果,list[dict]
        records = process_paper(md_file_path)
        data.extend(records)
    
    df = pd.DataFrame(data)
    df.to_csv("DeepSearchAcademic/middel_data/12.13.csv", encoding="utf-8-sig", index=False)
    
    print(df.info())

