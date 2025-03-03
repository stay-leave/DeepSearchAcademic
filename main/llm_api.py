from openai import OpenAI
import time as t
import base64
from zhipuai import ZhipuAI
import json

# 使用智谱回答问题，用于评估时充当真实回复
def zhipu_eval(query):
    system_prompt = """
    你是一个学术问答助手，专门用于处理和回答与论文相关的问题。
    """

    client = ZhipuAI(api_key="") # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4v-flash",  # 填写需要调用的模型名称
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.7
    )
    return response.choices[0].message.content


# 回复单一问题，纯文本,多轮对话
def deepseek_chat(query, contexts, history):
    # 当前的查询，当前的
    
    user_prompt = f"""
    ### 用户查询：
    {{query}}

    ### 上下文信息：
    {{contexts}}
    """

    
    # 组织成字符串
    contexts = "\n".join(contexts)
    user_prompt = user_prompt.format_map({"query": query,"contexts": contexts}) # 填充查询和上下文
    # 每次把当前用户消息输入，第一轮时已经有系统提示词了，第二轮时history已经有系统提示词和第一轮的输入输出
    history.append({
		"role": "user",
		"content": user_prompt,	
	})

    client = OpenAI(api_key="", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=history,
        stream=False,
        temperature=1.2,
    )
    history.pop() # 弹出新增的带上下文的用户查询
    return response.choices[0].message.content


# 回复单一问题，纯文本,精炼回复
def deepseek_chat_refine(query):
    # 当前的查询，当前的
    system_prompt = """
    你是一个Markdown格式化助手。请将用户提供的文本中的公式和表格修正为正确的Markdown格式，包括使用LaTeX语法的公式和标准的Markdown表格。
    确保所有公式使用美元符号包围（行内公式使用单个美元符号 `$...$`，块级公式使用双美元符号 `$$...$$`），表格按照Markdown语法正确对齐。
    请以纯文本形式、精炼的方式的json格式回复。

    示例输出：

    {
    "content": "润色修正后的Markdown内容"
    }
    """
    
    client = OpenAI(api_key="", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        stream=False,
        temperature=0.7,
        response_format={
            'type': 'json_object'
            }
    )

    try:
        res = json.loads(response.choices[0].message.content)["content"]
    except Exception as e:
        print(f"精炼失败，原因是{e}")
        res = query

    return res




# 使用智谱回答图文问题
def zhipu_image_query(query, contexts, img_path):
    system_prompt = """
    你是一个多模态的学术问答助手，专门用于处理和回答与论文相关的问题。
    你能够理解和分析文本和图片两种类型的上下文信息，并基于检索到的内容生成准确、简洁的回答。
    #### 你的职责包括但不限于：
    1. **理解用户的问题**：准确理解用户提出的学术问题，无论是基于文本还是图片。
    2. **处理多模态上下文**：根据用户的问题，结合提供的文本内容和图片信息，提取相关的关键信息。
    3. **生成回答**：基于检索到的上下文内容，提供清晰、直接且相关的回答。无需生成参考文献。
    4. **保持准确性和相关性**：确保所有回答内容准确无误，并严格基于检索到的上下文，避免引入不相关的信息。
    5. **简洁明了**：用简洁、易懂的语言回答问题，避免过于学术化或冗长的解释。
    """
    user_prompt = f"""
    ### 用户查询：
    {{query}}

    ### 上下文信息：
    {{contexts}}
    """
    with open(img_path, 'rb') as img_file:
        img_base = base64.b64encode(img_file.read()).decode('utf-8')
    # 组织成字符串
    contexts = "\n".join(contexts)
    user_prompt = user_prompt.format_map({"query": query,"contexts": contexts}) # 填充查询和上下文

    client = ZhipuAI(api_key="") # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4v-flash",  # 填写需要调用的模型名称
        messages=[
        {"role": "system", "content": system_prompt},
        {   
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": user_prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url" : img_base
                }
            }
            ]
        }
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content


# 回复单一问题，纯文本
def deepseek_single_query(query, contexts):
    system_prompt = """
    你是一个专门为论文提问设计的助手，擅长根据检索到的上下文内容回答用户的问题。
    你的回答应当基于提供的上下文，确保准确性和相关性。
    避免过度解释或脱离实际的内容，尽量简洁明了，聚焦于用户提问的具体问题。
    """
    user_prompt = f"""
    ### 用户查询：
    {{query}}

    ### 上下文信息：
    {{contexts}}
    """
    # 组织成字符串
    contexts = "\n".join(contexts)
    user_prompt = user_prompt.format_map({"query": query,"contexts": contexts}) # 填充查询和上下文

    client = OpenAI(api_key="", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        temperature=1.2,
    )

    return response.choices[0].message.content

# 回复多个独立子问题
def deepseek_sub_queries(query, contexts):
    """
    输入：查询，上下文[list]
    输出：回复
    """
    system_prompt = """
    你是一个专业的舆情分析论文助手，擅长整合多个子问题的回答，生成最终的学术性强、结构清晰、逻辑严谨的总结答案。请根据用户的原始查询和提供的各个子问题的回答，完成以下任务：

    1. **理解子问题和答案**：每个子问题和答案都应被视为一个完整的信息单元。请确保你能够准确理解每个子问题，并根据答案内容进行推理和分析。每个子问题与答案的格式为：
    - **子问题**：[子问题内容]
    - **答案**：[回答内容]

    2. **综合分析**：根据所有提供的子问题及其答案，综合分析各部分信息，形成一个完整且连贯的学术性回答。回答应当从整体上总结相关内容，确保不遗漏关键信息，同时避免冗余和重复。

    3. **确保学术性**：生成的答案应具有学术标准，内容准确、清晰，并且严格遵循逻辑。请特别注意在最终总结时，提供完整且精炼的结论，避免过于冗长或简单的描述。

    4. **最终回答格式**：你需要给出一段清晰、连贯且结构合理的回答，帮助用户解决其查询。确保内容不仅解答原始查询，还能有效整合子问题的答案，给出深入的分析和结论。

    请注意，信息应尽可能简洁准确，不要重复已知内容，避免在总结时遗漏任何关键点。
    """

    user_prompt = f"""
    ### 用户查询：
    {{query}}

    ### 子问题及回答：
    {{contexts}}
    """
    # 组织成 子问题1：{{sub_query_1}} 答案：{{answer_1}} 的形式
    contexts = "\n\n".join(contexts)
    user_prompt = user_prompt.format_map({"query": query,"contexts": contexts}) # 填充查询和上下文

    client = OpenAI(api_key="", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        temperature=0.8,
    )

    return response.choices[0].message.content

# 回复层级多跳子问题
def deepseek_multi_step_queries(query, contexts):
    """
    输入：查询，上下文 str
    输出：回复
    """
    system_prompt = """
    你是一个专业的舆情分析论文助手，擅长整合多个递进的子问题及其答案，生成最终的学术性强、结构清晰、逻辑严谨的总结回答。请根据用户的原始查询和提供的各个子问题的回答，完成以下任务：

    1. **理解子问题与递进关系**：
    - 每个子问题和答案都是递进的，答案将成为下一个问题的前置条件。
    - 确保你理解每个子问题的背景，并通过前一个问题的答案推理出下一个问题的相关信息。
    - 你的任务是根据所有子问题及其答案，逐步构建出一个连贯的分析过程。

    2. **子问题格式与回答**：
    - 每个子问题及其回答的格式如下：
      - **子问题**：[子问题内容]
      - **答案**：[回答内容]
    - 逐步理解并分析每个子问题和其答案，避免信息遗漏。

    3. **综合分析与总结**：
    - 基于所有子问题及其答案，针对用户的查询，综合分析并形成一个完整的回答。回答应从整体上总结相关内容，确保不遗漏关键信息。
    - 重点是要显示出递进关系，前一个问题的答案如何成为下一个问题的基础。
    - 在最终总结时，请确保逻辑清晰，避免冗余或重复，给出精炼且完整的结论。

    4. **最终回答格式**：
    - 请生成一个结构合理、条理清晰的总结，帮助用户解答其查询。
    - 确保内容不仅解答了原始查询，还能整合各个子问题的答案，清晰展现出问题之间的递进关系。

    请特别注意，生成的回答要具备学术性，准确简洁，且不会遗漏任何关键点。
    """

    user_prompt = f"""
    ### 用户查询：
    {{query}}

    ### 子问题及回答：
    {{contexts}}
    """

    user_prompt = user_prompt.format_map({"query": query,"contexts": contexts}) # 填充查询和上下文

    client = OpenAI(api_key="", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        temperature=0.8,
    )

    return response.choices[0].message.content


# 判断是否进入rag
def deepseek_judge_rag(user_prompt):
    # 判断 论文问答 或 日常会话
    # 论文问答就进入rag，否则直接用大模型回答
    system_prompt = """
    你是一名智能问答系统。用户会向你提出一个查询请求，你需要对该查询进行分析，并输出一个 JSON 对象，描述该查询的类型。

    JSON 中必须包含以下字段：

    - "query_type": 字符串类型，可取 "paper_qa"（论文问答）或 "daily_qa"（日常问答）之一。

    如果无法确定上述字段的值，请根据你的最佳判断选择最合适的值。

    下面是对输入和输出要求：

    - **输入**：用户的自然语言查询。
    - **输出**：JSON 格式的字符串，对上述字段进行标注。

    示例输出格式：

    {
    "query_type": "paper_qa"
    }

    现在请对用户的查询进行分析，并给出 JSON 格式的输出。
    """

    client = OpenAI(api_key="", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        temperature=0,
        response_format={
            'type': 'json_object'
            }
    )

    return response.choices[0].message.content

# 不是rag，回复日常会话
def deepseek_daily_qa(user_prompt):
    # 日常会话，直接用大模型回答
    system_prompt = "你是一名人工智能助手。"

    client = OpenAI(api_key="", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        temperature=1.0
    )

    return response.choices[0].message.content

# 进入rag流程
# 分解子问题，判断是否是多跳，
def kimi_decompose(user_prompt):

    system_prompt = """
    你是一名智能问答系统的分析助手。当用户提出一个查询问题时，你需要判断该问题是否可以分解成多个子问题，或者它是否是一个层层递进的多跳问题。请遵守以下规则：

    - **如果问题可以分解为几个独立的子问题**，请输出一个包含子问题列表的 JSON 对象，字段名为 "sub_queries"。每个子问题是一个字符串，表示一个单独的问题。
    - **如果问题是多跳问题**，即需要进行多个步骤的推理才能得到最终答案，请输出一个包含推理步骤列表的 JSON 对象，字段名为 "multi_step_queries"。每个步骤是一个字符串，表示一个单独的推理步骤。
    - **如果问题不能分解，也不是多跳问题**，请返回一个 JSON 对象，字段为 "single_query"，其值为原问题。
    - 不要在输出中添加任何额外的解释，只输出 JSON 格式的结果。

    请根据以下规则分析以下问题，并给出你的判断：

    1. 问题是否可以分解为多个子问题？
    2. 问题是否是多跳问题，包含多个推理步骤？
    3. 如果是多跳问题，请列出推理步骤。

    请使用如下 JSON 格式输出你的回复：

    示例1：
        {
        "sub_queries": [
            "LDA模型的定义是什么？",
            "大模型和LDA怎么结合",
            "LDA模型的流程是怎样的？"
        ]
        }

    示例2：
        {
        "multi_step_queries": [
            "经济危机对企业和消费者的影响是什么？",
            "经济危机如何影响股市投资者的行为？",
            "经济危机后股票市场波动的主要原因是什么？"
        ]
        }

    示例3：
        {
        "single_query": "跟LDA有关的图表有哪些"
        }

    """

    client = OpenAI(api_key="", base_url="https://api.moonshot.cn/v1")

    response = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        temperature=0.7,
        response_format={
            'type': 'json_object'
            }
    )

    return response.choices[0].message.content

# 先考虑单跳
# 单跳问题，判断是针对找论文，还是找细节
def deepseek_judge_pc(user_prompt):

    system_prompt = """
    你是一个智能问答系统的助手。当用户提出一个查询问题时，你需要判断该问题的目标是查找论文相关的信息还是查找具体的细节信息。请根据以下规则进行判断，并输出 JSON 格式的结果：

    1. **论文相关问题**：如果问题的目标是找到一篇论文或者论文的相关信息，如标题、摘要、作者、研究方法等，请输出一个 JSON 对象，字段为 "is_paper_query"，值为 `true`。
    2. **细节相关问题**：如果问题的目标是查询更具体的细节信息，如技术步骤、概念解释、方法流程、数据支持等，请输出一个 JSON 对象，字段为 "is_paper_query"，值为 `false`。

    请使用如下 JSON 格式输出你的回复：

    {
        "is_paper_query": false
    }

    """

    client = OpenAI(api_key="", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        temperature=0,
        response_format={
            'type': 'json_object'
            }
    )

    return response.choices[0].message.content

# 如果找论文，是根据作者,还是内容。这个决定了在哪些字段上检索。如果只有作者，就在论文表检索。如果有内容，需要结合chunk表检索。
def kimi_paper_col(user_prompt):
    # 如果缺失某个键，说明没有
    system_prompt = """
    你是一个智能问答系统的助手。当用户提出一个查询问题时，你需要分析该查询问题并返回与论文检索相关的信息。请根据以下规则进行分析：

    1. **作者**：如果查询中包含具体的作者姓名，提取出作者的名字并返回。
    2. **内容**：如果查询中涉及论文的具体内容，如关键词、论文的主题、方法、摘要等，请提取出相关内容并返回。
    3. 如果查询没有包含对应的信息，则返回为空。

    返回 JSON 格式的结果，其中包括：
    - `author`：查询中涉及的作者（如果有）
    - `content`：查询中涉及的论文内容（如关键词、主题、方法、摘要等）

    请使用如下 JSON 格式输出你的回复：

    {
        "author": ["张三"],
        "content": ["深度学习"]
    }

    """

    client = OpenAI(api_key="", base_url="https://api.moonshot.cn/v1")

    response = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        temperature=0.7,
        response_format={
            'type': 'json_object'
            }
    )

    return response.choices[0].message.content

# 如果是找细节，判断是否明确需要多媒体内容（也不用，直接全都返回）
# 对chunk进行检索。检索到了之后，把同一个章节的按顺序合并，还原多媒体内容。
# 单跳完成


# 分解的问题
# 首先检查是子问题，还是多跳
# 对于子问题，并行采用单跳的方式，进行对应的检索，合并检索到的内容
# 对于多跳的问题，先检查是否第一个问题是限定论文的，进入到单跳的找论文，然后继续在指定论文里找chunk；
    # 如果不是，则并行找chunk，然后顺序放置，进行回答


# 查询改写
def deepseek_rewrite(user_prompt,history):
    # 对于单跳查询的改写

    system_prompt = """
    你是一名专业的舆情分析专家。用户会向你提出一个查询请求，你需要将该查询改写为使用专业术语的形式，适用于舆情数据库的检索。

    请参考以下用户之前的查询：
    {previous_queries}

    现在，请将当前的查询改写为专业术语的形式，确保使用陈述句，且不使用疑问句。

    请按照以下要求输出：

    - 输出一个 JSON 对象，包含字段 "professional_query"。
    - 只输出 JSON 对象，不要包含其他文字或格式标记。

    请使用如下 JSON 格式输出你的回复：

    {{
        "professional_query": "当前社交媒体中关于环保议题的舆情趋势分析。"
    }}
    """
    # 提取之前的用户查询
    previous_queries = [msg["content"] for msg in history if msg["role"] == "user"]
    previous_queries_formatted = "\n".join([f"{idx+1}. {q}" for idx, q in enumerate(previous_queries)])
    # 填充系统提示词
    system_prompt_filled = system_prompt.format(previous_queries=previous_queries_formatted)

    client = OpenAI(api_key="", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt_filled},
                    {"role": "user", "content": user_prompt}
                ],
                stream=False,
                temperature=0.7,
                response_format={
                    'type': 'json_object'
                    }
            )
    

    return response.choices[0].message.content.strip()







if __name__ == "__main__":

    history = [{'role': 'system', 'content': '\n        你是一个专门为论文提问设计的助手，擅长根据检索到的上下文内容回答用户的问题。\n        你的回答应当基于提供的上下文，确保准确性和相关性。\n        避免过度解释或脱离实际的内容，尽量简洁明了，聚焦于用户提问的具体问题。\n        '}, {'role': 'user', 'content': '情感建模的主要方法'}, {'role': 'assistant', 'content': '情感建模的主要方法可以分为以下几类：\n\n1. **基于情感词典的方法**：\n   - 依赖于人工构建的情感词典，词典中的每个词汇都具有情感极性信息。\n   - 例如，周咏梅等人提出了基于HowNet和SentiWordNet的中文情感词典，这种方法在微博文本情感分析中表现良好。\n   - 优点是简单直观，但缺点是构建词典耗时且难以适应新词和变化的语言环境。\n\n2. **基于机器学习的方法**：\n   - 包括有监督学习和无监督学习两种。\n   - 常用的机器学习算法有朴素贝叶斯、K最近邻、最大熵、支持向量机（SVM）等。\n   - 优点是能够处理大量数据，提高分类准确性，且不需要频繁更新词典。\n   - 缺点是需要大量标记数据和专业知识来构造特征，且在特定场景下泛化能力有限。\n\n3. **基于深度学习的方法**：\n   - 包括单一网络和混合网络两类。\n   - 常用的深度学习模型有卷积神经网络（CNN）、长短期记忆网络（LSTM）、门控循环单元（GRU）等。\n   - 优点是不需要人工构造特征，扩展性强，能够自动从数据中学习特征。\n   - 例如，文献中提到的SA-SDCCN方法通过空洞卷积网络和稀疏注意力机制提高特征提取能力。\n\n这些方法各有优缺点，选择哪种方法取决于具体的应用场景和数据特性。基于深度学习的方法近年来成为情感分析领域的热门技术，因其强大的特征学习和模型扩展能力。'}]

    res = deepseek_rewrite("事理图谱的原理是什么?", history)
    print(res)
    
    