# DeepSearchAcademic
基于舆情中文核心论文的deepsearch的个人项目

博客[https://stay-leave.github.io/post/%E5%A4%9A%E6%A8%A1%E6%80%81rag%E8%AE%BE%E8%AE%A1%E5%8F%8A%E5%AE%9E%E8%B7%B5/]

多模态RAG的一些理论方法，分为两类，一种是解析式文档多模态RAG(将一个文档切分为页面，然后再用版式识别的方式对文档进行各种模态元素进行分割、解析、提取，然后再嵌入、检索）；另一种是DocVQA式文档多模态RAG(将文档切分为页面图像，不再细分，然后根据页面图像级别进行检索)。

![RAG流程](https://github.com/user-attachments/assets/92c832c7-cff6-4a61-b485-5f63e952a1f0)

## 概述

用户的查询是文本，输出是文本和图片。

数据解析阶段：MinerU库对pdf进行解析，提取md文件，包括表格、公式、文本、图片。

索引阶段：Langchain进行md标题分块，分为三级标题。再对每个章节进行递归分块。得到每个块对应的表格、公式、图片。

数据库设计：论文表，章节表，文档块表，图片表，表格表，公式表。

检索阶段：用户的查询进来，进行改写，用改写后的来查询文档块表，得到文档块表后联结图片、表格、公式，还原出原本的文档块内容。

生成阶段：将原本的文档块内容作为上下文，和用户查询一起输入给大模型，得到回复。利用重排模型计算回复中每个句子对每条上下文的相关性得分，取最高的为参考文献。将上下文中的图片一并输出。

## 起源
2024年12月份，RAG的范式基本差不多了，纯文本的RAG已经非常成熟了，多模态RAG也在迅速兴起。博主的硕士毕业论文写的就是多模态舆情分析，但是苦于自己创造的新定义、理论找不到支撑文献，于是做了一个多模态的RAG系统，旨在搜集20年以来的中文舆情分析期刊论文，试图结合最新的信息检索技术，给论文找到合适的方法和理论。这样的方法对于很多大学生都是适用的。

在博主看来，RAG是信息检索，LLM同样也是信息检索，只不过前者的知识在数据库，后者的知识在FFN上的区别。人类知识浩如烟海，能够高效查找和整合信息是核心。

## 环境依赖

```
conda create --name myenv 
conda activate myenv
pip install -r requirements.txt
```

## 使用方法

向量数据库：infinity
大模型：deepseek,kimi,zhipu

1.下载MinerU库，手动下载pdf文件。解析pdf为md，image文件。

2.使用indexing文件夹，先分块存储为csv。然后存储到向量数据库。

3.使用main文件夹，选择app后缀的，开始运行。

## 领域
在中文核心期刊，主要是南大、北大核心中有关舆情分析的论文。手动下载了PDF文件。

## 支持的输入
文本。

后续应支持图文混合输入和公式输入。

## 输出
文本、图片、参考来源。

目前仅做了搜索和简单回答，有待让大模型整理资料，逐步生成研究报告。

## 效果图
![image](https://github.com/user-attachments/assets/fd5cdb58-264c-4cde-8369-8d145629c172)

![image](https://github.com/user-attachments/assets/35c4a181-6546-462d-b419-fc7de6ed7d5e)

## Ragas评测
使用智谱AI的回答作为ground_truth，因为免费。设计了95道问题，评测结果如下。

![image](https://github.com/user-attachments/assets/c83ae8a1-c469-472d-bec5-c3dd958891eb)

