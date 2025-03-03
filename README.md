# DeepSearchAcademic
基于舆情中文核心论文的deepsearch的个人项目

博客：https://stay-leave.github.io/post/%E5%A4%9A%E6%A8%A1%E6%80%81rag%E8%AE%BE%E8%AE%A1%E5%8F%8A%E5%AE%9E%E8%B7%B5/

## 起源
2024年12月份，RAG的范式基本差不多了，纯文本的RAG已经非常成熟了，多模态RAG也在迅速兴起。博主的硕士毕业论文写的就是多模态舆情分析，但是苦于自己创造的新定义、理论找不到支撑文献，于是做了一个多模态的RAG系统，旨在搜集20年以来的中文舆情分析期刊论文，试图结合最新的信息检索技术，给论文找到合适的方法和理论。这样的方法对于很多大学生都是适用的。‘’

在博主看来，RAG是信息检索，LLM同样也是信息检索，只不过前者的知识在数据库，后者的知识在FFN上的区别。人类知识浩如烟海，能够高效查找和整合信息是核心。

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

