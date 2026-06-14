# 导入数值计算库，用于分数排序、索引处理
# Import numerical computing library for score sorting and index processing
import numpy as np
# 导入MySQL数据库连接驱动
# Import MySQL database connection driver
import mysql.connector
# 导入HTTP请求库，用于调用DeepSeek云端大模型API
# Import HTTP request library for calling DeepSeek cloud LLM API
import requests
# 导入Gradio库，快速搭建Web聊天交互界面
# Import Gradio library to quickly build web chat interface
import gradio as gr
# 导入正则表达式库，过滤分词结果中的无效字符
# Import regular expression library to filter invalid characters in word segmentation
import re
# 导入BM25检索算法实现类
# Import BM25 retrieval algorithm implementation
from rank_bm25 import BM25Okapi
# 导入日志模块，记录程序运行日志、问题排查
# Import logging module to record program logs and troubleshoot issues
import logging
# 导入jieba中文分词库，用于中文文本分词处理
# Import jieba Chinese word segmentation library for Chinese text processing
import jieba

# 配置日志：日志等级为INFO，格式为 时间戳 - 日志内容
# Configure log: log level is INFO, format: timestamp - log message
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# 初始化日志对象
# Initialize logger instance
logger = logging.getLogger(__name__)

# DeepSeek API 密钥 / DeepSeek API access key
# 请替换为你个人在平台申请的有效密钥 / Replace with your personal API key applied from official platform
API_KEY = "更换为个人的api"

# DeepSeek 接口请求地址 / DeepSeek API request URL
# 替换为对应大模型厂商的完整接口地址 / Replace with complete API URL of your LLM provider
API_URL = "更换为对应厂商的url/"

# 指定调用的云端模型名称 / Specify the name of cloud LLM model
# 填写平台支持的合法模型名 / Fill in valid model name supported by the platform
CLOUD_MODEL_NAME = "你选择的模型"

# MySQL数据库连接配置
# MySQL database connection configuration
DB_CONFIG = {
    "host": "localhost",    # 数据库地址 / Database host
    "user": "root",         # 数据库用户名 / Database username
    "password": "123456",   # 数据库密码 / Database password
    "database": "my_vector_db"  # 数据库名 / Database name
}

# BM25检索返回的Top-K结果数量
# Number of top relevant results returned by BM25 retrieval
TOP_K_RETRIEVAL = 3

# 知识库类：基于BM25+Jieba实现中文文档检索
# Knowledge Base Class: Implement Chinese document retrieval based on BM25 + Jieba
class KnowledgeBase:
    # 构造方法：初始化知识库并加载数据
    # Constructor: Initialize knowledge base and load data
    def __init__(self):
        self.contents = []       # 存储所有文本块内容 / Store all text chunks
        self.bm25 = None         # BM25算法实例 / BM25 algorithm instance
        self._load()             # 执行数据加载与索引构建 / Load data and build index

    # 静态方法：中文分词+字符过滤
    # Static method: Chinese word segmentation and character filtering
    @staticmethod
    def _tokenize(text: str):
        """中文分词 + 保留英文/数字 + 过滤标点与空字符"""
        """Chinese segmentation + keep English/numbers + filter punctuation and empty chars"""
        # jieba精确模式分词
        # Use jieba precise mode for word segmentation
        words = jieba.lcut(text)
        result = []
        for w in words:
            # 去除首尾空白并转为小写
            # Strip whitespace and convert to lowercase
            w = w.strip().lower()
            # 正则匹配：仅保留中文、字母、数字
            # Regex match: keep only Chinese characters, letters and numbers
            if re.search(r'[\w\u4e00-\u9fff]', w):
                result.append(w)
        return result

    # 私有方法：从MySQL加载文本数据，构建BM25索引
    # Private method: Load data from MySQL and build BM25 index
    def _load(self):
        # 连接数据库
        # Connect to MySQL database
        db = mysql.connector.connect(**DB_CONFIG)
        cursor = db.cursor()
        # 查询pdf_chunks表中所有文本内容
        # Query all content fields from pdf_chunks table
        cursor.execute("SELECT content FROM pdf_chunks")
        rows = cursor.fetchall()
        db.close()

        # 数据库无数据时打印警告
        # Print warning if no data in database
        if not rows:
            logger.warning("知识库为空！Knowledge base is empty!")
            return

        # 转换查询结果为字符串列表
        # Convert query result to string list
        self.contents = [str(r[0]) for r in rows]
        logger.info("正在对文档进行中文分词，请稍候... Performing Chinese word segmentation...")
        # 对所有文档批量分词
        # Batch segmentation for all documents
        tokenized_docs = [self._tokenize(doc) for doc in self.contents]
        # 初始化BM25检索模型
        # Initialize BM25 retrieval model
        self.bm25 = BM25Okapi(tokenized_docs)
        logger.info(f"知识库加载完成：共 {len(self.contents)} 段（BM25 + jieba 分词）")
        logger.info(f"Knowledge base loaded: {len(self.contents)} segments (BM25 + jieba)")

    # 检索方法：根据用户问题匹配最相关文本块
    # Retrieval method: Match most relevant text chunks by user question
    def retrieve(self, question):
        # 校验知识库与模型状态
        # Check status of knowledge base and BM25 model
        if not self.contents or self.bm25 is None:
            return []

        # 对用户问题分词
        # Segment user question
        query_tokens = self._tokenize(question)
        # 计算所有文档的BM25相似度分数
        # Calculate BM25 similarity scores for all documents
        bm25_scores = self.bm25.get_scores(query_tokens)
        # 分数降序排序，取前TOP_K个索引
        # Sort scores in descending order and get top-K indices
        top_indices = np.argsort(bm25_scores)[::-1][:TOP_K_RETRIEVAL]
        # 根据索引取出对应文本块
        # Get corresponding text chunks by indices
        results = [self.contents[i] for i in top_indices]

        # 打印检索日志
        # Print retrieval log
        logger.info("=" * 60)
        logger.info(f"用户问题：{question} | User question: {question}")
        logger.info(f"BM25 返回 TOP{len(results)} 条（BM25 分数）：")
        logger.info(f"BM25 Top {len(results)} results (BM25 scores):")
        for i, idx in enumerate(top_indices):
            logger.info(f" 分数 {bm25_scores[idx]:.4f} | {self.contents[idx][:100]}...")
        logger.info("=" * 60)

        return results

# 全局实例化知识库对象
# Instantiate global knowledge base object
kb = KnowledgeBase()

# 调用DeepSeek云端大模型接口
# Call DeepSeek cloud LLM API
def cloud_llm(prompt):
    # 构造请求头：身份认证 + 数据格式
    # Build request headers: authentication & content type
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    # 构造请求体：模型、对话内容、生成参数
    # Build request body: model, message and generation parameters
    data = {
        "model": CLOUD_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,    # 温度参数，0代表回答严谨无创作性 / Temperature: 0 for rigorous answers
        "max_tokens": 1024     # 最大生成长度 / Max generate tokens
    }
    try:
        # 发送POST请求，超时时间120秒
        # Send POST request, timeout 120 seconds
        res = requests.post(API_URL, headers=headers, json=data, timeout=120)
        # 解析返回结果，提取回答内容
        # Parse response and extract answer content
        return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        # 捕获异常，返回错误信息（对应网页访问失败问题）
        # Catch exception and return error message
        return f"调用失败：{str(e)} | API Call Failed: {str(e)}"

# 对话主逻辑：检索知识库 + 拼接提示词 + 调用大模型
# Main chat logic: retrieve knowledge + build prompt + call LLM
def chat(message, history):
    # 检索相关上下文资料
    # Retrieve relevant context
    retrieved = kb.retrieve(message)
    # 拼接多个文本块为上下文
    # Combine multiple chunks into one context string
    context = "\n\n---\n\n".join(retrieved)

    # 构造提示词，约束模型基于资料回答
    # Build prompt, restrict LLM to answer based on provided materials
    prompt = f"""你是无人机法规专家，必须严格根据资料回答。
资料：
{context}

问题：{message}
"""
    # 日志记录完整提示词
    # Log full prompt content
    logger.info("\n【发送给AI】\n【Sent to AI】\n" + prompt)
    # 调用大模型并返回结果
    # Call LLM and return response
    return cloud_llm(prompt)

# 程序入口：启动Gradio Web界面
# Program entry: Launch Gradio web interface
if __name__ == "__main__":
    # 创建聊天界面
    # Create chat interface
    demo = gr.ChatInterface(
        fn=chat,
        title="无人机法规RAG（BM25+Jieba分词·精准检索）",
        examples=["无人机禁飞高度", "上海无人机规定"]
    )
    # 启动服务，监听全网地址，端口7860
    # Launch service, listen on all network interfaces, port 7860
    demo.launch(server_name="0.0.0.0", server_port=7860)