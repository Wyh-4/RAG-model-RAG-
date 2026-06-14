# RAG-model-RAG-
A general RAG intelligent Q&amp;A system based on jieba segmentation, BM25 retrieval and LLM API. It supports text preprocessing, knowledge base search and multi-turn dialogue. Gradio is used to build the web interface quickly. It delivers accurate results, reduces hallucination, and features simple deployment and strong practicability.一款通用抹布智能Q&；一个基于jieba分割、BM25检索和LLM API的系统。它支持文本预处理、知识库搜索和多轮对话。Gradio用于快速构建web界面。结果准确，减少幻觉，部署简单，实用性强。
# 词向量项目总结 / Word Vector Project Summary

> **项目名称**：基于BM25+TF-IDF的无人机法规RAG知识库问答系统  
> **Project Name**: Drone Regulation RAG Knowledge Base Q&A System Based on BM25 + TF-IDF

---

## 一、项目概述 / 1. Project Overview

本项目构建了一个面向中文无人机法规的**检索增强生成（RAG）**系统。系统首先对法规文档进行中文分句、滑动窗口分块和TF-IDF向量化，然后将文本块存入MySQL数据库；在问答阶段，利用BM25算法和Jieba分词进行精准检索，结合云端大语言模型（如DeepSeek）生成专业回答，并通过Gradio提供Web交互界面。

This project builds a **Retrieval-Augmented Generation (RAG)** system for Chinese drone regulations. The system first performs Chinese sentence splitting, sliding-window chunking, and TF-IDF vectorization on regulatory documents, then stores the text chunks in a MySQL database. During the Q&A phase, it uses the BM25 algorithm with Jieba word segmentation for precise retrieval, combines with a cloud large language model (e.g., DeepSeek) to generate professional answers, and provides a web-based interactive interface via Gradio.

---

## 二、项目结构 / 2. Project Structure

```
词向量/
├── 分割词.py           # 文本预处理与向量化入库脚本 / Text preprocessing & vectorization pipeline
├── 云服务RAG.py         # RAG问答主程序（BM25检索+Gradio界面）/ Main RAG app (BM25 retrieval + Gradio UI)
├── test.sql            # 数据库建表脚本 / Database schema script
├── test.txt            # 原始语料文件（约500KB中文法规文本）/ Raw corpus (~500KB Chinese regulatory text)
├── tfidf_model.pkl     # 训练好的TF-IDF模型（持久化）/ Trained TF-IDF model (persisted)
├── RAG具体实现.docx     # RAG实现说明文档 / RAG implementation documentation
└── __pycache__/        # Python字节码缓存 / Python bytecode cache
```

---

## 三、技术架构 / 3. Technical Architecture

### 3.1 整体流程 / Overall Pipeline

```
原始文本 → 中文分句 → 滑动窗口分块 → TF-IDF向量化 → MySQL存储
                                                          ↓
用户问题 → Jieba分词 → BM25检索 → 拼接Prompt → 云端LLM → Web回复
```

---

## 四、模块详解 / 4. Module Details

### 4.1 文本预处理模块 — `分割词.py` / Text Preprocessing Module

| 步骤 / Step | 描述 / Description |
|---|---|
| **文本读取** / Text Reading | 从`test.txt`以UTF-8编码读取全文（约50万字符）/ Read full text from `test.txt` with UTF-8 encoding (~500K chars) |
| **中文分句** / Sentence Splitting | 基于正则表达式按中文标点（`。！？；`）分割句子，保留分隔符 / Split by Chinese punctuation marks (。！？；) using regex, retaining delimiters |
| **滑动窗口分块** / Sliding Window Chunking | 每块400字符，相邻块重叠20%，保证上下文连续性 / 400 chars per chunk, 20% overlap between adjacent chunks for context continuity |
| **TF-IDF向量化** / TF-IDF Vectorization | 使用`sklearn.feature_extraction.text.TfidfVectorizer`在全部文本块上训练TF-IDF模型，学习词汇表与IDF权重 / Train TF-IDF model on all chunks using `TfidfVectorizer` to learn vocabulary and IDF weights |
| **模型持久化** / Model Persistence | 将训练好的向量化器保存为`tfidf_model.pkl`，供后续加载使用 / Save trained vectorizer as `tfidf_model.pkl` for later use |
| **数据入库** / Database Storage | 使用`mysql.connector`将每个文本块及其TF-IDF向量（JSON字符串格式）插入`pdf_chunks`表 / Insert each chunk and its TF-IDF vector (JSON format) into the `pdf_chunks` table via `mysql.connector` |

#### 关键代码逻辑 / Key Code Logic

```python
# 分句函数 / Sentence split function
def split_chinese_sentences(text):
    text = re.sub(r'\s+', '', text)
    pattern = r'(?<=[。！？；])'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]

# 滑动分块函数 / Sliding chunk function
def build_sliding_chunks(sentences, max_len, overlap):
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= max_len:
            current += sent
        else:
            chunks.append(current)
            current = current[-overlap:] + sent
    if current:
        chunks.append(current)
    return chunks
```

---

### 4.2 RAG问答模块 — `云服务RAG.py` / RAG Q&A Module

| 组件 / Component | 描述 / Description |
|---|---|
| **Jieba分词器** / Jieba Tokenizer | 精确模式中文分词，保留中英文字符和数字，过滤标点 / Precise-mode Chinese segmentation, retains Chinese/English chars and digits, filters punctuation |
| **BM25检索** / BM25 Retrieval | 使用`rank_bm25.BM25Okapi`对分词后的文档建立索引，计算查询与文档的BM25相似度，返回Top-K（默认3）最相关文本块 / Build index on tokenized documents, compute BM25 similarity scores, return Top-K (default=3) most relevant chunks |
| **云端LLM调用** / Cloud LLM Call | 通过`requests`库调用DeepSeek API（兼容OpenAI接口格式），温度参数设为0保证回答严谨性 / Call DeepSeek API via `requests` (OpenAI-compatible format), temperature=0 for rigorous answers |
| **Prompt构造** / Prompt Construction | 将检索到的上下文资料与用户问题拼接，约束模型严格基于资料回答 / Concatenate retrieved context with user question, constrain model to answer strictly based on materials |
| **Gradio界面** / Gradio UI | 使用`gr.ChatInterface`搭建Web聊天界面，监听`0.0.0.0:7860`端口 / Build web chat UI with `gr.ChatInterface`, listening on `0.0.0.0:7860` |

#### KnowledgeBase 类结构 / KnowledgeBase Class Structure

```python
class KnowledgeBase:
    def __init__(self):          # 初始化并加载数据 / Initialize and load data
    def _tokenize(text):         # 静态方法：中文分词+过滤 / Static: Chinese word segmentation + filtering
    def _load(self):            # 从MySQL加载数据，构建BM25索引 / Load data from MySQL, build BM25 index
    def retrieve(self, question): # BM25检索，返回Top-K相关文本块 / BM25 retrieval, return Top-K chunks
```

#### Chat流程 / Chat Flow

```python
def chat(message, history):
    retrieved = kb.retrieve(message)          # 1. BM25检索相关文档 / Retrieve relevant docs
    context = "\n\n---\n\n".join(retrieved)    # 2. 拼接上下文 / Combine context
    prompt = f"""你是无人机法规专家...       # 3. 构造专家提示词 / Build expert prompt
资料：{context}
问题：{message}"""
    return cloud_llm(prompt)                   # 4. 调用云端LLM生成回答 / Call cloud LLM
```

---

### 4.3 数据库设计 — `test.sql` / Database Design

| 字段 / Field | 类型 / Type | 说明 / Description |
|---|---|---|
| `id` | INT UNSIGNED AUTO_INCREMENT | 自增主键 / Primary key |
| `content` | LONGTEXT | 文本块内容 / Text chunk content |
| `vector` | LONGTEXT | TF-IDF向量（JSON字符串）/ TF-IDF vector as JSON string |
| `source` | VARCHAR(255) | 数据来源文件名 / Source file name |
| `create_time` | DATETIME | 入库时间戳 / Insertion timestamp |

```sql
CREATE TABLE IF NOT EXISTS pdf_chunks (
  id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  content LONGTEXT NOT NULL,
  vector LONGTEXT NOT NULL,
  source VARCHAR(255) NOT NULL DEFAULT '',
  create_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;
```

---

## 五、技术栈 / 5. Tech Stack

| 层级 / Layer | 技术 / Technology | 用途 / Purpose |
|---|---|---|
| **分词** / Segmentation | `jieba` | 中文精确模式分词 / Chinese precise-mode tokenization |
| **检索算法** / Retrieval | `rank_bm25` (BM25Okapi) | 稀疏检索，基于词频的相似度排序 / Sparse retrieval, term-frequency based ranking |
| **向量化** / Vectorization | `scikit-learn TfidfVectorizer` | TF-IDF特征提取 / TF-IDF feature extraction |
| **数据库** / Database | `MySQL 8.0+ (utf8mb4)` | 文本块与向量持久化存储 / Persistent storage for chunks and vectors |
| **大语言模型** / LLM | `DeepSeek API` | 基于检索上下文生成专业回答 / Generate professional answers based on retrieved context |
| **Web界面** / Web UI | `Gradio (ChatInterface)` | 聊天式交互界面 / Chat-style interactive interface |
| **数值计算** / Computing | `numpy` | BM25分数排序与索引处理 / Score sorting and index processing |

---

## 六、核心特点 / 6. Key Features

1. **中文深度优化 / Chinese-Optimized**：使用Jieba分词+BM25稀疏检索，专门针对中文场景调优，无需GPU即可高效运行 / Uses Jieba + BM25 for Chinese-specific optimization, runs efficiently without GPU.

2. **滑动窗口分块 / Sliding Window Chunking**：20%重叠率保证上下文连续性，避免关键信息被截断 / 20% overlap ensures context continuity, preventing critical information from being truncated.

3. **检索增强生成 / Retrieval-Augmented Generation**：BM25检索提供精准的领域知识支撑，LLM基于实际资料回答，减少幻觉 / BM25 retrieval provides precise domain knowledge, LLM answers based on actual materials, reducing hallucinations.

4. **模块化设计 / Modular Design**：预处理、检索、LLM调用、Web界面各自独立，易于维护和扩展 / Preprocessing, retrieval, LLM call, and web UI are independent modules, easy to maintain and extend.

5. **零温度严谨回答 / Zero-Temperature Rigorous Answers**：`temperature=0.0`确保法律场景下回答的确定性和可靠性 / `temperature=0.0` ensures deterministic and reliable answers for legal scenarios.

---

## 七、运行方式 / 7. How to Run

### 步骤 1：准备数据库 / Step 1: Prepare Database

```bash
mysql -u root -p < test.sql
```

### 步骤 2：文本预处理入库 / Step 2: Text Preprocessing

```bash
python 分割词.py
```

执行后将完成分句、分块、TF-IDF训练与数据入库 / This will complete sentence splitting, chunking, TF-IDF training, and data storage.

### 步骤 3：启动RAG问答服务 / Step 3: Launch RAG Q&A Service

```bash
pip install jieba rank_bm25 numpy mysql-connector-python gradio requests scikit-learn
python 云服务RAG.py
```

访问 `http://localhost:7860` 即可使用聊天界面 / Visit `http://localhost:7860` to use the chat interface.

---

## 八、依赖清单 / 8. Dependencies

| 包名 / Package | 版本要求 / Version |
|---|---|
| `scikit-learn` | 用于TF-IDF向量化 / For TF-IDF vectorization |
| `mysql-connector-python` | MySQL数据库驱动 / MySQL driver |
| `numpy` | 数值计算 / Numerical computing |
| `rank_bm25` | BM25检索算法 / BM25 retrieval algorithm |
| `jieba` | 中文分词 / Chinese word segmentation |
| `gradio` | Web交互界面 / Web interactive UI |
| `requests` | HTTP API调用 / HTTP API calls |
| `joblib` | 模型持久化 / Model persistence |

---

## 九、适用场景 / 9. Use Cases

- 📋 无人机法规智能问答 / Drone regulation intelligent Q&A
- 📚 企业知识库检索系统 / Enterprise knowledge base retrieval system
- ⚖️ 法律/合规文档RAG系统 / Legal/compliance document RAG system
- 🏛️ 政府规章制度问答助手 / Government regulation Q&A assistant
- 🔍 中文长文档语义检索 / Chinese long-document semantic retrieval

---

## 十、扩展方向 / 10. Future Improvements

| 方向 / Direction | 描述 / Description |
|---|---|
| **稠密向量检索** / Dense Retrieval | 引入Embedding模型（如BGE/M3E）实现语义级稠密检索，与BM25混合使用 / Introduce embedding models (e.g., BGE/M3E) for dense semantic retrieval, hybrid with BM25 |
| **重排序** / Re-ranking | 在BM25粗排后增加Cross-Encoder精排，提升Top结果准确率 / Add Cross-Encoder re-ranking after BM25 coarse ranking to improve top result accuracy |
| **多格式支持** / Multi-format | 支持PDF、Word、网页等多样化的文档格式解析 / Support PDF, Word, HTML and other document formats |
| **对话历史** / Conversation History | 引入多轮对话记忆，支持上下文连续问答 / Introduce multi-turn conversation memory for contextual follow-up Q&A |
| **向量数据库** / Vector Database | 迁移至Milvus/Qdrant等专业向量数据库，提升大规模检索性能 / Migrate to professional vector databases (Milvus/Qdrant) for large-scale retrieval |
| **流式输出** / Streaming Output | 实现LLM回答的流式返回，改善用户体验 / Implement streaming response from LLM to improve user experience |

---

> 📅 项目日期 / Project Date: 2026-06  
> 📝 文档生成 / Document Generated: 2026-06-14
