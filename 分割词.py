# Import TF-IDF text vectorization tool from Scikit-learn
#  scikit-learn 导入 TF-IDF 文本向量化工具
from sklearn.feature_extraction.text import TfidfVectorizer
# Import MySQL database connection driver
# 导入 MySQL 数据库连接驱动
import mysql.connector
# Import JSON module, convert vector list to string for storage
# 导入 JSON 模块，用于将向量列表转换为字符串存储
import json
# Import regular expression module for text cleaning and sentence splitting
# 导入正则表达式模块，用于文本清洗和分句
import re
# Import joblib for persisting trained TF-IDF model
# 导入 joblib 模块，用于持久化保存训练好的 TF-IDF 模型
import joblib

# Database host address
# 数据库主机地址
DB_HOST = "localhost"
# Database username
# 数据库用户名
DB_USER = "root"
# Database password
# 数据库密码
DB_PASSWORD = "123456"
# Database name
# 数据库名称
DB_NAME = "my_vector_db"

# Open test.txt in read-only mode with utf-8 encoding, read full content
# 以只读方式打开 test.txt 文件，编码为 utf-8，读取全部内容
with open("test.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
# Print total character length of loaded text
# 打印读取到的文本长度（字符数）
print(f"Text length: {len(raw_text)} characters | 读取文本长度：{len(raw_text)} 字符")

# Define Chinese sentence splitting function
# 定义中文分句函数
def split_chinese_sentences(text):
    # Remove all whitespace characters (space, line break, tab etc.)
    # 去除文本中所有空白字符（空格、换行、制表符等）
    text = re.sub(r'\s+', '', text)
    # Define split symbols: Chinese period, question mark, exclamation mark, semicolon
    # 定义分句分隔符：中文句号、问号、感叹号、分号
    pattern = r'(?<=[。！？；])'
    # Split text by regex (positive lookbehind to retain delimiters)
    # 使用正则表达式进行分割，保留分隔符（正向肯定环视）
    sentences = re.split(pattern, text)
    # Filter empty strings and trim extra spaces
    # 返回去除首尾空白后的非空句子列表
    return [s.strip() for s in sentences if s.strip()]

# Split raw text into sentence list
# 调用分句函数，将原始文本分割为句子列表
sentences = split_chinese_sentences(raw_text)
# Print total number of split sentences
# 打印分句结果的数量
print(f"Split completed: {len(sentences)} sentences | 分句完成：{len(sentences)} 句")

# Target character length per text chunk
# 设置每个文本块的目标长度（字符数）
chunk_size = 400
# Overlap length: 20% of chunk size for context continuity
# 计算重叠长度，为目标长度的20%（用于保持上下文连续）
overlap_size = int(chunk_size * 0.2)

# Build sliding window chunks with overlap based on sentences
# 定义滑动分块函数，基于句子构建重叠的文本块
def build_sliding_chunks(sentences, max_len, overlap):
    chunks = []          # Store final text chunks | 存储生成的文本块
    current = ""         # Temporary string for current chunk | 当前正在构建的块内容
    # Traverse each sentence
    # 遍历每一个句子
    for sent in sentences:
        # Append sentence if total length does not exceed limit
        # 如果当前块加上新句子的长度未超过最大长度限制，则直接拼接
        if len(current) + len(sent) <= max_len:
            current += sent
        else:
            # Save current chunk and start new chunk with overlapping content
            # 否则将当前块保存为一个完成的块，新块从上一个块末尾重叠部分开始
            chunks.append(current)
            current = current[-overlap:] + sent
    # Add the last unfinished chunk
    # 循环结束后，如果当前块非空，则作为最后一个块添加
    if current:
        chunks.append(current)
    return chunks

# Generate overlapping text chunks
# 调用滑动分块函数，将句子列表组合成文本块
chunks = build_sliding_chunks(sentences, chunk_size, overlap_size)
# Print chunk quantity and overlap ratio
# 打印分块结果的数量，并提示重叠比例
print(f"Chunk completed: {len(chunks)} chunks (20% overlap) | 分块完成：{len(chunks)} 块，重叠20%")

# Initialize TF-IDF vectorizer with default parameters
# 实例化 TF-IDF 向量化器（采用默认参数）
vectorizer = TfidfVectorizer()
# Train TF-IDF model on all text chunks (learn vocabulary & IDF weight)
# 使用所有文本块拟合向量化模型（学习词汇表及 IDF 权重）
vectorizer.fit(chunks)
# Save trained TF-IDF model to local file
# 将训练好的模型保存到文件 tfidf_model.pkl，后续 RAG 系统可加载使用
joblib.dump(vectorizer, "tfidf_model.pkl")
print("TF-IDF model saved as tfidf_model.pkl | TF-IDF 模型已保存为 tfidf_model.pkl")

# Establish MySQL database connection
# 建立与 MySQL 数据库的连接
db = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_NAME
)
# Create cursor for executing SQL statements
# 创建游标对象，用于执行 SQL 语句
cursor = db.cursor()

# Traverse all text chunks and insert into database
# 遍历每一个文本块，执行入库操作
for chunk in chunks:
    # Convert text chunk to TF-IDF dense vector
    # 将当前块转换为 TF-IDF 稠密向量
    vec = vectorizer.transform([chunk]).toarray()[0].tolist()
    # SQL insert statement: content, vector(json string), source
    # 构造插入语句：向 pdf_chunks 表插入内容、向量、来源三个字段
    sql_insert = "INSERT INTO pdf_chunks (content, vector, source) VALUES (%s, %s, %s)"
    # Execute insert, convert vector list to JSON string for storage
    # 执行插入操作，向量转为 JSON 字符串存储
    cursor.execute(sql_insert, (chunk, json.dumps(vec), "article.txt"))

# Commit all insert transactions
# 提交所有插入操作，数据持久化到数据库
db.commit()
# Close database connection
# 关闭数据库连接
db.close()
print("\nAll data imported successfully! | 全部入库完成！")