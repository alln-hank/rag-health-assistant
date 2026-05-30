import json
import jieba
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage
from rank_bm25 import BM25Okapi
from dashscope import TextReRank
import os
import numpy as np
from dotenv import load_dotenv

# ================= 配置（请根据你的实际情况修改）=================
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
if DASHSCOPE_API_KEY:
    os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY
else:
    raise RuntimeError("未设置 DASHSCOPE_API_KEY。请先配置环境变量，或参考 .env.example。")

CHROMA_DIR = "./chroma_db"
EVAL_DATA_PATH = "eval_data.json"

# 检索参数（你可以修改这些值，跑完看效果）
HYBRID_TOP_K = 10      # 混合检索返回的候选数
RERANK_TOP_K = 5       # 重排序后保留的文档数（最终给LLM的）
BM25_WEIGHT = 0.4      # 混合检索中 BM25 的权重

# LLM 配置（用于生成答案，评估时需要调用一次）
LLM = ChatTongyi(model="qwen-max", temperature=0.1)
EMBEDDINGS = DashScopeEmbeddings(model="text-embedding-v1")

# ================= 加载知识库 =================
print("正在加载知识库...")
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=EMBEDDINGS)
all_data = db.get()
bm25_docs = all_data.get('documents', [])
if bm25_docs:
    tokenized_docs = [list(jieba.cut(doc)) for doc in bm25_docs]
    bm25_index = BM25Okapi(tokenized_docs)
    print(f"✅ 知识库加载完成，共 {len(bm25_docs)} 个文档片段")
else:
    raise ValueError("知识库为空，请先构建知识库！")

# ================= 复制你的检索函数（避免循环导入） =================
def hybrid_search(query, top_k=HYBRID_TOP_K):
    tokenized_query = list(jieba.cut(query))
    bm25_scores = bm25_index.get_scores(tokenized_query)
    vector_results = db.similarity_search_with_score(query, k=top_k)

    if len(bm25_scores) == 0:
        return [doc for doc, _ in vector_results[:top_k]]

    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    doc_map = {}
    for doc, score in vector_results:
        content = doc.page_content
        vec_sim = 1.0 / (1.0 + score)
        doc_map[content] = {'doc': doc, 'vector_score': vec_sim, 'bm25_score': 0.0}

    max_bm25 = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
    for idx in top_bm25_indices:
        content = bm25_docs[idx]
        bm25_norm = bm25_scores[idx] / max_bm25
        if content in doc_map:
            doc_map[content]['bm25_score'] = bm25_norm
        else:
            from langchain_core.documents import Document
            doc_map[content] = {'doc': Document(page_content=content), 'vector_score': 0.0, 'bm25_score': bm25_norm}

    results = []
    for content, scores in doc_map.items():
        final_score = (1 - BM25_WEIGHT) * scores['vector_score'] + BM25_WEIGHT * scores['bm25_score']
        results.append((scores['doc'], final_score))
    results.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in results[:top_k]]

def rerank(query, docs, top_k=RERANK_TOP_K):
    if not docs:
        return []
    documents = [doc.page_content for doc in docs]
    resp = TextReRank.call(model='gte-rerank', query=query, documents=documents, top_n=top_k)
    if resp.status_code == 200:
        indices = [item['index'] for item in resp.output['results']]
        return [docs[i] for i in indices]
    else:
        print(f"重排序失败：{resp.message}，使用原始排序")
        return docs[:top_k]

# ================= 评估指标计算 =================
def compute_recall_at_k(query, relevant_keywords, top_docs):
    """
    如果 top_docs 中至少有一个文档的 page_content 包含任意一个相关关键词，则视为命中。
    """
    for doc in top_docs:
        for kw in relevant_keywords:
            if kw in doc.page_content:
                return 1.0
    return 0.0

def compute_answer_coverage(generated_answer, expected_keywords):
    if not expected_keywords:
        return 1.0
    answer_lower = generated_answer.lower()
    covered = 0
    for kw in expected_keywords:
        # 只要答案中包含关键词的任意 2/3 字符，就算覆盖
        kw_chars = set(kw)
        matched = sum(1 for c in kw_chars if c in answer_lower)
        if matched >= len(kw_chars) * 0.6:   # 60% 的字符匹配即可
            covered += 1
    return covered / len(expected_keywords)

# ================= 生成答案的函数 =================
def generate_answer(question, top_docs):
    context = "\n\n---\n\n".join([doc.page_content for doc in top_docs])
    prompt = f"""你是一个专业的健康养生科普助手。请根据提供的【参考资料】回答用户问题。
如果资料足够，请**尽可能完整地**列出所有相关要点、方法或注意事项，**不要遗漏任何关键信息**。
如果资料无法回答问题，回复“根据现有资料无法回答”。

参考资料：{context}

用户问题：{question}
助手："""
    response = LLM.invoke([HumanMessage(content=prompt)])
    return response.content


# ================= 主评估流程 =================
def main():
    with open(EVAL_DATA_PATH, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    total_recall = 0.0
    total_coverage = 0.0
    count = len(eval_data)

    for i, item in enumerate(eval_data, 1):
        q = item['question']
        expected_kw = item.get('expected_keywords', [])
        print(f"\n[{i}/{count}] 正在评估: {q}")

        # 检索
        candidates = hybrid_search(q, top_k=HYBRID_TOP_K)
        top_docs = rerank(q, candidates, top_k=RERANK_TOP_K)

        # 计算 Recall (这里用关键词近似)
        recall = compute_recall_at_k(q, expected_kw, top_docs)
        total_recall += recall
        print(f"  Recall@{RERANK_TOP_K}: {recall:.2f}")

        # 生成答案
        generated_answer = generate_answer(q, top_docs)
        # 可打印生成答案供观察（调试时打开）
        # print(f"  生成答案: {generated_answer[:100]}...")

        # 计算覆盖率
        coverage = compute_answer_coverage(generated_answer, expected_kw)
        total_coverage += coverage
        print(f"  答案覆盖率: {coverage:.2f}")

    avg_recall = total_recall / count
    avg_coverage = total_coverage / count
    print("\n" + "="*50)
    print(f"评估完成！共 {count} 个问题")
    print(f"平均 Recall@{RERANK_TOP_K}: {avg_recall:.3f}")
    print(f"平均答案覆盖率: {avg_coverage:.3f}")
    print("="*50)

if __name__ == "__main__":
    main()
