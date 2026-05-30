import os
import time
import logging
from logging.handlers import RotatingFileHandler
from collections import defaultdict, OrderedDict
import asyncio
import base64
import traceback

import gradio as gr
import tempfile
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatTongyi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from rank_bm25 import BM25Okapi
import jieba
from dashscope import TextReRank, MultiModalConversation
import dashscope

load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
if DASHSCOPE_API_KEY:
    os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY
else:
    print("警告：未设置 DASHSCOPE_API_KEY，调用通义千问/DashScope 时会失败。请参考 .env.example 配置。")

# ========== 日志配置 ==========
logger = logging.getLogger("health_rag")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler = RotatingFileHandler("app.log", maxBytes=1*1024*1024, backupCount=5, encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ========== 初始化大模型（文本） ==========
llm = ChatTongyi(model="qwen-max", temperature=0.1)

# ========== 全局变量 ==========
db = None
bm25_index = None
bm25_docs = []

# ========== 缓存系统 ==========
answer_cache = OrderedDict()
MAX_CACHE_SIZE = 200

def get_cache(question):
    if question in answer_cache:
        answer_cache.move_to_end(question)
        return answer_cache[question]
    return None

def set_cache(question, answer):
    if len(answer_cache) >= MAX_CACHE_SIZE:
        answer_cache.popitem(last=False)
    answer_cache[question] = answer

# ========== 频率限制 ==========
rate_limit = defaultdict(list)
RATE_LIMIT = 10
RATE_WINDOW = 60

def is_rate_limited(ip):
    now = time.time()
    requests = rate_limit[ip]
    while requests and requests[0] < now - RATE_WINDOW:
        requests.pop(0)
    if len(requests) >= RATE_LIMIT:
        return True
    requests.append(now)
    return False

MAX_INPUT_LENGTH = 200

# ========== 实体扩充字典 ==========
ENTITY_EXPANSION = {
    "头痛": "风寒头痛 肝阳头痛 血虚头痛 痰浊头痛",
    "失眠": "心肾不交失眠 肝火扰心失眠 心脾两虚失眠",
    "咳嗽": "风寒咳嗽 风热咳嗽 痰湿咳嗽 阴虚咳嗽",
    "便秘": "热秘 气秘 虚秘 冷秘",
    "腹痛": "胃寒腹痛 食积腹痛 气滞腹痛",
    "月经不调": "肝郁月经不调 血虚月经不调 肾虚月经不调",
    "腰痛": "寒湿腰痛 肾虚腰痛 血瘀腰痛",
    "头晕": "气血亏虚头晕 肝阳上亢头晕 痰湿中阻头晕",
    "胃痛": "寒邪犯胃 食滞胃痛 肝气犯胃 脾胃虚寒胃痛",
    "乏力": "气虚乏力 湿困乏力 阳虚乏力",
}

def expand_query(user_input):
    expansions = []
    for symptom, mapping in ENTITY_EXPANSION.items():
        if symptom in user_input:
            expansions.append(mapping)
    if expansions:
        return user_input + " " + " ".join(expansions)
    return user_input

# ===================== 知识库构建 =====================
def build_knowledge_base(files):
    global db, bm25_index, bm25_docs
    if not files:
        return "请先上传文件！"
    documents = []
    for file_path in files:
        try:
            if file_path.endswith(".pdf"):
                docs = PyPDFLoader(file_path).load()
            elif file_path.endswith(".docx"):
                docs = Docx2txtLoader(file_path).load()
            elif file_path.endswith(".txt") or file_path.endswith(".md"):
                docs = TextLoader(file_path, encoding="utf-8").load()
            else:
                continue
            documents.extend(docs)
        except Exception as e:
            error_msg = f"加载文件失败：{file_path}，错误详情：{str(e)}"
            print(error_msg)
            return f"❌ {error_msg}"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=80,
        separators=["\n\n", "\n", "。", "，"]
    )
    texts = text_splitter.split_documents(documents)
    embeddings = DashScopeEmbeddings(model="text-embedding-v1")
    if db is not None:
        db.add_documents(texts)
        new_texts = [t.page_content for t in texts]
        bm25_docs.extend(new_texts)
        tokenized_docs = [list(jieba.cut(doc)) for doc in bm25_docs]
        bm25_index = BM25Okapi(tokenized_docs)
        logger.info(f"追加 {len(texts)} 个片段，BM25 已更新")
    else:
        db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
        bm25_docs = [t.page_content for t in texts]
        tokenized_docs = [list(jieba.cut(doc)) for doc in bm25_docs]
        bm25_index = BM25Okapi(tokenized_docs)
        logger.info("新知识库创建完成")
    return "✅ 知识库更新完成！可以开始提问了！"

# ===================== 混合检索+重排序 =====================
def hybrid_search(query, db, bm25_index, bm25_docs, top_k=10):
    if not query or not query.strip():
        return []
    vector_results = db.similarity_search_with_score(query, k=top_k)
    tokenized_query = list(jieba.cut(query))
    bm25_scores = bm25_index.get_scores(tokenized_query)
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
            fake_doc = Document(page_content=content)
            doc_map[content] = {'doc': fake_doc, 'vector_score': 0.0, 'bm25_score': bm25_norm}
    results = []
    for content, scores in doc_map.items():
        final_score = 0.6 * scores['vector_score'] + 0.4 * scores['bm25_score']
        results.append((scores['doc'], final_score))
    results.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in results[:top_k]]

def rerank(query, docs, top_k=5):
    if not docs or not query:
        return docs[:top_k] if docs else []
    documents = [doc.page_content for doc in docs]
    resp = TextReRank.call(model='gte-rerank', query=query, documents=documents, top_n=top_k)
    if resp.status_code == 200:
        indices = [item['index'] for item in resp.output['results']]
        return [docs[i] for i in indices]
    else:
        logger.warning(f"重排序失败：{resp.message}")
        return docs[:top_k]

# ===================== 会话管理 =====================
DEFAULT_COLLECTED_INFO = {
    "symptom": None,
    "duration": None,
    "accompany": None,
    "habits": None,
    "gender": None,
    "age": None
}

def create_session(sessions, name_prefix="对话"):
    new_id = max(sessions.keys(), default=0) + 1
    sessions[new_id] = {
        "name": f"{name_prefix}{new_id}",
        "history": [],
        "first_question": None,
        "dialog_state": "idle",
        "collected_info": DEFAULT_COLLECTED_INFO.copy(),
        "ask_step": 0
    }
    return sessions, new_id

def get_session_display_name(session):
    if session["first_question"]:
        q = session["first_question"].strip()
        summary = q[:20] + "..." if len(q) > 20 else q
        return f"{session['name']}: {summary}"
    return session["name"]

# ===================== 问诊逻辑 =====================
DISCLAIMER = "\n\n⚠️ 本建议不能替代医生诊断，如果症状持续或加重，请及时就医。"

ASKING_QUESTIONS = {
    1: "这种情况持续多久了？",
    2: "还有哪些伴随症状？（如恶心、怕冷、乏力等）",
    3: "平时有什么生活习惯？（如熬夜、饮食不规律、压力大等）"
}

def handle_dialog(session, user_input, user_profile):
    state = session["dialog_state"]
    info = session["collected_info"]
    step = session["ask_step"]

    if user_input.strip() in ["重新开始", "结束问诊", "退出"]:
        session["dialog_state"] = "idle"
        session["ask_step"] = 0
        session["collected_info"] = DEFAULT_COLLECTED_INFO.copy()
        return "好的，问诊已结束。有什么可以帮您的？", False, None

    if state == "idle":
        if is_health_query(user_input):
            info["symptom"] = user_input
            if user_profile.get("age"):
                info["age"] = user_profile["age"]
            if user_profile.get("gender") and user_profile["gender"] != "保密":
                info["gender"] = user_profile["gender"]
            session["dialog_state"] = "asking"
            session["ask_step"] = 1
            return ASKING_QUESTIONS[1], False, None
        else:
            return None, True, user_input

    elif state == "asking":
        if step == 1:
            info["duration"] = user_input
            session["ask_step"] = 2
            return ASKING_QUESTIONS[2], False, None
        elif step == 2:
            info["accompany"] = user_input
            session["ask_step"] = 3
            return ASKING_QUESTIONS[3], False, None
        elif step == 3:
            info["habits"] = user_input
            parts = []
            if info.get("symptom"):
                parts.append(f"症状：{info['symptom']}")
            if info.get("duration"):
                parts.append(f"持续时间：{info['duration']}")
            if info.get("accompany"):
                parts.append(f"伴随症状：{info['accompany']}")
            if info.get("habits"):
                parts.append(f"生活习惯：{info['habits']}")
            if info.get("age"):
                parts.append(f"年龄：{info['age']}岁")
            if info.get("gender"):
                parts.append(f"性别：{info['gender']}")
            final_query = "，".join(parts) + " 调理建议"
            session["dialog_state"] = "idle"
            session["ask_step"] = 0
            return None, True, final_query
        else:
            session["dialog_state"] = "idle"
            session["ask_step"] = 0
            return "系统异常，请重新描述您的问题。", False, None

def is_health_query(text):
    if not text:
        return False
    health_keywords = ["头痛", "头晕", "失眠", "胃痛", "咳嗽", "便秘", "腹泻", "乏力",
                       "腰酸", "腿痛", "感冒", "发烧", "恶心", "胸闷", "心慌", "月经",
                       "关节", "皮肤", "湿疹", "上火", "口干", "口苦", "脱发", "耳鸣"]
    return any(kw in text for kw in health_keywords)

# ===================== 多模态：图像分析 =====================
async def analyze_image(image_path: str) -> str:
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        messages = [{
            "role": "user",
            "content": [
                {"image": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"},
                {"text": "请用中文简要描述这张图片的内容，重点关注与健康、养生相关的特征（例如舌苔颜色、厚薄、裂纹，或食材名称、新鲜度等）。"}
            ]
        }]
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: MultiModalConversation.call(model='qwen-vl-plus', messages=messages)
        )
        if response.status_code == 200:
            desc = response.output.choices[0].message.content[0]["text"]
            print(f"✅ 图片描述成功：{desc}")
            return desc
        else:
            error_detail = f"多模态API返回错误：状态码={response.status_code}，消息={response.message}"
            print(error_detail)
            return f"图片分析失败：{error_detail}"
    except Exception as e:
        error_detail = f"图片处理异常：{str(e)}"
        print(error_detail)
        traceback.print_exc()
        return f"图片分析异常：{error_detail}"

def is_image_analysis_error(result: str) -> bool:
    if not result:
        return True
    error_prefixes = (
        "图片分析失败：",
        "图片分析异常：",
        "图片处理异常：",
    )
    return any(result.startswith(prefix) for prefix in error_prefixes)

# ===================== 核心聊天函数 =====================
async def respond(message, chat_history, sessions, current_id, user_profile, request: gr.Request, image=None):
    global db, bm25_index, bm25_docs

    client_ip = request.client.host if request else "unknown"

    if not message or not message.strip():
        message = "请分析我提供的舌象/食材描述"

    if len(message) > MAX_INPUT_LENGTH:
        error_msg = f"问题过长，请限制在 {MAX_INPUT_LENGTH} 字以内。"
        chat_history.append({"role": "user", "content": message[:MAX_INPUT_LENGTH] + "..."})
        chat_history.append({"role": "assistant", "content": error_msg})
        return "", chat_history, sessions, gr.update(), f"当前在：{sessions[current_id]['name']}"

    if is_rate_limited(client_ip):
        error_msg = "请求过于频繁，请稍后再试。"
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": error_msg})
        logger.warning(f"频率限制触发 IP: {client_ip}")
        return "", chat_history, sessions, gr.update(), f"当前在：{sessions[current_id]['name']}"

    # ---------- 多模态分析 ----------
    image_desc = ""
    if image is not None:
        logger.info("开始分析用户上传的图片...")
        image_desc = await analyze_image(image)
        if not is_image_analysis_error(image_desc):
            logger.info(f"图片描述成功：{image_desc}")
        else:
            logger.warning(f"图片分析未成功：{image_desc}")

    session = sessions[current_id]

    # 缓存（无图片时）
    if session.get("dialog_state") == "idle" and not is_health_query(message) and not image:
        cached = get_cache(message)
        if cached:
            logger.info(f"缓存命中 IP:{client_ip} | 问题:{message[:50]}")
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": cached})
            session["history"] = chat_history
            choices = [(get_session_display_name(sessions[k]), k) for k in sessions]
            dropdown_update = gr.update(choices=choices, value=current_id)
            return "", chat_history, sessions, dropdown_update, f"当前在：{session['name']}"

    if db is None or bm25_index is None:
        fallback = "请先上传并构建知识库！"
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": fallback})
        session["history"] = chat_history
        return "", chat_history, sessions, gr.update(), f"当前在：{session['name']}"

    if not chat_history and message.strip():
        session["first_question"] = message
        session["name"] = f"对话{current_id}"

    # ---------- 问诊流程（有图片跳过追问） ----------
    if image is not None:
        dialog_response, should_search, final_query = None, True, message
    else:
        dialog_response, should_search, final_query = handle_dialog(session, message, user_profile)

    if dialog_response is not None:
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": dialog_response})
        session["history"] = chat_history
        choices = [(get_session_display_name(sessions[k]), k) for k in sessions]
        dropdown_update = gr.update(choices=choices, value=current_id)
        return "", chat_history, sessions, dropdown_update, f"当前在：{session['name']}"

    if not should_search:
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": "系统错误，请重试。"})
        session["history"] = chat_history
        return "", chat_history, sessions, gr.update(), f"当前在：{session['name']}"

    if not final_query or not final_query.strip():
        final_query = message

    input_source = "text"

    # ✅ 核心改动：将图片识别结果融入查询，同时保留“来源是图片”这个事实
    if image_desc and not is_image_analysis_error(image_desc):
        input_source = "image"
        # 构造一个完整的用户描述，不再用括号附加，而是直接融入
        if "舌" in message or "苔" in message:
            final_query_with_image = f"用户上传了一张舌象图片，图像识别结果如下：{image_desc}。请根据这张图片反映出的舌象帮我分析健康状况并给出调理建议。"
        else:
            final_query_with_image = f"{final_query}。用户还上传了一张图片，图像识别结果如下：{image_desc}。请结合这张图片进行分析。"
    else:
        final_query_with_image = final_query

    expanded_query = expand_query(final_query_with_image)
    logger.info(f"检索查询（扩充后）: {expanded_query}")

    loop = asyncio.get_event_loop()
    top_docs = await loop.run_in_executor(
        None,
        lambda: rerank(expanded_query, hybrid_search(expanded_query, db, bm25_index, bm25_docs, top_k=10), top_k=5)
    )

    context = "\n\n---\n\n".join([doc.page_content for doc in top_docs]) if top_docs else "暂无相关知识库资料。"

    history_text = ""
    for turn in chat_history[-4:]:
        if turn["role"] == "user":
            history_text += f"用户：{turn['content']}\n"
        else:
            history_text += f"助手：{turn['content']}\n"

    profile_text = ""
    if user_profile.get("age"):
        profile_text += f"用户年龄：{user_profile['age']}岁。"
    if user_profile.get("gender") and user_profile["gender"] != "保密":
        profile_text += f"性别：{user_profile['gender']}。"
    if user_profile.get("health"):
        profile_text += f"健康状况：{user_profile['health']}。"
    if profile_text:
        profile_text = "【用户背景】" + profile_text + "\n"

    source_hint = "用户上传了图片，你看到的是图像识别后的内容，不要把它说成“用户自己文字描述的舌象”。回答时应表述为“根据你上传的图片”或“从图片来看”。" if input_source == "image" else "用户提供的是文字描述。回答时可以表述为“根据你的描述”。"

    # 强制分析规则：如果问题中包含“舌象”“观察描述”等，直接回答，不反问
    prompt = f"""你是一个专业的健康养生科普助手。请根据下面提供的【参考资料】回答用户问题。

{profile_text}规则（非常重要）：
1. 如果用户已经提供了具体的症状描述或观察记录（例如舌象颜色、厚薄、干湿等），你必须直接基于这些信息给出分析和建议，严禁说“需要更多信息”或“请补充细节”。
2. 如果用户问题很模糊且没有提供任何具体信息，才可以礼貌地追问。
3. 结合用户背景和描述，给出个性化建议，要具体到颜色、厚薄等特征。
4. 严格基于资料，不得编造。
5. 如果资料完全没有覆盖，回复“根据现有资料无法回答”。
6. 如果是健康建议，末尾添加：“{DISCLAIMER}”
7. {source_hint}
8. 回答开头要和信息来源一致：
 - 如果用户上传的是图片，就用“根据你上传的图片”或“从这张图片来看”这类说法。
 - 如果用户提供的是文字，就用“根据你的描述”这类说法。

历史对话：
{history_text}

参考资料：
{context}

用户问题（含具体描述）：{final_query_with_image}
助手："""

    response = await loop.run_in_executor(
        None,
        lambda: llm.invoke([HumanMessage(content=prompt)])
    )
    answer = response.content

    if "本建议不能替代医生诊断" not in answer:
        answer += DISCLAIMER

    ref_text = ""
    if top_docs:
        ref_text = "\n\n**📚 参考资料引用：**\n"
        for i, doc in enumerate(top_docs, 1):
            snippet = doc.page_content[:150].replace("\n", " ")
            ref_text += f"- 片段{i}：{snippet}...\n"

    full_answer = answer + ref_text

    acupoint_images = {
        "足三里": "static/images/zusanli.jpg",
        "涌泉穴": "static/images/yongquan.jpg",
    }
    for point, img_path in acupoint_images.items():
        if point in answer:
            full_answer += f"\n![{point}示意图](/{img_path})"
            break

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": full_answer})
    session["history"] = chat_history

    if final_query == message and not image:
        set_cache(message, full_answer)

    log_docs = " | ".join([doc.page_content[:30].replace("\n", " ") for doc in top_docs[:3]]) if top_docs else "无"
    logger.info(
        f"IP:{client_ip} | 会话:{current_id} | 查询:{final_query_with_image[:50]} | "
        f"检索片段:{log_docs} | 回答:{answer[:100]}"
    )

    choices = [(get_session_display_name(sessions[k]), k) for k in sessions]
    dropdown_update = gr.update(choices=choices, value=current_id)
    display_text = f"当前在：{session['name']}"

    return "", chat_history, sessions, dropdown_update, display_text

# ===================== 导出对话 =====================
def export_chat(chat_history):
    if not chat_history:
        return None
    md_content = "# 健康养生对话记录\n\n"
    for msg in chat_history:
        role = "👤 用户" if msg["role"] == "user" else "🤖 助手"
        md_content += f"**{role}**：{msg['content']}\n\n"
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8")
    tmp.write(md_content)
    tmp.close()
    return tmp.name

# ===================== Gradio 界面 =====================
# ===================== Gradio ?? =====================
custom_css = """
:root {
  --bg-0: #060812;
  --bg-1: #0a1020;
  --bg-2: #12172b;
  --line: rgba(255, 255, 255, 0.08);
  --text: #eef4ff;
  --muted: #9aa8c9;
  --mint: #8ef3d1;
  --cyan: #7ddff2;
  --violet: #8f85ff;
  --shadow: 0 28px 90px rgba(0, 0, 0, 0.42);
  --radius-xl: 24px;
  --radius-lg: 18px;
}

html, body, .gradio-container {
  background:
    radial-gradient(circle at 20% 18%, rgba(143, 133, 255, 0.22), transparent 20%),
    radial-gradient(circle at 76% 16%, rgba(125, 223, 242, 0.16), transparent 22%),
    radial-gradient(circle at 84% 72%, rgba(142, 243, 209, 0.12), transparent 18%),
    linear-gradient(135deg, var(--bg-0) 0%, var(--bg-1) 44%, var(--bg-2) 100%);
  color: var(--text);
  font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}

.gradio-container {
  max-width: 1600px !important;
  padding: 22px !important;
}

#health-shell {
  position: relative;
  overflow: hidden;
  isolation: isolate;
}

#bg-canvas {
  position: fixed;
  inset: 0;
  width: 100vw;
  height: 100vh;
  z-index: 0;
  opacity: 0.72;
  pointer-events: none;
}

#cursor-glow {
  position: fixed;
  width: 260px;
  height: 260px;
  margin-left: -130px;
  margin-top: -130px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(142, 243, 209, 0.18), rgba(125, 223, 242, 0.08) 36%, rgba(143, 133, 255, 0.02) 62%, transparent 72%);
  filter: blur(10px);
  opacity: 0;
  transform: translate3d(-999px, -999px, 0);
  transition: opacity 0.25s ease;
  pointer-events: none;
  z-index: 1;
  mix-blend-mode: screen;
}

.ambient-fog {
  position: fixed;
  inset: auto 10% 74% 10%;
  height: 220px;
  border-radius: 999px;
  background:
    radial-gradient(circle at 30% 40%, rgba(143,133,255,0.24), transparent 20%),
    radial-gradient(circle at 58% 54%, rgba(125,223,242,0.16), transparent 18%),
    radial-gradient(circle at 70% 44%, rgba(142,243,209,0.18), transparent 22%);
  filter: blur(26px);
  opacity: 0.78;
  pointer-events: none;
  z-index: 0;
  animation: fogShift 18s ease-in-out infinite;
}

.dashboard-frame {
  position: relative;
  z-index: 2;
  display: grid;
  grid-template-columns: 88px minmax(0, 1.45fr) minmax(320px, 0.72fr);
  gap: 18px;
  min-height: calc(100vh - 44px);
}

.glass-shell,
.glass-card,
.glass-chat,
.glass-sidebar {
  border: 1px solid var(--line) !important;
  background: linear-gradient(180deg, rgba(21, 27, 49, 0.82), rgba(12, 16, 29, 0.70)) !important;
  backdrop-filter: blur(24px) saturate(120%);
  -webkit-backdrop-filter: blur(24px) saturate(120%);
  box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,0.05);
}

.glass-sidebar {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  border-radius: 28px;
  padding: 18px 14px;
}

.brand-mark {
  width: 56px;
  height: 56px;
  margin: 0 auto 16px;
  border-radius: 18px;
  background: radial-gradient(circle at 28% 28%, rgba(255,255,255,0.26), transparent 30%), linear-gradient(135deg, rgba(142,243,209,0.24), rgba(143,133,255,0.32));
  border: 1px solid rgba(255,255,255,0.12);
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 22px;
  box-shadow: 0 18px 40px rgba(106, 92, 255, 0.18);
}

.nav-stack, .sidebar-foot {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.nav-item {
  position: relative;
  width: 100%;
  padding: 12px 0;
  border-radius: 16px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.04);
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--muted);
  transition: transform 0.22s ease, background 0.22s ease, color 0.22s ease, box-shadow 0.22s ease;
  overflow: hidden;
}

.nav-item.active {
  color: white;
  background: linear-gradient(180deg, rgba(143,133,255,0.34), rgba(125,223,242,0.16));
  box-shadow: inset 3px 0 0 var(--mint), 0 10px 26px rgba(109, 110, 255, 0.16);
}

.nav-item:hover {
  transform: translateY(-1px) scale(1.02);
  color: white;
  background: rgba(255,255,255,0.06);
}

.nav-item::after,
.shortcut-btn::after,
.gr-button::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(110deg, transparent 20%, rgba(255,255,255,0.18) 50%, transparent 80%);
  transform: translateX(-130%);
  transition: transform 0.55s ease;
}

.nav-item:hover::after,
.shortcut-btn:hover::after,
.gr-button:hover::after {
  transform: translateX(130%);
}

.main-column {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto;
  gap: 18px;
  min-width: 0;
}

.hero-banner {
  position: relative;
  overflow: hidden;
  border-radius: 28px;
  padding: 26px 28px;
}

.hero-banner::before {
  content: "";
  position: absolute;
  inset: -20% auto auto -8%;
  width: 60%;
  height: 160%;
  background: radial-gradient(circle, rgba(143,133,255,0.24), transparent 56%);
  filter: blur(10px);
}

.hero-banner::after {
  content: "";
  position: absolute;
  inset: auto -18% -30% auto;
  width: 50%;
  height: 150%;
  background: radial-gradient(circle, rgba(142,243,209,0.18), transparent 56%);
  filter: blur(8px);
}

.hero-row {
  position: relative;
  z-index: 1;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
}

.hero-copy { max-width: 760px; }

.hero-kicker {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 8px 14px;
  border-radius: 999px;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.08);
  color: var(--muted);
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.hero-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--mint);
  box-shadow: 0 0 0 0 rgba(142,243,209,0.55);
  animation: pulseGlow 2.2s ease-out infinite;
}

.hero-title {
  margin: 16px 0 10px;
  font-size: 40px;
  line-height: 1.06;
  font-weight: 800;
  letter-spacing: -0.03em;
  color: #f8fbff;
}

.hero-subtitle {
  margin: 0;
  max-width: 680px;
  font-size: 15px;
  line-height: 1.78;
  color: var(--muted);
}

.hero-pills, .mode-switches, .quick-prompt-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 18px;
}

.hero-pill, .mode-chip, .quick-pill {
  padding: 10px 14px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.04);
  color: #d7def5;
  font-size: 13px;
}

.hero-mode {
  min-width: 220px;
  padding: 16px;
  border-radius: 20px;
  background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
  border: 1px solid rgba(255,255,255,0.08);
}

.mode-label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
.mode-value { margin-top: 10px; font-size: 18px; font-weight: 700; }
.mode-meta { margin-top: 10px; display: flex; justify-content: space-between; color: var(--muted); font-size: 12px; }
.mode-chip.active { background: linear-gradient(90deg, rgba(143,133,255,0.28), rgba(142,243,209,0.16)); box-shadow: inset 0 0 0 1px rgba(142,243,209,0.18); }

.chat-shell {
  border-radius: 30px;
  padding: 18px !important;
  display: grid;
  grid-template-rows: auto auto auto minmax(0, 1fr) auto auto;
  gap: 14px;
  min-height: 0;
}

.chat-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.title-stack h3 { margin: 0; font-size: 20px; font-weight: 700; color: #ffffff; }
.title-stack p { margin: 6px 0 0; color: var(--muted); font-size: 13px; }
.top-meta { display: flex; align-items: center; gap: 10px; }
.status-badge, .time-chip {
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.08);
  font-size: 12px;
  color: var(--muted);
}
.status-badge strong { color: var(--mint); font-weight: 600; }

#chatbot { min-height: 0; background: transparent !important; }
#chatbot .message { animation: riseIn 0.42s ease-out; }
#chatbot .message.user {
  background: linear-gradient(135deg, rgba(125,223,242,0.18), rgba(143,133,255,0.12)) !important;
  border: 1px solid rgba(125,223,242,0.16) !important;
  color: #f6fbff !important;
  box-shadow: 0 10px 22px rgba(77, 149, 179, 0.10);
}
#chatbot .message.bot {
  background: linear-gradient(180deg, rgba(20,27,46,0.94), rgba(14,18,33,0.92)) !important;
  border: 1px solid rgba(142,243,209,0.12) !important;
  color: #edf5ff !important;
  box-shadow: 0 12px 28px rgba(0,0,0,0.24);
}
#chatbot strong, #chatbot h1, #chatbot h2, #chatbot h3, #chatbot code { color: var(--mint) !important; }

.scroll-fade { position: relative; min-height: 0; }
.scroll-fade::before, .scroll-fade::after {
  content: "";
  position: absolute;
  left: 0; right: 0; height: 24px; z-index: 2; pointer-events: none;
}
.scroll-fade::before { top: 0; background: linear-gradient(180deg, rgba(12,16,29,0.94), rgba(12,16,29,0)); }
.scroll-fade::after { bottom: 0; background: linear-gradient(0deg, rgba(12,16,29,0.94), rgba(12,16,29,0)); }

#session_display textarea, #status_box textarea {
  background: transparent !important;
  color: #dce8ff !important;
  font-weight: 600;
}

#session_display, #status_box, #msg_box, #action_bar, #profile_box, #session_box, #image_well, #knowledge_box, #examples_box {
  border-radius: var(--radius-xl) !important;
}

#msg_box {
  background: linear-gradient(180deg, rgba(21,26,47,0.94), rgba(14,18,33,0.88)) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}

#msg_box:focus-within {
  border-color: rgba(125,223,242,0.30) !important;
  box-shadow: 0 0 0 1px rgba(125,223,242,0.18), 0 0 26px rgba(125,223,242,0.12);
}

#msg_box textarea, #status_box textarea, #session_display textarea, #health_box textarea {
  color: #edf4ff !important;
  font-size: 15px !important;
  line-height: 1.72 !important;
}

#action_bar { gap: 10px !important; }

.gr-button {
  position: relative;
  overflow: hidden;
  border-radius: 16px !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  box-shadow: 0 12px 30px rgba(0,0,0,0.16);
  transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
}

.gr-button:hover {
  transform: translateY(-1px) scale(1.02);
  box-shadow: 0 16px 36px rgba(0,0,0,0.24);
}

#send_btn {
  background: linear-gradient(135deg, rgba(125,223,242,0.94), rgba(142,243,209,0.92)) !important;
  color: #07101f !important;
  font-weight: 800;
}

#build_btn, #save_profile_btn, #new_session_btn {
  background: linear-gradient(135deg, rgba(143,133,255,0.28), rgba(125,223,242,0.16)) !important;
  color: #f6fbff !important;
}

#clear_btn, #export_btn { background: rgba(255,255,255,0.05) !important; color: #eef4ff !important; }

.gr-input, .gr-box, .gr-file, .gr-image, .gradio-dropdown { border-radius: 16px !important; }

.side-column {
  display: grid;
  grid-template-rows: auto auto auto 1fr auto auto;
  gap: 18px;
  min-width: 0;
}

.side-card {
  border-radius: 26px;
  padding: 18px !important;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.side-card:hover, #chat_shell:hover {
  transform: translateY(-2px) scale(1.01);
  box-shadow: 0 20px 36px rgba(0,0,0,0.20);
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 14px;
}

.card-title { margin: 0; font-size: 16px; font-weight: 700; color: #ffffff; }
.card-note { margin: 4px 0 0; color: var(--muted); font-size: 12px; }
.mini-tag {
  padding: 7px 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  color: var(--muted);
  font-size: 11px;
}

.data-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.stat-box {
  padding: 14px;
  border-radius: 18px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
}

.stat-box strong { display: block; font-size: 22px; margin-top: 8px; }
.stat-box span { color: var(--muted); font-size: 12px; }
.progress-stack { display: grid; gap: 12px; margin-top: 14px; }
.progress-item { display: grid; gap: 8px; }
.progress-head { display: flex; justify-content: space-between; color: #dce8ff; font-size: 13px; }
.bar { height: 8px; border-radius: 999px; background: rgba(255,255,255,0.06); overflow: hidden; }
.bar > span {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--mint), var(--cyan), var(--violet));
  background-size: 180% 100%;
  animation: flowBar 7s linear infinite;
}

.sparkline {
  margin-top: 16px;
  height: 90px;
  border-radius: 16px;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01)),
    repeating-linear-gradient(90deg, transparent 0 34px, rgba(255,255,255,0.04) 34px 35px),
    repeating-linear-gradient(180deg, transparent 0 26px, rgba(255,255,255,0.04) 26px 27px);
  position: relative;
  overflow: hidden;
}

.sparkline::after {
  content: "";
  position: absolute;
  inset: 16px 14px 18px 14px;
  border-radius: 999px;
  background:
    linear-gradient(180deg, rgba(142,243,209,0.08), transparent 70%),
    linear-gradient(90deg, transparent 0%, rgba(142,243,209,0.95) 14%, rgba(125,223,242,0.88) 48%, rgba(143,133,255,0.92) 100%);
  clip-path: polygon(0% 72%, 10% 66%, 18% 68%, 28% 50%, 38% 54%, 50% 34%, 61% 44%, 73% 28%, 83% 42%, 100% 10%, 100% 100%, 0% 100%);
  box-shadow: 0 0 24px rgba(142,243,209,0.26);
}

.todo-list { display: grid; gap: 10px; }
.todo-item {
  padding: 14px;
  border-radius: 18px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.todo-item:hover { transform: translateY(-2px); box-shadow: 0 16px 30px rgba(0,0,0,0.18); }
.todo-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.todo-title { font-size: 14px; font-weight: 600; color: #f2f8ff; }
.todo-time { color: var(--muted); font-size: 11px; }
.todo-copy { color: var(--muted); font-size: 12px; line-height: 1.7; }

.shortcut-grid { display: grid; gap: 10px; }
.shortcut-btn {
  padding: 14px 16px;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(143,133,255,0.20), rgba(125,223,242,0.10));
  border: 1px solid rgba(255,255,255,0.08);
  color: #eef5ff;
  font-size: 13px;
  font-weight: 600;
  position: relative;
  overflow: hidden;
}

.knowledge-note {
  margin-top: 12px;
  padding: 14px;
  border-radius: 16px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  color: var(--muted);
  line-height: 1.72;
  font-size: 12px;
}

@keyframes fogShift {
  0%, 100% { transform: translateX(0) scaleX(1); opacity: 0.76; }
  50% { transform: translateX(20px) scaleX(1.04); opacity: 1; }
}

@keyframes pulseGlow {
  0% { box-shadow: 0 0 0 0 rgba(142,243,209,0.56); }
  70% { box-shadow: 0 0 0 14px rgba(142,243,209,0); }
  100% { box-shadow: 0 0 0 0 rgba(142,243,209,0); }
}

@keyframes riseIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes flowBar {
  0% { background-position: 0% 50%; }
  100% { background-position: 180% 50%; }
}

@media (max-width: 1320px) {
  .dashboard-frame { grid-template-columns: 76px minmax(0, 1fr); }
  .side-column {
    grid-column: 1 / -1;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    grid-template-rows: none;
  }
}

@media (max-width: 900px) {
  .dashboard-frame { grid-template-columns: 1fr; }
  .glass-sidebar { flex-direction: row; align-items: center; gap: 12px; }
  .nav-stack, .sidebar-foot { flex-direction: row; }
  .hero-row, .chat-topbar { flex-direction: column; align-items: flex-start; }
  .side-column { grid-template-columns: 1fr; }
}
"""

ui_script = """
<canvas id=\"bg-canvas\"></canvas>
<div id=\"cursor-glow\"></div>
<script>
(function () {
  const setup = () => {
    const canvas = document.getElementById('bg-canvas');
    const glow = document.getElementById('cursor-glow');
    if (!canvas || !glow || canvas.dataset.ready === '1') return;
    canvas.dataset.ready = '1';
    const ctx = canvas.getContext('2d');
    const pointer = { x: innerWidth / 2, y: innerHeight / 2, tx: -999, ty: -999, r: 130 };
    const particles = [];
    const maxParticles = innerWidth < 900 ? 24 : 44;
    const resize = () => {
      const ratio = Math.min(devicePixelRatio || 1, 1.8);
      canvas.width = Math.floor(innerWidth * ratio);
      canvas.height = Math.floor(innerHeight * ratio);
      canvas.style.width = innerWidth + 'px';
      canvas.style.height = innerHeight + 'px';
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    };
    const make = () => ({
      x: Math.random() * innerWidth,
      y: Math.random() * innerHeight,
      vx: (Math.random() - 0.5) * 0.12,
      vy: -0.04 - Math.random() * 0.18,
      r: 0.8 + Math.random() * 2.2,
      a: 0.08 + Math.random() * 0.20,
      c: Math.random() > 0.5 ? '142,243,209' : (Math.random() > 0.5 ? '125,223,242' : '143,133,255')
    });
    for (let i = 0; i < maxParticles; i += 1) particles.push(make());
    const draw = () => {
      ctx.clearRect(0, 0, innerWidth, innerHeight);
      particles.forEach((p) => {
        p.x += p.vx;
        p.y += p.vy;
        if (p.y < -20) p.y = innerHeight + 20;
        if (p.x < -20) p.x = innerWidth + 20;
        if (p.x > innerWidth + 20) p.x = -20;
        ctx.beginPath();
        ctx.fillStyle = `rgba(${p.c}, ${p.a})`;
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
      });
      const g = ctx.createRadialGradient(pointer.x, pointer.y, 0, pointer.x, pointer.y, pointer.r);
      g.addColorStop(0, 'rgba(142,243,209,0.10)');
      g.addColorStop(0.45, 'rgba(125,223,242,0.05)');
      g.addColorStop(1, 'rgba(125,223,242,0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(pointer.x, pointer.y, pointer.r, 0, Math.PI * 2);
      ctx.fill();
      pointer.x += (pointer.tx - pointer.x) * 0.09;
      pointer.y += (pointer.ty - pointer.y) * 0.09;
      requestAnimationFrame(draw);
    };
    addEventListener('resize', resize);
    addEventListener('mousemove', (e) => {
      pointer.tx = e.clientX;
      pointer.ty = e.clientY;
      pointer.r = 120 + Math.min(Math.abs(e.movementX) + Math.abs(e.movementY), 46) * 2;
      glow.style.opacity = '1';
      glow.style.transform = `translate3d(${e.clientX}px, ${e.clientY}px, 0)`;
    }, { passive: true });
    addEventListener('mouseleave', () => { glow.style.opacity = '0'; }, { passive: true });
    resize();
    draw();
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setup, { once: true });
  } else {
    setup();
  }
})();
</script>
"""

with gr.Blocks(css=custom_css, title="灵境养生 AI 助手", elem_id="health-shell") as demo:
    gr.HTML(ui_script)
    gr.HTML('<div class="ambient-fog"></div>')

    sessions_state = gr.State({
        1: {
            "name": "对话1", "history": [], "first_question": None,
            "dialog_state": "idle",
            "collected_info": DEFAULT_COLLECTED_INFO.copy(),
            "ask_step": 0
        }
    })
    current_session_state = gr.State(1)
    user_profile_state = gr.State({"age": "", "gender": "保密", "health": ""})

    gr.HTML('<div class="dashboard-frame">')

    with gr.Column(elem_classes="glass-sidebar"):
        gr.HTML('<div><div class="brand-mark">养</div><div class="nav-stack"><div class="nav-item active">H</div><div class="nav-item">K</div><div class="nav-item">D</div><div class="nav-item">P</div><div class="nav-item">S</div></div></div><div class="sidebar-foot"><div class="nav-item">L</div><div class="nav-item">D</div></div>')

    with gr.Column(elem_classes="main-column"):
        with gr.Group(elem_classes=["glass-shell", "hero-banner"]):
            gr.HTML(
                """
                <div class="hero-row">
                  <div class="hero-copy">
                    <div class="hero-kicker"><span class="hero-dot"></span> Health Wellness Intelligence</div>
                    <h1 class="hero-title">灵境养生对话舱</h1>
                    <p class="hero-subtitle">把对话问诊、图像观察、知识检索与健康节律管理放进同一个深色养生面板里。整体以深靛蓝、薄荷绿、浅青和淡紫的柔和光感组织信息，让科技感、自然感和陪伴感同时成立。</p>
                    <div class="hero-pills">
                      <span class="hero-pill">深色科技风</span>
                      <span class="hero-pill">玻璃拟态</span>
                      <span class="hero-pill">舒缓对话模式</span>
                      <span class="hero-pill">图像识别辅助</span>
                    </div>
                  </div>
                  <div class="hero-mode">
                    <div class="mode-label">Current Mode</div>
                    <div class="mode-value">舒缓对话模式</div>
                    <div class="mode-meta"><span>在线状态：稳定</span><span>07:30 PM</span></div>
                  </div>
                </div>
                """
            )

        with gr.Group(elem_classes=["glass-chat", "chat-shell"], elem_id="chat_shell"):
            gr.HTML(
                """
                <div class="chat-topbar">
                  <div class="title-stack">
                    <h3>对话中枢</h3>
                    <p>支持养生咨询、图片观察、知识库问答与日常调理建议。</p>
                  </div>
                  <div class="top-meta">
                    <span class="status-badge"><strong>在线</strong> · 健康助理</span>
                    <span class="time-chip">今日模式：深度养生咨询</span>
                  </div>
                </div>
                <div class="mode-switches">
                  <span class="mode-chip active">舒缓对话模式</span>
                  <span class="mode-chip">深度养生咨询模式</span>
                  <span class="mode-chip">舌象观察辅助模式</span>
                </div>
                """
            )
            current_session_display = gr.Textbox(value="当前在：对话1", interactive=False, show_label=False, elem_id="session_display")
            gr.HTML('<div class="scroll-fade">')
            chatbot = gr.Chatbot(height=520, label="", elem_id="chatbot", buttons=["copy"])
            gr.HTML('</div>')
            gr.HTML('<div class="quick-prompt-row"><span class="quick-pill">今日养生建议</span><span class="quick-pill">睡眠调理方案</span><span class="quick-pill">饮食节律检查</span><span class="quick-pill">根据图片分析</span></div>')
            msg = gr.Textbox(placeholder="输入你的症状、困扰或养生目标，也可以结合上传图片进行观察...", label="输入你的问题", lines=3, max_lines=5, elem_id="msg_box")
            with gr.Row(elem_id="action_bar"):
                send_btn = gr.Button("发送", variant="primary", elem_id="send_btn")
                clear_btn = gr.Button("清空当前会话", elem_id="clear_btn")
                export_btn = gr.Button("导出对话记录", elem_id="export_btn")
            export_file = gr.File(label="下载文件", visible=False)

        with gr.Group(elem_classes=["glass-card", "side-card"], elem_id="examples_box"):
            gr.Examples(
                examples=[
                    "请给我一份适合今天状态的养生建议",
                    "最近睡眠浅、容易醒，怎么调理比较好？",
                    "请根据我上传的图片提供一些健康建议"
                ],
                inputs=msg,
                label="快捷指令"
            )

    with gr.Column(elem_classes="side-column"):
        with gr.Group(elem_classes=["glass-card", "side-card"], elem_id="health_panel"):
            gr.HTML(
                """
                <div class="card-header">
                  <div>
                    <h3 class="card-title">健康数据看板</h3>
                    <p class="card-note">心率、睡眠与饮食状态概览</p>
                  </div>
                  <span class="mini-tag">Live</span>
                </div>
                <div class="data-grid">
                  <div class="stat-box"><span>睡眠</span><strong>7.4h</strong></div>
                  <div class="stat-box"><span>心率</span><strong>72</strong></div>
                  <div class="stat-box"><span>饮水</span><strong>1.8L</strong></div>
                </div>
                <div class="progress-stack">
                  <div class="progress-item">
                    <div class="progress-head"><span>今日恢复度</span><span>82%</span></div>
                    <div class="bar"><span style="width:82%"></span></div>
                  </div>
                  <div class="progress-item">
                    <div class="progress-head"><span>睡眠节律</span><span>68%</span></div>
                    <div class="bar"><span style="width:68%"></span></div>
                  </div>
                </div>
                <div class="sparkline"></div>
                """
            )

        with gr.Group(elem_classes=["glass-card", "side-card"], elem_id="reminder_panel"):
            gr.HTML(
                """
                <div class="card-header">
                  <div>
                    <h3 class="card-title">今日养生提醒</h3>
                    <p class="card-note">像任务流一样管理你的日常节律</p>
                  </div>
                  <span class="mini-tag">3 Items</span>
                </div>
                <div class="todo-list">
                  <div class="todo-item">
                    <div class="todo-top"><span class="todo-title">午后轻运动</span><span class="todo-time">14:30</span></div>
                    <div class="todo-copy">散步 20 分钟，帮助气血流动，缓解久坐后的沉滞感。</div>
                  </div>
                  <div class="todo-item">
                    <div class="todo-top"><span class="todo-title">晚间少寒凉</span><span class="todo-time">18:00</span></div>
                    <div class="todo-copy">晚餐避免过凉饮食，优先温润、易消化的搭配。</div>
                  </div>
                  <div class="todo-item">
                    <div class="todo-top"><span class="todo-title">睡前放松</span><span class="todo-time">22:10</span></div>
                    <div class="todo-copy">减少强刺激信息输入，让睡眠节律逐步回稳。</div>
                  </div>
                </div>
                """
            )

        with gr.Group(elem_classes=["glass-card", "side-card"], elem_id="shortcut_panel"):
            gr.HTML(
                """
                <div class="card-header">
                  <div>
                    <h3 class="card-title">快捷功能区</h3>
                    <p class="card-note">一键进入高频养生场景</p>
                  </div>
                  <span class="mini-tag">Fast Action</span>
                </div>
                <div class="shortcut-grid">
                  <div class="shortcut-btn">打开养生知识库</div>
                  <div class="shortcut-btn">生成一周调理计划</div>
                  <div class="shortcut-btn">查看睡眠调理方案</div>
                </div>
                """
            )

        with gr.Group(elem_classes=["glass-card", "side-card"], elem_id="session_box"):
            gr.HTML('<div class="card-header"><div><h3 class="card-title">会话与画像</h3><p class="card-note">管理当前会话并保存个体背景</p></div><span class="mini-tag">Profile</span></div>')
            session_dropdown = gr.Dropdown(label="当前会话", choices=[("对话1", 1)], value=1, interactive=True)
            new_session_btn = gr.Button("开启新会话", elem_id="new_session_btn")
            with gr.Accordion("用户画像", open=False, elem_id="profile_box"):
                with gr.Row():
                    age_input = gr.Number(label="年龄", value=None, precision=0, minimum=0, maximum=150)
                    gender_input = gr.Radio(label="性别", choices=["男", "女", "保密"], value="保密")
                health_input = gr.Textbox(label="健康背景", placeholder="例如：最近熬夜多、容易疲惫、脾胃偏弱...", lines=3, elem_id="health_box")
                save_profile_btn = gr.Button("保存画像", elem_id="save_profile_btn")

        with gr.Group(elem_classes=["glass-card", "side-card"], elem_id="image_well"):
            gr.HTML('<div class="card-header"><div><h3 class="card-title">图像观察舱</h3><p class="card-note">上传舌象或食材图片，自动融入对话分析</p></div><span class="mini-tag">Vision</span></div>')
            image_input = gr.Image(type="filepath", label="上传图片")
            gr.HTML('<div class="knowledge-note">支持上传舌象、食材等健康相关图片。提问越具体，系统越容易结合识别结果给出贴近场景的建议。</div>')

        with gr.Group(elem_classes=["glass-card", "side-card"], elem_id="knowledge_box"):
            gr.HTML('<div class="card-header"><div><h3 class="card-title">养生知识库</h3><p class="card-note">上传资料并构建本地参考库</p></div><span class="mini-tag">RAG</span></div>')
            file_upload = gr.File(file_count="multiple", label="上传知识文件")
            build_btn = gr.Button("构建知识库", variant="primary", elem_id="build_btn")
            status_text = gr.Textbox(label="状态回响", value="等待上传资料或开始一段新的健康对话。", interactive=False, elem_id="status_box")
            gr.HTML('<div class="knowledge-note">支持将你的养生资料、笔记或健康文档整理进问答系统，让回答更贴近你的知识范围。</div>')

    gr.HTML('</div>')

    # ---------- 事件绑定 ----------
    build_btn.click(build_knowledge_base, inputs=file_upload, outputs=status_text)

    send_btn.click(
        respond,
        inputs=[msg, chatbot, sessions_state, current_session_state, user_profile_state, image_input],
        outputs=[msg, chatbot, sessions_state, session_dropdown, current_session_display]
    )
    msg.submit(
        respond,
        inputs=[msg, chatbot, sessions_state, current_session_state, user_profile_state, image_input],
        outputs=[msg, chatbot, sessions_state, session_dropdown, current_session_display]
    )

    def clear_current_session(sessions, current_id):
        sessions[current_id]["history"] = []
        sessions[current_id]["first_question"] = None
        sessions[current_id]["dialog_state"] = "idle"
        sessions[current_id]["ask_step"] = 0
        sessions[current_id]["collected_info"] = DEFAULT_COLLECTED_INFO.copy()
        sessions[current_id]["name"] = f"对话{current_id}"
        choices = [(get_session_display_name(sessions[k]), k) for k in sessions]
        return [], sessions, gr.update(choices=choices, value=current_id), f"当前在：对话{current_id}"

    clear_btn.click(
        clear_current_session,
        inputs=[sessions_state, current_session_state],
        outputs=[chatbot, sessions_state, session_dropdown, current_session_display]
    )

    export_btn.click(export_chat, inputs=chatbot, outputs=export_file).then(
        lambda f: gr.update(visible=True), None, export_file
    )

    def switch_session(selected_id, sessions):
        history = sessions[selected_id]["history"]
        display = f"当前在：{sessions[selected_id]['name']}"
        return history, selected_id, display

    session_dropdown.change(
        switch_session,
        inputs=[session_dropdown, sessions_state],
        outputs=[chatbot, current_session_state, current_session_display]
    )

    def new_session(sessions):
        sessions, new_id = create_session(sessions)
        choices = [(get_session_display_name(sessions[k]), k) for k in sessions]
        dropdown_update = gr.update(choices=choices, value=new_id)
        return sessions, new_id, dropdown_update, [], f"当前在：{sessions[new_id]['name']}"

    new_session_btn.click(
        new_session,
        inputs=[sessions_state],
        outputs=[sessions_state, current_session_state, session_dropdown, chatbot, current_session_display]
    )

    def save_profile(age, gender, health):
        return {"age": age, "gender": gender, "health": health}

    save_profile_btn.click(
        save_profile,
        inputs=[age_input, gender_input, health_input],
        outputs=user_profile_state
    )

if __name__ == "__main__":
    embeddings = DashScopeEmbeddings(model="text-embedding-v1")
    try:
        db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        logger.info("已加载现有向量知识库")
        all_data = db.get()
        bm25_docs = all_data.get('documents', [])
        if bm25_docs:
            tokenized_docs = [list(jieba.cut(doc)) for doc in bm25_docs]
            bm25_index = BM25Okapi(tokenized_docs)
            logger.info(f"BM25 索引构建完成，共 {len(bm25_docs)} 个文档")
        else:
            logger.info("知识库中暂无文档。")
    except Exception as e:
        db = None
        bm25_index = None
        bm25_docs = []
        logger.warning(f"未找到现有知识库：{e}")

    demo.launch(server_port=7861, allowed_paths=["static/images"])
