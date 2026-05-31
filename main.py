import os
import time
import threading
import socket
import warnings
import base64
import mimetypes
import asyncio
import logging
from collections import defaultdict, OrderedDict
from logging.handlers import RotatingFileHandler
import gradio as gr
from dotenv import load_dotenv
from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings("ignore", message="`langchain-community` is being sunset.*")
warnings.filterwarnings("ignore", message="The class `Chroma` was deprecated.*")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatTongyi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

# 新增导入：用于 Refine 链
from langchain_classic.chains.summarize import load_summarize_chain

from rank_bm25 import BM25Okapi
import jieba
from dashscope import TextReRank, MultiModalConversation
import dashscope

from backend.app.services.health_tool_service import health_tool_service

load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
if DASHSCOPE_API_KEY:
    os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY
else:
    print("警告：未设置 DASHSCOPE_API_KEY，调用通义千问/DashScope 时会失败。请参考 .env.example 配置。")

# 日志：仅记录必要排障/优化字段，避免保存完整隐私信息。
logger = logging.getLogger("health_rag_app")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = RotatingFileHandler("app.log", maxBytes=1 * 1024 * 1024, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# 初始化大模型（懒加载，避免未配置 API Key 时启动即崩溃）
llm = None


def has_dashscope_key():
    return bool(os.getenv("DASHSCOPE_API_KEY", "").strip())


def get_llm():
    global llm
    if not has_dashscope_key():
        return None
    if llm is None:
        llm = ChatTongyi(model="qwen-max", temperature=0.1)
    return llm

# 全局变量
db = None
bm25_index = None
bm25_docs = []

answer_cache = OrderedDict()
MAX_CACHE_SIZE = 200
rate_limit = defaultdict(list)
RATE_LIMIT = 10
RATE_WINDOW = 60
MAX_INPUT_LENGTH = 500

DISCLAIMER = "\n\n提示：以上内容仅用于健康养生科普，不能替代医生诊断；如症状持续、加重或出现急性不适，请及时就医。"

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
    "舌苔": "舌质 舌色 舌苔厚薄 齿痕 裂纹 舌象",
    "祛湿": "湿气 痰湿 脾虚 运化 饮食调理",
}

HEALTH_KEYWORDS = [
    "痛", "酸", "胀", "咳", "眠", "睡", "胃", "脾", "肝", "肾", "湿", "火", "寒", "热",
    "舌", "苔", "便秘", "腹泻", "乏力", "头晕", "月经", "养生", "调理", "饮食", "运动",
]

RED_FLAG_KEYWORDS = [
    "胸痛", "呼吸困难", "昏迷", "抽搐", "大出血", "剧烈头痛", "偏瘫", "高烧不退", "黑便", "咯血",
]

DEFAULT_COLLECTED_INFO = {
    "symptom": None,
    "duration": None,
    "accompany": None,
    "habits": None,
    "gender": None,
    "age": None,
}


def create_chroma_from_documents(documents, embeddings):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LangChainDeprecationWarning)
        return Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")


def load_chroma_store(embeddings):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LangChainDeprecationWarning)
        return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


def sanitize_for_log(text, limit=120):
    text = (text or "").replace("\n", " ").replace("\r", " ").strip()
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if api_key:
        text = text.replace(api_key, "[API_KEY]")
    return text[:limit]


def get_cache(cache_key):
    if cache_key in answer_cache:
        answer_cache.move_to_end(cache_key)
        return answer_cache[cache_key]
    return None


def set_cache(cache_key, answer):
    if len(answer_cache) >= MAX_CACHE_SIZE:
        answer_cache.popitem(last=False)
    answer_cache[cache_key] = answer


def is_rate_limited(client_id):
    now = time.time()
    requests = rate_limit[client_id]
    while requests and requests[0] < now - RATE_WINDOW:
        requests.pop(0)
    if len(requests) >= RATE_LIMIT:
        return True
    requests.append(now)
    return False


def expand_query(user_input):
    expansions = []
    for symptom, mapping in ENTITY_EXPANSION.items():
        if symptom in user_input:
            expansions.append(mapping)
    return user_input + " " + " ".join(expansions) if expansions else user_input


def is_health_query(text):
    return any(keyword in (text or "") for keyword in HEALTH_KEYWORDS)


def has_red_flag(text):
    return any(keyword in (text or "") for keyword in RED_FLAG_KEYWORDS)


def create_session(sessions=None, name_prefix="对话"):
    sessions = sessions or {}
    new_id = max(sessions.keys(), default=0) + 1
    sessions[new_id] = {
        "name": f"{name_prefix}{new_id}",
        "history": [],
        "first_question": None,
        "dialog_state": "idle",
        "collected_info": DEFAULT_COLLECTED_INFO.copy(),
        "ask_step": 0,
    }
    return sessions, new_id


def get_session_display_name(session):
    first_question = (session.get("first_question") or "").strip()
    if first_question:
        summary = first_question[:18] + "..." if len(first_question) > 18 else first_question
        return f"{session.get('name', '对话')}：{summary}"
    return session.get("name", "对话")


def session_choices(sessions):
    return [(get_session_display_name(sessions[key]), key) for key in sorted(sessions)]


def profile_to_text(user_profile):
    if not user_profile:
        return ""
    parts = []
    age = str(user_profile.get("age") or "").strip()
    gender = str(user_profile.get("gender") or "").strip()
    health = str(user_profile.get("health") or "").strip()
    if age:
        parts.append(f"年龄：{age}")
    if gender and gender != "保密":
        parts.append(f"性别：{gender}")
    if health:
        parts.append(f"健康状况/关注点：{health}")
    return "；".join(parts)


# ===================== 文件上传 + 构建知识库 =====================
def build_knowledge_base(files):
    global db, bm25_index, bm25_docs

    if not has_dashscope_key():
        return "请先在 .env 中配置 DASHSCOPE_API_KEY，再构建知识库。"

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
            return f"加载失败：{error_msg}"

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
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
        print(f"已追加 {len(texts)} 个新片段，BM25 索引已更新")
    else:
        db = create_chroma_from_documents(texts, embeddings)
        bm25_docs = [t.page_content for t in texts]
        tokenized_docs = [list(jieba.cut(doc)) for doc in bm25_docs]
        bm25_index = BM25Okapi(tokenized_docs)
        print("已创建新知识库，BM25 索引已构建")

    return "知识库更新完成，可以开始提问了。"


def hybrid_search(query, db, bm25_index, bm25_docs, top_k=10):
    # 1. 向量检索
    vector_results = db.similarity_search_with_score(query, k=top_k)

    # 2. BM25 检索
    tokenized_query = list(jieba.cut(query))
    bm25_scores = bm25_index.get_scores(tokenized_query)
    if len(bm25_scores) == 0:
        return [doc for doc, _ in vector_results[:top_k]]

    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]

    # 3. 合并
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
    if not docs:
        return []

    documents = [doc.page_content for doc in docs]
    try:
        resp = TextReRank.call(
            model='gte-rerank',
            query=query,
            documents=documents,
            top_n=top_k
        )
        if resp.status_code == 200:
            reranked_indices = [item['index'] for item in resp.output['results']]
            return [docs[i] for i in reranked_indices]
        logger.warning(f"重排序不可用，已降级为原始检索排序：{getattr(resp, 'message', 'unknown error')}")
    except Exception as exc:
        logger.warning(f"重排序异常，已降级为原始检索排序：{exc}")
    return docs[:top_k]


def rebuild_bm25_from_db():
    global bm25_index, bm25_docs

    if db is None:
        return

    try:
        all_data = db.get(include=["documents"])
        existing_docs = all_data.get("documents", [])
        if not existing_docs:
            print("知识库中暂无文档")
            return

        tokenized_docs = [list(jieba.cut(doc)) for doc in existing_docs]
        bm25_docs = existing_docs
        bm25_index = BM25Okapi(tokenized_docs)
        print(f"BM25 索引构建完成，共 {len(bm25_docs)} 个文档")
    except Exception as exc:
        bm25_docs = []
        bm25_index = None
        print(f"BM25 索引后台构建失败：{exc}")


def schedule_bm25_rebuild(delay_seconds=45):
    def worker():
        time.sleep(delay_seconds)
        rebuild_bm25_from_db()

    threading.Thread(target=worker, daemon=True).start()


def is_port_available(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def get_launch_port(default_port=7860, search_count=80):
    env_port = os.environ.get("GRADIO_SERVER_PORT")
    if env_port:
        try:
            requested_port = int(env_port)
        except ValueError:
            print(f"GRADIO_SERVER_PORT={env_port} 不是有效端口，将自动寻找可用端口")
        else:
            if is_port_available(requested_port):
                return requested_port
            print(f"端口 {requested_port} 已被占用，将自动寻找可用端口")

    for port in range(default_port, default_port + search_count):
        if is_port_available(port):
            if port != default_port:
                print(f"端口 {default_port} 已被占用，已切换到 {port}")
            return port

    raise OSError(f"未找到可用端口，请释放 {default_port}-{default_port + search_count - 1} 范围内的端口")


# ===================== 多模态：图像分析 =====================
def normalize_image_path(image):
    if not image:
        return None
    if isinstance(image, str):
        return image
    if isinstance(image, dict):
        return image.get("path") or image.get("name")
    return getattr(image, "name", None) or str(image)


def analyze_image(image) -> str:
    if not has_dashscope_key():
        return "图片分析失败：请先在 .env 中配置 DASHSCOPE_API_KEY。"

    image_path = normalize_image_path(image)
    if not image_path or not os.path.exists(image_path):
        return ""

    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

        mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        messages = [{
            "role": "user",
            "content": [
                {"image": f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"},
                {"text": "请用中文简要描述这张图片的内容，重点关注与健康、养生相关的特征。例如舌苔颜色、厚薄、裂纹、齿痕，或食材名称、新鲜度、烹饪状态等。不要做医学诊断。"}
            ]
        }]
        response = MultiModalConversation.call(model="qwen-vl-plus", messages=messages)
        if response.status_code == 200:
            return response.output.choices[0].message.content[0]["text"]
        return f"图片分析失败：多模态API返回状态码 {response.status_code}，消息：{response.message}"
    except Exception as exc:
        return f"图片分析异常：{exc}"


def is_image_analysis_error(result: str) -> bool:
    if not result:
        return True
    return result.startswith(("图片分析失败：", "图片分析异常：", "图片处理异常："))


def build_image_augmented_query(message: str, image_desc: str) -> str:
    base_message = (message or "请根据我上传的图片提供健康养生相关分析和建议。").strip()
    image_context = f"{base_message}\n{image_desc}"

    if any(keyword in image_context for keyword in ("舌", "舌象", "舌苔", "苔", "齿痕", "裂纹")):
        return (
            f"{base_message}\n\n"
            f"用户上传了一张舌象相关图片，图像识别结果如下：{image_desc}。"
            "请根据这张图片反映出的舌象特征，结合健康养生知识进行科普分析，并给出日常调理建议。"
        )

    return (
        f"{base_message}\n\n"
        f"用户上传了一张健康养生相关图片，图像识别结果如下：{image_desc}。"
        "请结合图片内容进行分析，并给出可执行的养生建议。"
    )


# ===================== 核心聊天函数（追问澄清）=====================
def current_session_label(sessions, current_id):
    session = sessions.get(current_id, {})
    return f"当前会话：{get_session_display_name(session)}"


async def _stream_answer(chat_history, answer, sessions, current_id):
    chat_history[-1]["content"] = ""
    chunk_size = 3 if len(answer) < 900 else 8
    delay = 0.012 if len(answer) < 900 else 0.004

    for index in range(0, len(answer), chunk_size):
        chat_history[-1]["content"] = answer[:index + chunk_size]
        sessions[current_id]["history"] = chat_history
        yield (
            "",
            chat_history,
            sessions,
            gr.update(choices=session_choices(sessions), value=current_id),
            current_session_label(sessions, current_id),
        )
        await asyncio.sleep(delay)


def retrieve_docs(query):
    if db is None:
        return []
    if bm25_index is None or not bm25_docs:
        candidates = db.similarity_search(query, k=10)
    else:
        candidates = hybrid_search(query, db, bm25_index, bm25_docs, top_k=10)
    return rerank(query, candidates, top_k=5)


async def call_llm(prompt):
    model = get_llm()
    if model is None:
        return "请先在 .env 中配置 DASHSCOPE_API_KEY，再使用智能问答功能。"
    response = await asyncio.to_thread(lambda: model.invoke([HumanMessage(content=prompt)]))
    return response.content


async def respond(
    message,
    chat_history,
    sessions,
    current_id,
    user_profile,
    image=None,
    request: gr.Request = None,
):
    global db, bm25_index, bm25_docs

    sessions = sessions or {}
    if not sessions:
        sessions, current_id = create_session({})
    current_id = int(current_id or min(sessions.keys()))
    if current_id not in sessions:
        sessions, current_id = create_session(sessions)

    session = sessions[current_id]
    chat_history = list(session.get("history") or chat_history or [])
    message = (message or "").strip()
    image_path = normalize_image_path(image)
    client_id = request.client.host if request and request.client else "local"

    if not message and not image_path:
        yield "", chat_history, sessions, gr.update(), current_session_label(sessions, current_id)
        return

    if not message and image_path:
        message = "请根据我上传的图片提供健康养生相关分析和建议。"

    if len(message) > MAX_INPUT_LENGTH:
        display_message = message[:MAX_INPUT_LENGTH] + "..."
        answer = f"问题过长，请控制在 {MAX_INPUT_LENGTH} 字以内。"
        chat_history.append({"role": "user", "content": display_message})
        chat_history.append({"role": "assistant", "content": answer})
        session["history"] = chat_history
        logger.warning(f"输入过长 | IP:{client_id} | 会话:{current_id} | 长度:{len(message)}")
        yield "", chat_history, sessions, gr.update(choices=session_choices(sessions), value=current_id), current_session_label(sessions, current_id)
        return

    if is_rate_limited(client_id):
        answer = "请求过于频繁，请稍后再试。"
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": answer})
        session["history"] = chat_history
        logger.warning(f"频率限制触发 | IP:{client_id} | 会话:{current_id}")
        yield "", chat_history, sessions, gr.update(choices=session_choices(sessions), value=current_id), current_session_label(sessions, current_id)
        return

    if not session.get("first_question"):
        session["first_question"] = message
        session["name"] = f"对话{current_id}"

    profile_text = profile_to_text(user_profile)
    display_message = message if not image_path else f"{message}\n\n[已上传图片，正在进行图像观察]"
    prior_history = chat_history[-4:]
    initial_tool_results = health_tool_service.run_tools(message, user_profile)

    if health_tool_service.is_tool_capability_question(message):
        answer = health_tool_service.capability_response()
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": answer})
        session["history"] = chat_history
        choices = [(get_session_display_name(sessions[k]), k) for k in sessions]
        yield "", chat_history, sessions, gr.update(choices=choices, value=current_id), current_session_label(sessions, current_id)
        return

    if not image_path and not initial_tool_results:
        cache_key = f"text::{message}::profile::{profile_text}"
        cached = get_cache(cache_key)
        if cached:
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": cached})
            session["history"] = chat_history
            logger.info(f"缓存命中 | IP:{client_id} | 会话:{current_id} | 问题:{sanitize_for_log(message)}")
            yield "", chat_history, sessions, gr.update(choices=session_choices(sessions), value=current_id), current_session_label(sessions, current_id)
            return

    if db is None and not image_path and not initial_tool_results:
        answer = "请先上传并构建知识库。"
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": answer})
        session["history"] = chat_history
        logger.info(f"知识库未就绪 | IP:{client_id} | 会话:{current_id} | 问题:{sanitize_for_log(message)}")
        yield "", chat_history, sessions, gr.update(choices=session_choices(sessions), value=current_id), current_session_label(sessions, current_id)
        return

    chat_history.append({"role": "user", "content": display_message})
    status = "正在分析图片并检索健康资料..." if image_path else "正在检索健康资料..."
    if db is None and image_path:
        status = "正在分析图片..."
    elif db is None and initial_tool_results:
        status = "正在调用健康计算工具..."
    chat_history.append({"role": "assistant", "content": status})
    session["history"] = chat_history
    yield "", chat_history, sessions, gr.update(choices=session_choices(sessions), value=current_id), current_session_label(sessions, current_id)

    try:
        image_desc = ""
        image_hint = "本轮未上传图片。"
        final_query = message

        if image_path:
            logger.info(f"开始图片分析 | IP:{client_id} | 会话:{current_id}")
            image_desc = await asyncio.to_thread(analyze_image, image_path)
            if image_desc and not is_image_analysis_error(image_desc):
                image_hint = (
                    "用户上传了图片。你看到的是图像识别后的内容，回答时使用“根据你上传的图片”或“从图片来看”。"
                    "图片识别结果仅作健康养生科普参考，不可替代医生诊断。"
                )
                final_query = build_image_augmented_query(message, image_desc)
            else:
                final_query = f"{message}\n\n图片识别未成功：{image_desc or '未获得有效图片描述'}"
                logger.warning(f"图片分析未成功 | IP:{client_id} | 会话:{current_id} | {sanitize_for_log(image_desc)}")

        expanded_query = expand_query(final_query)
        tool_results = health_tool_service.run_tools(final_query, user_profile)
        tool_context = health_tool_service.format_for_prompt(tool_results)
        tool_catalog = health_tool_service.format_catalog_for_prompt()
        if tool_results and not image_path and not is_health_query(final_query):
            answer = health_tool_service.direct_answer(tool_results)
            log_docs = "工具直接回答"
            logger.info(
                f"回答完成 | IP:{client_id} | 会话:{current_id} | 查询:{sanitize_for_log(final_query)} | "
                f"检索片段:{log_docs} | 回答:{sanitize_for_log(answer)}"
            )
            async for payload in _stream_answer(chat_history, answer, sessions, current_id):
                yield payload
            return

        if tool_results and db is None:
            top_docs = []
        elif tool_results and not is_health_query(final_query) and not image_path:
            top_docs = []
        else:
            top_docs = await asyncio.to_thread(retrieve_docs, expanded_query)
        context = "\n\n---\n\n".join([doc.page_content for doc in top_docs]) if top_docs else "暂无相关知识库资料。"

        history_text = ""
        for turn in prior_history:
            speaker = "用户" if turn["role"] == "user" else "助手"
            history_text += f"{speaker}：{turn['content']}\n"

        red_flag_rule = (
            "用户描述包含可能需要及时就医的危险信号。回答开头先提示尽快就医或寻求专业医生帮助，再给一般性养生注意事项。"
            if has_red_flag(final_query)
            else "如果用户描述出现胸痛、呼吸困难、昏迷、抽搐、大出血、偏瘫、高烧不退等危险信号，应优先建议就医。"
        )

        full_prompt = f"""你是一个专业、谨慎的健康养生科普助手。请根据【参考资料】和【用户背景】回答问题。

规则：
1. 回答必须以健康科普和日常养生建议为边界，不做确定性医学诊断，不承诺疗效。
2. 如果问题过于模糊，先礼貌追问具体方向，如饮食、运动、睡眠、穴位、症状持续时间、伴随症状等。
3. 优先结合用户画像，年龄、性别、健康状况会影响措辞和建议强度。
4. 严格基于参考资料综合回答；资料没有覆盖时说明“根据现有资料无法回答”，但可建议补充资料或咨询专业人士。
5. 如果用户上传了图片，请结合图片识别结果和参考资料；不要说成用户自己文字描述。
6. 你具备【可用健康工具】中列出的内置工具调用能力；如果用户询问你能否调用工具，请如实说明可调用这些内置健康工具，但未接入联网搜索、天气、地图、日历等外部平台工具。
7. 如果触发了健康计算工具，请优先引用【工具调用结果】中的确定性计算结果，但不要把它当作医学诊断。
8. {red_flag_rule}
9. 建议按“观察/可能相关因素/日常调理/需要就医的情况”组织，保持简洁、可执行。

【用户背景】
{profile_text or "用户未填写画像。"}

【历史对话】
{history_text or "暂无历史对话。"}

【图片处理提示】
{image_hint}

【可用健康工具】
{tool_catalog}

【工具调用结果】
{tool_context}

【参考资料】
{context}

【用户当前问题】
{final_query}

助手："""

        answer = await call_llm(full_prompt)
        if (is_health_query(final_query) or image_path) and "不能替代医生诊断" not in answer:
            answer += DISCLAIMER

        if not image_path:
            set_cache(f"text::{message}::profile::{profile_text}", answer)

        log_docs = " | ".join([sanitize_for_log(doc.page_content, 50) for doc in top_docs[:3]]) if top_docs else "无"
        logger.info(
            f"回答完成 | IP:{client_id} | 会话:{current_id} | 查询:{sanitize_for_log(final_query)} | "
            f"检索片段:{log_docs} | 回答:{sanitize_for_log(answer)}"
        )
    except Exception as exc:
        logger.exception(f"回答失败 | IP:{client_id} | 会话:{current_id} | 问题:{sanitize_for_log(message)}")
        answer = f"暂时无法完成回复：{exc}"

    async for payload in _stream_answer(chat_history, answer, sessions, current_id):
        yield payload


# ===================== Gradio 界面 =====================
APP_HEAD = """
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<style>
  html { background: #030712; }
</style>
"""

APP_CSS = r"""
:root {
  --bg-0: #02050c;
  --bg-1: #061124;
  --glass: rgba(13, 27, 51, 0.56);
  --glass-strong: rgba(17, 38, 71, 0.72);
  --line: rgba(166, 220, 255, 0.32);
  --line-hot: rgba(185, 226, 255, 0.64);
  --text: rgba(249, 253, 255, 0.98);
  --text-soft: rgba(231, 243, 255, 0.88);
  --muted: rgba(218, 233, 247, 0.78);
  --muted-dim: rgba(188, 207, 226, 0.62);
  --field: rgba(4, 14, 30, 0.42);
  --field-soft: rgba(10, 25, 48, 0.34);
  --field-line: rgba(181, 226, 255, 0.18);
  --cyan: #82dfff;
  --ice: #b9e7ff;
  --violet: #bda8ff;
  --rose: #f1a6df;
  --mint: #8deac8;
  --shadow: 0 24px 80px rgba(0, 0, 0, 0.42);
  --glow: 0 0 28px rgba(102, 204, 255, 0.25);
  color-scheme: dark;
}

* {
  box-sizing: border-box;
  scrollbar-width: thin;
  scrollbar-color: rgba(126, 215, 255, 0.78) rgba(255, 255, 255, 0.04);
}

*::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

*::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.035);
  border-radius: 999px;
}

*::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, rgba(122, 217, 255, 0.9), rgba(179, 160, 255, 0.75));
  border-radius: 999px;
  box-shadow: 0 0 14px rgba(117, 215, 255, 0.36);
}

*::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(180deg, rgba(182, 235, 255, 0.95), rgba(238, 166, 224, 0.85));
}

body,
.gradio-container {
  min-height: 100vh !important;
  color: var(--text) !important;
  font-family: "Inter", "Microsoft YaHei", "PingFang SC", "Segoe UI", Arial, sans-serif !important;
  background:
    radial-gradient(circle at 12% 9%, rgba(70, 151, 255, 0.18), transparent 29%),
    radial-gradient(circle at 85% 18%, rgba(183, 151, 255, 0.14), transparent 31%),
    radial-gradient(circle at 62% 86%, rgba(78, 223, 194, 0.10), transparent 27%),
    linear-gradient(145deg, #02050c 0%, #061124 48%, #020710 100%) !important;
  overflow-x: hidden;
}

body::before,
body::after {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
}

body::before {
  opacity: 0.42;
  background-image:
    radial-gradient(circle, rgba(255, 255, 255, 0.75) 0 1px, transparent 1.4px),
    radial-gradient(circle, rgba(133, 216, 255, 0.62) 0 1px, transparent 1.6px);
  background-size: 128px 128px, 211px 211px;
  background-position: 24px 42px, 81px 16px;
  animation: star-drift 28s linear infinite;
}

body::after {
  opacity: 0.20;
  background:
    linear-gradient(115deg, transparent 0 18%, rgba(141, 234, 200, 0.18) 18.3%, transparent 18.8% 100%),
    linear-gradient(70deg, transparent 0 63%, rgba(130, 223, 255, 0.13) 63.2%, transparent 63.7% 100%),
    radial-gradient(ellipse at 83% 74%, rgba(141, 234, 200, 0.16), transparent 24%),
    radial-gradient(ellipse at 12% 79%, rgba(241, 166, 223, 0.10), transparent 22%);
  filter: blur(0.2px);
}

#cosmic-particles,
#cursor-aura,
#cursor-trail {
  position: fixed;
  pointer-events: none;
  z-index: 1;
}

#cosmic-particles {
  inset: 0;
  opacity: 0.78;
  mix-blend-mode: screen;
}

#cursor-aura {
  width: 220px;
  height: 220px;
  border-radius: 999px;
  margin: -110px 0 0 -110px;
  background:
    radial-gradient(circle, rgba(185, 231, 255, 0.34) 0%, rgba(189, 168, 255, 0.20) 33%, rgba(241, 166, 223, 0.10) 50%, transparent 72%);
  filter: blur(12px);
  opacity: 0;
  transform: translate3d(-300px, -300px, 0) scale(0.78);
  transition: opacity 0.24s ease, transform 0.16s ease;
  mix-blend-mode: screen;
}

#cursor-trail {
  width: 86px;
  height: 86px;
  border-radius: 999px;
  margin: -43px 0 0 -43px;
  background: radial-gradient(circle, rgba(130, 223, 255, 0.24), transparent 68%);
  filter: blur(8px);
  opacity: 0;
  transform: translate3d(-300px, -300px, 0);
  transition: opacity 0.2s ease;
}

body.cursor-active #cursor-aura,
body.cursor-active #cursor-trail {
  opacity: 1;
}

body.ui-hovering #cursor-aura {
  transform: translate3d(var(--cursor-x, -300px), var(--cursor-y, -300px), 0) scale(1.18);
  filter: blur(10px) saturate(1.2);
}

.gradio-container {
  max-width: none !important;
  padding: 0 !important;
  position: relative;
  z-index: 2;
}

.gradio-container label,
.gradio-container textarea,
.gradio-container input,
.gradio-container .prose,
.gradio-container .markdown,
.gradio-container .wrap {
  color: var(--text-soft) !important;
}

.gradio-container textarea:disabled,
.gradio-container input:disabled {
  opacity: 1 !important;
  -webkit-text-fill-color: var(--text-soft) !important;
}

footer {
  display: none !important;
}

#app-shell {
  width: min(1500px, calc(100vw - 34px));
  min-height: calc(100vh - 34px);
  margin: 17px auto !important;
  gap: 18px !important;
  align-items: stretch !important;
  position: relative;
  z-index: 2;
}

#left-rail,
#chat-panel,
#right-panel {
  border: 1px solid var(--line) !important;
  background:
    linear-gradient(145deg, rgba(255, 255, 255, 0.105), rgba(88, 159, 255, 0.045) 42%, rgba(188, 161, 255, 0.06)),
    rgba(9, 20, 39, 0.48) !important;
  box-shadow: var(--shadow), inset 0 1px 0 rgba(255, 255, 255, 0.16), var(--glow);
  backdrop-filter: blur(24px) saturate(1.2);
  -webkit-backdrop-filter: blur(24px) saturate(1.2);
  border-radius: 26px !important;
  position: relative;
  overflow: hidden;
  animation: panel-rise 0.82s cubic-bezier(.18, .84, .28, 1) both;
}

#left-rail::before,
#chat-panel::before,
#right-panel::before {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: inherit;
  pointer-events: none;
  background:
    linear-gradient(115deg, rgba(255, 255, 255, 0.18), transparent 23%),
    radial-gradient(circle at var(--mouse-x, 50%) var(--mouse-y, 28%), rgba(130, 223, 255, 0.12), transparent 32%);
}

#left-rail {
  padding: 18px !important;
  min-width: 226px !important;
  max-width: 252px !important;
  height: calc(100vh - 34px);
  position: sticky !important;
  top: 17px;
  animation-delay: 0.04s;
}

#chat-panel {
  min-width: 420px !important;
  padding: 18px !important;
  animation-delay: 0.16s;
}

#right-panel {
  padding: 18px !important;
  min-width: 278px !important;
  max-width: 330px !important;
  height: calc(100vh - 34px);
  position: sticky !important;
  top: 17px;
  animation-delay: 0.28s;
}

.brand-lockup,
.chat-topline,
.panel-title {
  position: relative;
  z-index: 1;
}

.brand-lockup {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 22px;
}

.brand-mark {
  display: grid;
  place-items: center;
  width: 42px;
  height: 42px;
  border-radius: 16px;
  color: #061124;
  font-weight: 900;
  letter-spacing: 0;
  background: linear-gradient(135deg, var(--ice), var(--violet) 58%, var(--rose));
  box-shadow: 0 0 30px rgba(130, 223, 255, 0.35);
}

.brand-lockup strong,
.chat-title h1,
.panel-title h2 {
  color: var(--text);
  text-shadow: 0 0 18px rgba(154, 218, 255, 0.24);
  letter-spacing: 0;
}

.brand-lockup strong {
  display: block;
  font-size: 15px;
}

.brand-lockup small,
.eyebrow,
.metric-card small {
  color: var(--muted);
  text-shadow: 0 1px 8px rgba(0, 0, 0, 0.45);
}

.side-nav {
  display: grid;
  gap: 8px;
  margin-bottom: 18px;
  position: relative;
  z-index: 1;
}

.nav-link {
  display: flex;
  align-items: center;
  gap: 11px;
  min-height: 44px;
  padding: 0 12px;
  border: 1px solid rgba(150, 211, 255, 0.12);
  border-radius: 16px;
  color: rgba(236, 247, 255, 0.82);
  text-decoration: none;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.06), rgba(126, 203, 255, 0.025));
  transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease, background 0.22s ease;
}

.nav-link {
  cursor: pointer;
}

.nav-link:hover,
.nav-link.active {
  transform: translateY(-2px);
  border-color: rgba(167, 225, 255, 0.52);
  box-shadow: 0 14px 32px rgba(47, 174, 255, 0.17), inset 0 1px 0 rgba(255, 255, 255, 0.18);
  background:
    linear-gradient(90deg, rgba(117, 220, 255, 0.24), rgba(191, 168, 255, 0.15) 58%, rgba(241, 166, 223, 0.10)),
    rgba(255, 255, 255, 0.045);
}

.nav-link.active {
  position: relative;
}

.nav-link.active::before {
  content: "";
  position: absolute;
  left: -1px;
  top: 10px;
  bottom: 10px;
  width: 3px;
  border-radius: 999px;
  background: linear-gradient(180deg, var(--ice), var(--violet), var(--rose));
  box-shadow: 0 0 18px rgba(130, 223, 255, 0.65);
}

.nav-glyph {
  display: grid;
  place-items: center;
  width: 22px;
  height: 22px;
  border-radius: 8px;
  color: var(--ice);
  background: rgba(141, 217, 255, 0.10);
}

#knowledge-card,
#status-card {
  position: relative;
  z-index: 1;
  padding: 14px;
  border: 1px solid rgba(183, 226, 255, 0.11);
  border-radius: 20px;
  background:
    radial-gradient(circle at 12% 0%, rgba(133, 216, 255, 0.09), transparent 38%),
    linear-gradient(150deg, rgba(255, 255, 255, 0.045), rgba(122, 205, 255, 0.026) 52%, rgba(188, 161, 255, 0.035)),
    rgba(4, 13, 27, 0.22);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.10);
  overflow: hidden;
  transition: border-color 0.2s ease, background 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}

#knowledge-card {
  margin-top: auto;
}

#status-card {
  margin-top: 12px;
}

#knowledge-card:hover,
#status-card:hover {
  transform: translateY(-1px);
  border-color: rgba(190, 232, 255, 0.22);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.13), 0 12px 30px rgba(46, 164, 235, 0.08);
}

#knowledge-card label,
#status-card label,
#chat-input label {
  color: rgba(226, 242, 255, 0.86) !important;
  font-size: 12px !important;
  font-weight: 650 !important;
  letter-spacing: 0.03em;
  text-shadow: 0 1px 10px rgba(0, 0, 0, 0.42);
}

#file-upload,
#status-box,
#chat-input,
#chatbot {
  position: relative;
  z-index: 1;
}

#file-upload {
  overflow: hidden;
  border-radius: 18px;
}

#file-upload *,
#status-box *,
#chat-input * {
  color: var(--text-soft) !important;
}

#status-box .wrap,
#status-box .container,
#chat-input .wrap,
#chat-input .container {
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

#file-upload .wrap,
#file-upload .container,
#file-upload [class*="upload"],
#file-upload [class*="drop"],
#file-upload [data-testid*="file"],
#status-box textarea,
#status-box input,
#chat-input textarea,
textarea#chat-input {
  color: var(--text) !important;
  border: 1px solid var(--field-line) !important;
  background:
    linear-gradient(145deg, rgba(255, 255, 255, 0.055), rgba(117, 204, 255, 0.030)),
    var(--field) !important;
  border-radius: 18px !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.09), inset 0 -14px 28px rgba(0, 0, 0, 0.10);
  backdrop-filter: blur(18px) saturate(1.1);
  -webkit-backdrop-filter: blur(18px) saturate(1.1);
  transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease, transform 0.2s ease;
}

#file-upload [class*="upload"],
#file-upload [class*="drop"] {
  min-height: 94px;
  border-style: dashed !important;
  border-color: rgba(181, 226, 255, 0.22) !important;
}

#file-upload svg {
  color: rgba(184, 231, 255, 0.80) !important;
  filter: drop-shadow(0 0 10px rgba(130, 223, 255, 0.20));
}

#file-upload p,
#file-upload span,
#file-upload a {
  color: rgba(231, 243, 255, 0.86) !important;
  text-shadow: 0 1px 8px rgba(0, 0, 0, 0.36);
}

#file-upload:hover .wrap,
#file-upload:hover .container,
#file-upload:hover [class*="upload"],
#file-upload:hover [class*="drop"],
#status-box:hover textarea,
#chat-input:hover textarea,
#chat-input textarea:focus,
textarea#chat-input:hover,
textarea#chat-input:focus {
  border-color: rgba(190, 232, 255, 0.36) !important;
  background:
    linear-gradient(145deg, rgba(255, 255, 255, 0.075), rgba(126, 213, 255, 0.050)),
    rgba(7, 19, 38, 0.50) !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.13), 0 0 22px rgba(93, 193, 255, 0.10);
}

#status-box textarea {
  min-height: 76px !important;
  resize: none !important;
  font-size: 13.5px !important;
  line-height: 1.55 !important;
  cursor: default !important;
  -webkit-text-fill-color: rgba(235, 247, 255, 0.88) !important;
}

#chat-input textarea,
textarea#chat-input {
  min-height: 62px !important;
  max-height: 138px !important;
  padding: 16px 17px !important;
  font-size: 15px !important;
  line-height: 1.55 !important;
  outline: none !important;
}

#chat-input textarea::placeholder,
textarea#chat-input::placeholder {
  color: rgba(214, 231, 247, 0.64) !important;
}

button,
.gradio-button {
  border-radius: 16px !important;
  border: 1px solid rgba(158, 222, 255, 0.28) !important;
  color: var(--text) !important;
  background:
    linear-gradient(135deg, rgba(117, 214, 255, 0.22), rgba(188, 163, 255, 0.16) 62%, rgba(241, 166, 223, 0.10)),
    rgba(255, 255, 255, 0.045) !important;
  box-shadow: 0 12px 30px rgba(30, 160, 235, 0.14), inset 0 1px 0 rgba(255, 255, 255, 0.16) !important;
  text-shadow: 0 1px 10px rgba(0, 0, 0, 0.45);
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease, filter 0.18s ease !important;
}

button:hover,
.gradio-button:hover,
.metric-card:hover {
  transform: translateY(-2px) !important;
  border-color: var(--line-hot) !important;
  box-shadow: 0 20px 44px rgba(70, 181, 255, 0.20), 0 0 34px rgba(181, 160, 255, 0.13), inset 0 1px 0 rgba(255, 255, 255, 0.22) !important;
  filter: saturate(1.08);
}

button:active,
.gradio-button:active {
  transform: translateY(0) scale(0.985) !important;
}

#build-btn,
#build-btn button {
  background:
    linear-gradient(135deg, rgba(130, 223, 255, 0.26), rgba(189, 168, 255, 0.18) 62%, rgba(241, 166, 223, 0.10)),
    rgba(255, 255, 255, 0.045) !important;
  color: rgba(243, 251, 255, 0.96) !important;
  font-weight: 760 !important;
  text-shadow: 0 1px 10px rgba(0, 0, 0, 0.35) !important;
}

#send-btn,
#send-btn button {
  background:
    linear-gradient(135deg, rgba(137, 229, 255, 0.88), rgba(166, 169, 255, 0.72) 58%, rgba(241, 166, 223, 0.54)) !important;
  color: #03101f !important;
  font-weight: 800 !important;
  text-shadow: none !important;
}

#clear-btn,
#clear-btn button,
#file-upload button {
  min-height: 38px !important;
  background:
    linear-gradient(135deg, rgba(255, 255, 255, 0.075), rgba(130, 223, 255, 0.070)),
    rgba(5, 16, 32, 0.28) !important;
  color: rgba(232, 244, 255, 0.92) !important;
}

.chat-topline {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  padding: 2px 4px 16px;
}

.chat-title h1 {
  margin: 0;
  font-size: clamp(25px, 3vw, 40px);
  line-height: 1.08;
}

.chat-title p {
  margin: 9px 0 0;
  color: var(--muted);
  max-width: 660px;
  line-height: 1.7;
  text-shadow: 0 1px 10px rgba(0, 0, 0, 0.42);
}

.signal-pill {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  gap: 9px;
  padding: 10px 13px;
  border-radius: 999px;
  border: 1px solid rgba(146, 224, 255, 0.24);
  color: var(--ice);
  background: linear-gradient(135deg, rgba(130, 223, 255, 0.12), rgba(189, 168, 255, 0.08));
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.14);
}

.signal-dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: var(--mint);
  box-shadow: 0 0 14px var(--mint);
  animation: pulse-dot 1.8s ease-in-out infinite;
}

#chatbot {
  flex: 1 1 auto;
  min-height: 420px !important;
  border: 1px solid rgba(154, 219, 255, 0.18) !important;
  border-radius: 24px !important;
  overflow: hidden !important;
  background:
    radial-gradient(circle at 20% 15%, rgba(130, 223, 255, 0.08), transparent 34%),
    linear-gradient(145deg, rgba(255, 255, 255, 0.055), rgba(139, 165, 255, 0.035)),
    rgba(3, 10, 21, 0.38) !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.11);
}

#chatbot .message,
#chatbot [class*="message"] {
  animation: message-in 0.34s cubic-bezier(.2, .84, .3, 1) both;
}

#chatbot .bubble,
#chatbot [class*="bubble"] {
  border: 1px solid rgba(176, 229, 255, 0.20) !important;
  color: rgba(248, 252, 255, 0.96) !important;
  border-radius: 18px !important;
  background:
    linear-gradient(135deg, rgba(184, 232, 255, 0.15), rgba(190, 166, 255, 0.10)),
    rgba(10, 24, 45, 0.62) !important;
  box-shadow: 0 14px 34px rgba(0, 0, 0, 0.24), inset 0 1px 0 rgba(255, 255, 255, 0.16) !important;
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}

#chatbot .bubble:hover,
#chatbot [class*="bubble"]:hover {
  transform: translateY(-2px);
  border-color: rgba(188, 232, 255, 0.50) !important;
  box-shadow: 0 18px 42px rgba(73, 182, 255, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.20) !important;
}

#chatbot .user .bubble,
#chatbot [data-testid*="user"] .bubble,
#chatbot [class*="user"] [class*="bubble"] {
  background:
    linear-gradient(135deg, rgba(130, 223, 255, 0.30), rgba(133, 174, 255, 0.18) 62%, rgba(255, 255, 255, 0.08)),
    rgba(10, 31, 56, 0.68) !important;
  border-color: rgba(155, 227, 255, 0.34) !important;
}

#chatbot .bot .bubble,
#chatbot [data-testid*="bot"] .bubble,
#chatbot [class*="bot"] [class*="bubble"],
#chatbot [class*="assistant"] [class*="bubble"] {
  background:
    linear-gradient(135deg, rgba(189, 168, 255, 0.22), rgba(130, 223, 255, 0.16) 62%, rgba(141, 234, 200, 0.08)),
    rgba(12, 22, 45, 0.66) !important;
}

#chatbot .bot .bubble::after,
#chatbot [class*="bot"] [class*="bubble"]::after,
#chatbot [class*="assistant"] [class*="bubble"]::after {
  content: "";
  display: inline-block;
  width: 7px;
  height: 1.1em;
  margin-left: 4px;
  vertical-align: -2px;
  border-radius: 999px;
  background: linear-gradient(180deg, var(--ice), var(--violet));
  box-shadow: 0 0 12px rgba(174, 222, 255, 0.48);
  animation: cursor-blink 1.05s steps(2, start) infinite;
}

#chatbot pre,
#chatbot code {
  color: rgba(239, 249, 255, 0.96) !important;
  border-radius: 14px !important;
  border: 1px solid rgba(156, 220, 255, 0.18);
  background: rgba(2, 8, 19, 0.46) !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
}

#chatbot ul,
#chatbot ol {
  padding-left: 1.4em;
}

#input-dock {
  margin-top: 14px;
  padding: 10px;
  gap: 10px !important;
  align-items: flex-end !important;
  border: 1px solid rgba(176, 226, 255, 0.13);
  border-radius: 24px;
  background:
    radial-gradient(circle at 18% 0%, rgba(130, 223, 255, 0.08), transparent 42%),
    linear-gradient(135deg, rgba(255, 255, 255, 0.050), rgba(109, 197, 255, 0.030)),
    rgba(3, 11, 24, 0.34);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.10), 0 12px 34px rgba(0, 0, 0, 0.16);
  backdrop-filter: blur(20px) saturate(1.1);
  -webkit-backdrop-filter: blur(20px) saturate(1.1);
}

.panel-title {
  margin-bottom: 14px;
}

.eyebrow {
  display: inline-block;
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.panel-title h2 {
  margin: 7px 0 0;
  font-size: 24px;
}

.health-grid {
  display: grid;
  gap: 12px;
  position: relative;
  z-index: 1;
}

.metric-card {
  padding: 14px;
  border: 1px solid rgba(159, 223, 255, 0.18);
  border-radius: 18px;
  background:
    linear-gradient(145deg, rgba(199, 236, 255, 0.13), rgba(189, 168, 255, 0.07) 58%, rgba(141, 234, 200, 0.06)),
    rgba(8, 19, 37, 0.38);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.12), 0 12px 28px rgba(0, 0, 0, 0.18);
  transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}

.metric-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: baseline;
  color: var(--text);
  margin-bottom: 10px;
}

.metric-head strong {
  color: var(--ice);
  font-size: 20px;
  text-shadow: 0 0 18px rgba(130, 223, 255, 0.34);
}

.progress-track {
  height: 7px;
  border-radius: 999px;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.08);
  box-shadow: inset 0 1px 4px rgba(0, 0, 0, 0.24);
}

.progress-track span {
  display: block;
  height: 100%;
  width: var(--value);
  border-radius: inherit;
  background: linear-gradient(90deg, var(--mint), var(--cyan), var(--violet));
  box-shadow: 0 0 18px rgba(130, 223, 255, 0.38);
}

.pulse-chart {
  width: 100%;
  height: 96px;
  margin: 8px 0 2px;
  overflow: visible;
}

.pulse-chart polyline {
  fill: none;
  stroke: url(#pulseLine);
  stroke-width: 4;
  stroke-linecap: round;
  stroke-linejoin: round;
  filter: drop-shadow(0 0 9px rgba(126, 215, 255, 0.35));
  stroke-dasharray: 360;
  animation: draw-line 1.6s ease both;
}

#quick-actions {
  position: relative;
  z-index: 1;
  margin-top: 14px;
  gap: 9px !important;
}

.quick-button button {
  width: 100%;
  min-height: 42px !important;
  justify-content: flex-start !important;
  padding-left: 14px !important;
}

@keyframes star-drift {
  from { transform: translate3d(0, 0, 0); }
  to { transform: translate3d(-120px, 72px, 0); }
}

@keyframes panel-rise {
  from { opacity: 0; transform: translateY(22px) scale(0.985); filter: blur(6px); }
  to { opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }
}

@keyframes message-in {
  from { opacity: 0; transform: translateY(12px) translateX(-10px); }
  to { opacity: 1; transform: translateY(0) translateX(0); }
}

@keyframes cursor-blink {
  0%, 42% { opacity: 0.95; }
  43%, 100% { opacity: 0; }
}

@keyframes pulse-dot {
  0%, 100% { transform: scale(1); opacity: 0.72; }
  50% { transform: scale(1.55); opacity: 1; }
}

@keyframes draw-line {
  from { stroke-dashoffset: 360; opacity: 0.2; }
  to { stroke-dashoffset: 0; opacity: 1; }
}

@media (max-width: 1120px) {
  #app-shell {
    width: min(100vw - 22px, 980px);
    flex-wrap: wrap !important;
  }

  #left-rail {
    min-width: 100% !important;
    max-width: 100% !important;
    height: auto;
    position: relative !important;
    top: 0;
  }

  .side-nav {
    grid-template-columns: repeat(5, minmax(86px, 1fr));
    overflow-x: auto;
  }

  #right-panel {
    display: none !important;
  }
}

@media (max-width: 760px) {
  #app-shell {
    width: calc(100vw - 14px);
    margin: 7px auto !important;
    gap: 10px !important;
  }

  #left-rail,
  #chat-panel {
    border-radius: 20px !important;
    padding: 12px !important;
  }

  .brand-lockup {
    margin-bottom: 12px;
  }

  .side-nav {
    grid-template-columns: repeat(5, 54px);
  }

  .nav-link {
    justify-content: center;
    padding: 0;
  }

  .nav-link span:last-child {
    display: none;
  }

  .chat-topline {
    display: block;
    padding-bottom: 12px;
  }

  .signal-pill {
    margin-top: 12px;
  }

  .chat-title h1 {
    font-size: 26px;
  }

  #chatbot {
    min-height: 55vh !important;
  }

  #input-dock {
    gap: 8px !important;
    padding: 9px;
  }
}

/* Final visual pass: override Gradio component internals that otherwise keep white shells. */
.gradio-container {
  --body-text-color: rgba(248, 253, 255, 0.96);
  --body-text-color-subdued: rgba(218, 235, 250, 0.82);
  --block-title-text-color: rgba(239, 249, 255, 0.94);
  --block-label-text-color: rgba(226, 242, 255, 0.88);
  --block-info-text-color: rgba(205, 224, 242, 0.76);
  --block-background-fill: rgba(6, 18, 35, 0.18);
  --block-border-color: rgba(170, 223, 255, 0.16);
  --background-fill-primary: rgba(5, 15, 31, 0.28);
  --background-fill-secondary: rgba(9, 24, 45, 0.22);
  --input-background-fill: rgba(7, 19, 38, 0.54);
  --input-background-fill-focus: rgba(9, 25, 48, 0.62);
  --input-border-color: rgba(178, 226, 255, 0.18);
  --input-border-color-focus: rgba(190, 232, 255, 0.42);
  --input-placeholder-color: rgba(221, 237, 250, 0.68);
  --button-secondary-background-fill: rgba(9, 24, 45, 0.32);
  --button-secondary-background-fill-hover: rgba(17, 45, 76, 0.46);
  --button-secondary-text-color: rgba(240, 249, 255, 0.94);
}

.nav-link,
.metric-head span,
.metric-card small,
.chat-title p,
.brand-lockup small,
#chatbot .placeholder,
#chatbot [class*="placeholder"],
#chatbot .empty,
#chatbot [class*="empty"] {
  color: rgba(231, 243, 255, 0.88) !important;
  text-shadow: 0 1px 12px rgba(0, 0, 0, 0.55) !important;
}

.nav-link:hover,
.nav-link.active {
  color: rgba(250, 254, 255, 0.98) !important;
}

.metric-head strong,
.panel-title h2,
.chat-title h1 {
  color: rgba(250, 254, 255, 0.98) !important;
}

#file-upload,
#file-upload.block,
#status-box,
#status-box.block,
#chat-input,
#chat-input.block,
#input-dock .block,
#input-dock .form,
#knowledge-card .block,
#status-card .block {
  background: transparent !important;
  border-color: transparent !important;
  box-shadow: none !important;
}

#file-upload > *,
#status-box > *,
#chat-input > *,
#input-dock > * {
  background: transparent !important;
  border-color: transparent !important;
  box-shadow: none !important;
}

#file-upload label,
#file-upload label.container,
#file-upload .container,
#file-upload .wrap,
#file-upload .upload-container,
#file-upload [class*="dropzone"],
#file-upload [class*="file-upload"],
#file-upload [class*="upload-container"],
#file-upload [class*="upload"] {
  background:
    radial-gradient(circle at 50% 8%, rgba(150, 220, 255, 0.13), transparent 48%),
    linear-gradient(150deg, rgba(255, 255, 255, 0.060), rgba(120, 205, 255, 0.035) 54%, rgba(188, 161, 255, 0.050)),
    rgba(7, 20, 39, 0.44) !important;
  border: 1px dashed rgba(188, 231, 255, 0.28) !important;
  border-radius: 20px !important;
  color: rgba(246, 252, 255, 0.94) !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.12), 0 10px 26px rgba(0, 0, 0, 0.12) !important;
}

#file-upload label,
#file-upload label.container {
  min-height: 150px !important;
}

#file-upload .wrap {
  min-height: 150px !important;
}

#file-upload button,
#file-upload .label-clear-button,
#file-upload [role="button"] {
  background:
    linear-gradient(135deg, rgba(130, 223, 255, 0.22), rgba(189, 168, 255, 0.16)),
    rgba(7, 20, 39, 0.48) !important;
  border: 1px solid rgba(188, 231, 255, 0.24) !important;
  color: rgba(246, 252, 255, 0.96) !important;
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.16) !important;
}

#file-upload svg,
#file-upload path {
  color: rgba(202, 237, 255, 0.92) !important;
  stroke: rgba(202, 237, 255, 0.92) !important;
}

#file-upload p,
#file-upload span,
#file-upload a,
#file-upload div {
  color: rgba(244, 251, 255, 0.92) !important;
  text-shadow: 0 1px 12px rgba(0, 0, 0, 0.56) !important;
}

#status-card {
  min-height: 160px;
}

#status-box label,
#status-box label.container,
#status-box .container,
#status-box .wrap {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

#status-box textarea,
#status-box label.container textarea,
#status-box textarea:disabled {
  min-height: 96px !important;
  color: rgba(242, 250, 255, 0.94) !important;
  -webkit-text-fill-color: rgba(242, 250, 255, 0.94) !important;
  background:
    linear-gradient(145deg, rgba(255, 255, 255, 0.050), rgba(125, 210, 255, 0.025)),
    rgba(7, 20, 39, 0.40) !important;
  border: 1px solid rgba(188, 231, 255, 0.16) !important;
  border-radius: 18px !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.10) !important;
}

#input-dock {
  background:
    radial-gradient(circle at 12% 0%, rgba(130, 223, 255, 0.09), transparent 40%),
    linear-gradient(135deg, rgba(255, 255, 255, 0.035), rgba(109, 197, 255, 0.024)),
    rgba(5, 15, 31, 0.34) !important;
  border-color: rgba(184, 230, 255, 0.15) !important;
}

#chat-input label,
#chat-input label.container,
#chat-input .container,
#chat-input .wrap,
#chat-input .input-container {
  height: auto !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

#chat-input textarea,
#chat-input label.container textarea,
#chat-input textarea:focus,
textarea#chat-input,
textarea#chat-input:focus {
  color: rgba(247, 253, 255, 0.98) !important;
  -webkit-text-fill-color: rgba(247, 253, 255, 0.98) !important;
  background:
    linear-gradient(145deg, rgba(255, 255, 255, 0.060), rgba(125, 210, 255, 0.036)),
    rgba(7, 20, 39, 0.54) !important;
  border: 1px solid rgba(190, 232, 255, 0.20) !important;
  border-radius: 20px !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.11), 0 8px 24px rgba(0, 0, 0, 0.10) !important;
}

#chat-input textarea::placeholder,
textarea#chat-input::placeholder {
  color: rgba(229, 241, 252, 0.74) !important;
  opacity: 1 !important;
}

#chatbot,
#chatbot .wrap,
#chatbot [class*="wrap"],
#chatbot [class*="container"] {
  color: rgba(244, 251, 255, 0.94) !important;
}

#chatbot p,
#chatbot span,
#chatbot div {
  color: inherit;
}

.mini-label {
  margin: 0 0 9px;
  color: rgba(232, 246, 255, 0.88);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-shadow: 0 1px 10px rgba(0, 0, 0, 0.50);
}

#status-box {
  min-height: 96px;
  padding: 14px 15px;
  border: 1px solid rgba(188, 231, 255, 0.16) !important;
  border-radius: 18px;
  background:
    linear-gradient(145deg, rgba(255, 255, 255, 0.050), rgba(125, 210, 255, 0.025)),
    rgba(7, 20, 39, 0.40) !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.10) !important;
  color: rgba(242, 250, 255, 0.94) !important;
}

#status-box p,
#status-box span,
#status-box div {
  color: rgba(242, 250, 255, 0.94) !important;
  text-shadow: 0 1px 10px rgba(0, 0, 0, 0.48);
}

#image-well {
  position: relative;
  z-index: 1;
  margin-top: 14px;
  padding: 14px;
  border: 1px solid rgba(183, 226, 255, 0.13);
  border-radius: 20px;
  background:
    radial-gradient(circle at 18% 0%, rgba(130, 223, 255, 0.10), transparent 42%),
    linear-gradient(145deg, rgba(255, 255, 255, 0.050), rgba(189, 168, 255, 0.034)),
    rgba(5, 15, 31, 0.30);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.10), 0 12px 30px rgba(0, 0, 0, 0.14);
}

.vision-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}

.vision-head .mini-label {
  display: block;
  margin: 0 0 4px;
}

.vision-head strong {
  display: block;
  color: rgba(248, 253, 255, 0.96);
  font-size: 14px;
  line-height: 1.25;
  text-shadow: 0 1px 12px rgba(0, 0, 0, 0.48);
}

.vision-chip {
  flex: 0 0 auto;
  padding: 5px 8px;
  border: 1px solid rgba(139, 231, 210, 0.24);
  border-radius: 999px;
  background: rgba(94, 226, 197, 0.10);
  color: rgba(204, 255, 240, 0.92);
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.05em;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.10);
}

#image-input,
#image-input.block,
#image-input > *,
#image-input label,
#image-input .wrap,
#image-input .container {
  background: transparent !important;
  border-color: transparent !important;
  box-shadow: none !important;
}

#image-input img,
#image-input canvas,
#image-input [class*="image-container"],
#image-input [class*="upload"] {
  border-radius: 18px !important;
  border: 1px dashed rgba(188, 231, 255, 0.25) !important;
  background:
    linear-gradient(145deg, rgba(255, 255, 255, 0.050), rgba(125, 210, 255, 0.030)),
    rgba(7, 20, 39, 0.42) !important;
}

#image-note {
  margin-top: 8px;
  color: rgba(218, 235, 250, 0.78);
  font-size: 12px;
  line-height: 1.55;
  text-shadow: 0 1px 10px rgba(0, 0, 0, 0.46);
}

#session-card,
#profile-card {
  position: relative;
  z-index: 1;
  margin-top: 14px;
  padding: 14px;
  border: 1px solid rgba(183, 226, 255, 0.13);
  border-radius: 20px;
  background:
    radial-gradient(circle at 16% 0%, rgba(130, 223, 255, 0.09), transparent 42%),
    linear-gradient(145deg, rgba(255, 255, 255, 0.045), rgba(189, 168, 255, 0.030)),
    rgba(5, 15, 31, 0.26);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.10), 0 12px 30px rgba(0, 0, 0, 0.13);
}

#current-session-display,
#profile-status {
  padding: 10px 11px;
  border: 1px solid rgba(188, 231, 255, 0.14);
  border-radius: 14px;
  background: rgba(7, 20, 39, 0.34);
}

#current-session-display p,
#profile-status p {
  margin: 0;
  color: rgba(238, 249, 255, 0.90) !important;
  font-size: 12.5px;
  line-height: 1.5;
}

#session-dropdown,
#gender-input,
#age-input,
#health-input {
  color: rgba(247, 253, 255, 0.96) !important;
  background:
    linear-gradient(145deg, rgba(255, 255, 255, 0.055), rgba(125, 210, 255, 0.030)),
    rgba(7, 20, 39, 0.48) !important;
  border: 1px solid rgba(190, 232, 255, 0.18) !important;
  border-radius: 16px !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.10) !important;
}

#session-dropdown *,
#gender-input *,
#age-input *,
#health-input * {
  color: rgba(247, 253, 255, 0.94) !important;
}

#age-input::placeholder,
#health-input::placeholder {
  color: rgba(229, 241, 252, 0.70) !important;
}

#new-session-btn,
#save-profile-btn,
#new-session-btn button,
#save-profile-btn button {
  min-height: 38px !important;
  background:
    linear-gradient(135deg, rgba(130, 223, 255, 0.20), rgba(189, 168, 255, 0.14)),
    rgba(7, 20, 39, 0.38) !important;
}

/* High contrast corrections for Gradio-rendered buttons and chat bubbles. */
#send-btn,
#send-btn button,
button#send-btn,
#input-dock #send-btn,
#input-dock button#send-btn {
  color: #061124 !important;
  -webkit-text-fill-color: #061124 !important;
  font-weight: 900 !important;
  text-shadow: 0 1px 0 rgba(255, 255, 255, 0.22) !important;
  background:
    linear-gradient(135deg, rgba(161, 236, 255, 0.98), rgba(175, 186, 255, 0.92) 55%, rgba(244, 173, 226, 0.82)) !important;
  border-color: rgba(210, 244, 255, 0.62) !important;
  box-shadow: 0 16px 36px rgba(89, 205, 255, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.45) !important;
}

#chatbot [data-testid*="user"],
#chatbot [class*="user"],
#chatbot .user,
#chatbot [data-testid*="bot"],
#chatbot [class*="bot"],
#chatbot [class*="assistant"] {
  color: rgba(248, 253, 255, 0.98) !important;
}

#chatbot [data-testid*="user"] *,
#chatbot [class*="user"] *,
#chatbot .user *,
#chatbot [data-testid*="bot"] *,
#chatbot [class*="bot"] *,
#chatbot [class*="assistant"] * {
  color: rgba(248, 253, 255, 0.98) !important;
  -webkit-text-fill-color: rgba(248, 253, 255, 0.98) !important;
  text-shadow: 0 1px 12px rgba(0, 0, 0, 0.50) !important;
}

#chatbot [class*="message"],
#chatbot [class*="bubble"],
#chatbot [data-testid*="user"] [class*="bubble"],
#chatbot [class*="user"] [class*="bubble"],
#chatbot .user .bubble {
  background:
    linear-gradient(135deg, rgba(48, 91, 132, 0.70), rgba(40, 62, 112, 0.68) 58%, rgba(99, 82, 145, 0.55)) !important;
  border: 1px solid rgba(194, 233, 255, 0.46) !important;
  box-shadow: 0 16px 36px rgba(0, 0, 0, 0.24), inset 0 1px 0 rgba(255, 255, 255, 0.18) !important;
}

#chatbot [data-testid*="bot"] [class*="bubble"],
#chatbot [class*="bot"] [class*="bubble"],
#chatbot [class*="assistant"] [class*="bubble"],
#chatbot .bot .bubble {
  background:
    linear-gradient(135deg, rgba(57, 79, 126, 0.72), rgba(48, 91, 132, 0.64) 62%, rgba(73, 115, 111, 0.48)) !important;
  border-color: rgba(190, 228, 255, 0.42) !important;
}

#chatbot [class*="copy"],
#chatbot button,
#chatbot button * {
  color: rgba(236, 248, 255, 0.92) !important;
  -webkit-text-fill-color: rgba(236, 248, 255, 0.92) !important;
}

.nav-link.nav-flash {
  border-color: rgba(219, 244, 255, 0.86) !important;
  box-shadow: 0 0 0 1px rgba(130, 223, 255, 0.22), 0 18px 42px rgba(89, 205, 255, 0.28) !important;
}

.focus-pulse {
  animation: focus-pulse 0.95s ease both;
}

@keyframes focus-pulse {
  0%, 100% { box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,0.16), var(--glow); }
  45% { box-shadow: 0 0 0 1px rgba(190,232,255,0.42), 0 0 42px rgba(130,223,255,0.30), var(--shadow); }
}
"""

APP_JS = r"""
() => {
  if (window.__healthNebulaBooted) return;
  window.__healthNebulaBooted = true;

  const canvas = document.createElement("canvas");
  canvas.id = "cosmic-particles";
  document.body.prepend(canvas);

  const aura = document.createElement("div");
  aura.id = "cursor-aura";
  const trail = document.createElement("div");
  trail.id = "cursor-trail";
  document.body.append(aura, trail);

  const ctx = canvas.getContext("2d", { alpha: true });
  let width = 0;
  let height = 0;
  let dpr = 1;
  let particles = [];
  let lastMove = 0;
  const mouse = { x: -999, y: -999, tx: -999, ty: -999, active: false };

  const makeParticles = () => {
    const count = Math.min(92, Math.max(46, Math.floor((width * height) / 23000)));
    particles = Array.from({ length: count }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.14,
      vy: (Math.random() - 0.5) * 0.14,
      r: Math.random() * 1.7 + 0.45,
      a: Math.random() * 0.55 + 0.18
    }));
  };

  const resize = () => {
    dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    makeParticles();
  };

  const moveCursor = (event) => {
    mouse.tx = event.clientX;
    mouse.ty = event.clientY;
    mouse.active = true;
    lastMove = performance.now();
    document.body.classList.add("cursor-active");
    document.documentElement.style.setProperty("--mouse-x", `${(event.clientX / width) * 100}%`);
    document.documentElement.style.setProperty("--mouse-y", `${(event.clientY / height) * 100}%`);
    document.documentElement.style.setProperty("--cursor-x", `${event.clientX}px`);
    document.documentElement.style.setProperty("--cursor-y", `${event.clientY}px`);

    const hot = event.target.closest("button, a, textarea, input, .metric-card, #chatbot .bubble, [class*='file']");
    document.body.classList.toggle("ui-hovering", Boolean(hot));
  };

  const leaveCursor = () => {
    mouse.active = false;
    document.body.classList.remove("cursor-active", "ui-hovering");
  };

  const bindQuickPrompts = () => {
    document.querySelectorAll("[data-prompt]").forEach((button) => {
      if (button.dataset.bound) return;
      button.dataset.bound = "1";
      button.addEventListener("click", () => {
        const textbox = document.querySelector("#chat-input textarea, #chat-input input");
        if (!textbox) return;
        textbox.value = button.dataset.prompt || "";
        textbox.dispatchEvent(new Event("input", { bubbles: true }));
        textbox.focus();
      });
    });
  };

  const bindNav = () => {
    document.querySelectorAll(".nav-link[data-target]").forEach((link) => {
      if (link.dataset.navBound) return;
      link.dataset.navBound = "1";
      link.addEventListener("click", (event) => {
        event.preventDefault();
        document.querySelectorAll(".nav-link").forEach((item) => item.classList.remove("active", "nav-flash"));
        link.classList.add("active", "nav-flash");
        const target = document.getElementById(link.dataset.target);
        if (target) {
          target.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
          target.classList.remove("focus-pulse");
          void target.offsetWidth;
          target.classList.add("focus-pulse");
          const input = target.querySelector("textarea, input, button, [tabindex]");
          if (input && target.id !== "chat-panel") {
            setTimeout(() => input.focus({ preventScroll: true }), 300);
          }
        }
        setTimeout(() => link.classList.remove("nav-flash"), 950);
      });
    });
  };

  const draw = () => {
    mouse.x += (mouse.tx - mouse.x) * 0.16;
    mouse.y += (mouse.ty - mouse.y) * 0.16;
    aura.style.transform = `translate3d(${mouse.x}px, ${mouse.y}px, 0) scale(${document.body.classList.contains("ui-hovering") ? 1.16 : 0.96})`;
    trail.style.transform = `translate3d(${mouse.x + (mouse.tx - mouse.x) * 0.25}px, ${mouse.y + (mouse.ty - mouse.y) * 0.25}px, 0)`;

    if (performance.now() - lastMove > 2600) {
      document.body.classList.remove("cursor-active", "ui-hovering");
    }

    ctx.clearRect(0, 0, width, height);

    for (const p of particles) {
      if (mouse.active) {
        const dx = mouse.x - p.x;
        const dy = mouse.y - p.y;
        const dist = Math.hypot(dx, dy);
        if (dist < 180 && dist > 0.1) {
          const pull = (1 - dist / 180) * 0.022;
          p.vx += (dx / dist) * pull;
          p.vy += (dy / dist) * pull;
        }
      }

      p.vx *= 0.965;
      p.vy *= 0.965;
      p.x += p.vx;
      p.y += p.vy;

      if (p.x < -10) p.x = width + 10;
      if (p.x > width + 10) p.x = -10;
      if (p.y < -10) p.y = height + 10;
      if (p.y > height + 10) p.y = -10;

      ctx.beginPath();
      ctx.fillStyle = `rgba(172, 226, 255, ${p.a})`;
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }

    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const a = particles[i];
        const b = particles[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dist = Math.hypot(dx, dy);
        if (dist < 118) {
          ctx.beginPath();
          ctx.strokeStyle = `rgba(132, 216, 255, ${(1 - dist / 118) * 0.16})`;
          ctx.lineWidth = 1;
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }

    requestAnimationFrame(draw);
  };

  window.addEventListener("resize", resize, { passive: true });
  window.addEventListener("pointermove", moveCursor, { passive: true });
  window.addEventListener("pointerleave", leaveCursor, { passive: true });
  resize();
  bindQuickPrompts();
  bindNav();
  setInterval(bindQuickPrompts, 1200);
  setInterval(bindNav, 1200);
  requestAnimationFrame(draw);
}
"""

NAV_HTML = """
<div class="brand-lockup">
  <span class="brand-mark">H</span>
  <div>
    <strong>Health Nebula</strong>
    <small>AI 养生助手</small>
  </div>
</div>
<nav class="side-nav">
  <a class="nav-link active" href="#chat-panel" data-target="chat-panel"><span class="nav-glyph">01</span><span>智能问答</span></a>
  <a class="nav-link" href="#profile-card" data-target="profile-card"><span class="nav-glyph">02</span><span>健康档案</span></a>
  <a class="nav-link" href="#knowledge-card" data-target="knowledge-card"><span class="nav-glyph">03</span><span>知识库</span></a>
  <a class="nav-link" href="#quick-actions" data-target="quick-actions"><span class="nav-glyph">04</span><span>调养方案</span></a>
  <a class="nav-link" href="#session-card" data-target="session-card"><span class="nav-glyph">05</span><span>会话设置</span></a>
</nav>
"""

CHAT_HEADER_HTML = """
<div class="chat-topline">
  <div class="chat-title">
    <h1>健康养生 AI 对话中枢</h1>
    <p>基于本地知识库进行多轮问答，回答会优先引用已上传资料，并保持必要的健康风险边界。</p>
  </div>
  <div class="signal-pill"><span class="signal-dot"></span><span>RAG Core Online</span></div>
</div>
"""

RIGHT_PANEL_HTML = """
<div class="panel-title">
  <span class="eyebrow">HEALTH SIGNALS</span>
  <h2>今日健康态势</h2>
</div>
<div class="health-grid">
  <div class="metric-card">
    <div class="metric-head"><span>睡眠修复</span><strong>7.2h</strong></div>
    <div class="progress-track"><span style="--value: 82%"></span></div>
    <small>深睡稳定，建议保持固定入睡节律</small>
  </div>
  <div class="metric-card">
    <div class="metric-head"><span>饮食均衡</span><strong>82</strong></div>
    <div class="progress-track"><span style="--value: 74%"></span></div>
    <small>清淡温润，减少夜间高糖摄入</small>
  </div>
  <div class="metric-card">
    <div class="metric-head"><span>运动活力</span><strong>6.4k</strong></div>
    <div class="progress-track"><span style="--value: 68%"></span></div>
    <small>午后轻运动窗口仍可补足</small>
  </div>
  <div class="metric-card">
    <div class="metric-head"><span>节律曲线</span><strong>平稳</strong></div>
    <svg class="pulse-chart" viewBox="0 0 260 96" role="img" aria-label="健康节律趋势">
      <defs>
        <linearGradient id="pulseLine" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#8deac8" />
          <stop offset="48%" stop-color="#82dfff" />
          <stop offset="100%" stop-color="#bda8ff" />
        </linearGradient>
      </defs>
      <polyline points="4,68 35,55 64,61 94,36 126,44 156,24 188,38 220,31 256,18" />
    </svg>
    <small>睡眠、饮食、运动呈轻微上行趋势</small>
  </div>
</div>
"""


def make_initial_sessions():
    sessions, current_id = create_session({})
    return sessions, current_id


def new_session(sessions):
    sessions, current_id = create_session(sessions or {})
    return (
        sessions,
        current_id,
        gr.update(choices=session_choices(sessions), value=current_id),
        [],
        current_session_label(sessions, current_id),
        "",
        None,
    )


def switch_session(sessions, selected_id):
    sessions = sessions or {}
    current_id = int(selected_id or min(sessions.keys(), default=1))
    if current_id not in sessions:
        sessions, current_id = create_session(sessions)
    return current_id, sessions[current_id].get("history", []), current_session_label(sessions, current_id), "", None


def clear_current_session(sessions, current_id):
    sessions = sessions or {}
    current_id = int(current_id or min(sessions.keys(), default=1))
    if current_id not in sessions:
        sessions, current_id = create_session(sessions)
    sessions[current_id].update({
        "name": f"对话{current_id}",
        "history": [],
        "first_question": None,
        "dialog_state": "idle",
        "collected_info": DEFAULT_COLLECTED_INFO.copy(),
        "ask_step": 0,
    })
    return (
        [],
        sessions,
        gr.update(choices=session_choices(sessions), value=current_id),
        current_session_label(sessions, current_id),
        "",
        None,
    )


def save_profile(age, gender, health):
    profile = {
        "age": str(age or "").strip(),
        "gender": gender or "保密",
        "health": str(health or "").strip(),
    }
    summary = profile_to_text(profile) or "未填写画像。"
    logger.info(f"画像已更新 | {sanitize_for_log(summary)}")
    return profile, f"已保存：{summary}"


initial_sessions, initial_session_id = make_initial_sessions()

with gr.Blocks(
    title="健康养生 AI 对话助手",
    fill_height=True,
    fill_width=True,
    analytics_enabled=False,
) as demo:
    sessions_state = gr.State(initial_sessions)
    current_session_state = gr.State(initial_session_id)
    user_profile_state = gr.State({"age": "", "gender": "保密", "health": ""})

    with gr.Row(elem_id="app-shell"):
        with gr.Column(elem_id="left-rail", scale=0, min_width=226):
            gr.HTML(NAV_HTML)
            with gr.Column(elem_id="session-card"):
                gr.HTML('<div class="mini-label">多会话管理</div>')
                current_session_display = gr.Markdown(
                    value=current_session_label(initial_sessions, initial_session_id),
                    elem_id="current-session-display",
                )
                session_dropdown = gr.Dropdown(
                    choices=session_choices(initial_sessions),
                    value=initial_session_id,
                    label=None,
                    show_label=False,
                    container=False,
                    elem_id="session-dropdown",
                )
                new_session_btn = gr.Button("新建会话", elem_id="new-session-btn")
            with gr.Column(elem_id="knowledge-card"):
                gr.HTML('<div class="mini-label">上传资料</div>')
                file_upload = gr.File(
                    file_count="multiple",
                    label=None,
                    show_label=False,
                    height=150,
                    elem_id="file-upload",
                    elem_classes=["glass-hover"],
                )
                build_btn = gr.Button("构建知识库", variant="primary", elem_id="build-btn")
            with gr.Column(elem_id="status-card"):
                gr.HTML('<div class="mini-label">知识库状态</div>')
                status_text = gr.Markdown(
                    value="等待上传资料并构建知识库。",
                    elem_id="status-box",
                )

        with gr.Column(elem_id="chat-panel", scale=1, min_width=420):
            gr.HTML(CHAT_HEADER_HTML)
            chatbot = gr.Chatbot(
                height="calc(100vh - 266px)",
                min_height=420,
                show_label=False,
                render_markdown=True,
                layout="bubble",
                placeholder="上传资料并构建知识库后，即可开始健康养生问答。",
                elem_id="chatbot",
            )
            with gr.Row(elem_id="input-dock"):
                msg = gr.Textbox(
                    placeholder="输入你的养生问题，例如：最近睡眠浅，适合怎样调理？",
                    show_label=False,
                    lines=2,
                    max_lines=5,
                    autofocus=True,
                    container=False,
                    scale=8,
                    elem_id="chat-input",
                )
                send_btn = gr.Button("发送", variant="primary", scale=1, min_width=86, elem_id="send-btn")
                clear = gr.Button("清空", scale=1, min_width=86, elem_id="clear-btn")

        with gr.Column(elem_id="right-panel", scale=0, min_width=278):
            gr.HTML(RIGHT_PANEL_HTML)
            with gr.Column(elem_id="image-well"):
                gr.HTML('<div class="vision-head"><div><span class="mini-label">图片观察</span><strong>上传后随问题识别</strong></div><span class="vision-chip">Vision</span></div>')
                image_input = gr.Image(
                    type="filepath",
                    label=None,
                    show_label=False,
                    height=150,
                    elem_id="image-input",
                    elem_classes=["vision-upload"],
                )
                gr.HTML('<div id="image-note">支持上传舌象、食材等健康相关图片。发送问题时会自动识别，并把图片观察结果融入回答。</div>')
            with gr.Column(elem_id="profile-card"):
                gr.HTML('<div class="mini-label">用户画像</div>')
                age_input = gr.Textbox(
                    placeholder="年龄",
                    label=None,
                    show_label=False,
                    container=False,
                    elem_id="age-input",
                )
                gender_input = gr.Dropdown(
                    choices=["保密", "男", "女"],
                    value="保密",
                    label=None,
                    show_label=False,
                    container=False,
                    elem_id="gender-input",
                )
                health_input = gr.Textbox(
                    placeholder="健康状况/关注点，例如：睡眠浅、脾胃弱",
                    label=None,
                    show_label=False,
                    lines=2,
                    max_lines=3,
                    container=False,
                    elem_id="health-input",
                )
                save_profile_btn = gr.Button("保存画像", elem_id="save-profile-btn")
                profile_status = gr.Markdown(value="画像仅保存在当前前端会话状态中。", elem_id="profile-status")
            with gr.Column(elem_id="quick-actions"):
                quick_sleep = gr.Button("睡眠调养", elem_classes=["quick-button"])
                quick_diet = gr.Button("饮食建议", elem_classes=["quick-button"])
                quick_motion = gr.Button("运动节律", elem_classes=["quick-button"])
                quick_plan = gr.Button("一周方案", elem_classes=["quick-button"])

    build_btn.click(build_knowledge_base, inputs=file_upload, outputs=status_text)
    msg.submit(
        respond,
        [msg, chatbot, sessions_state, current_session_state, user_profile_state, image_input],
        [msg, chatbot, sessions_state, session_dropdown, current_session_display],
    )
    send_btn.click(
        respond,
        [msg, chatbot, sessions_state, current_session_state, user_profile_state, image_input],
        [msg, chatbot, sessions_state, session_dropdown, current_session_display],
    )
    clear.click(
        clear_current_session,
        [sessions_state, current_session_state],
        [chatbot, sessions_state, session_dropdown, current_session_display, msg, image_input],
    )
    new_session_btn.click(
        new_session,
        inputs=[sessions_state],
        outputs=[sessions_state, current_session_state, session_dropdown, chatbot, current_session_display, msg, image_input],
    )
    session_dropdown.change(
        switch_session,
        inputs=[sessions_state, session_dropdown],
        outputs=[current_session_state, chatbot, current_session_display, msg, image_input],
    )
    save_profile_btn.click(
        save_profile,
        inputs=[age_input, gender_input, health_input],
        outputs=[user_profile_state, profile_status],
    )
    quick_sleep.click(lambda: "最近睡眠浅、容易醒，适合从作息和饮食上怎样调理？", outputs=msg)
    quick_diet.click(lambda: "最近脾胃不太舒服，日常饮食应该注意什么？", outputs=msg)
    quick_motion.click(lambda: "请根据养生原则，给我一个温和的每日运动节律建议。", outputs=msg)
    quick_plan.click(lambda: "请结合睡眠、饮食、运动，给我制定一个一周养生计划。", outputs=msg)

# 启动
if __name__ == "__main__":
    if has_dashscope_key():
        embeddings = DashScopeEmbeddings(model="text-embedding-v1")

        try:
            db = load_chroma_store(embeddings)
            print("已加载现有向量知识库")
            schedule_bm25_rebuild()
        except Exception as e:
            db = None
            bm25_index = None
            bm25_docs = []
            print(f"未找到现有知识库：{e}")
    else:
        db = None
        bm25_index = None
        bm25_docs = []
        print("未设置 DASHSCOPE_API_KEY，已跳过现有知识库加载。页面会正常启动，但问答/构建知识库前需要配置 .env。")

    launch_port = get_launch_port()
    print(f"准备启动服务：http://127.0.0.1:{launch_port}")
    demo.queue().launch(
        server_port=launch_port,
        show_error=True,
        footer_links=[],
        css=APP_CSS,
        js=APP_JS,
        head=APP_HEAD,
    )
