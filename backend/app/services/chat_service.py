import asyncio
import hashlib
import json
import uuid
from collections import defaultdict

from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatTongyi

from backend.app.config import settings
from backend.app.schemas import ChatRequest, ChatResponse, SourceSnippet, ToolResult, UserProfile
from backend.app.services.cache_service import cache_service
from backend.app.services.health_tool_service import health_tool_service
from backend.app.services.image_service import analyze_image, is_image_analysis_error
from backend.app.services.rag_service import rag_service


HEALTH_KEYWORDS = [
    "痛", "酸", "胀", "咳", "眠", "睡", "胃", "脾", "肝", "肾", "湿", "火", "寒", "热",
    "舌", "苔", "便秘", "腹泻", "乏力", "头晕", "月经", "养生", "调理", "饮食", "运动",
]

RED_FLAG_KEYWORDS = [
    "胸痛", "呼吸困难", "昏迷", "抽搐", "大出血", "剧烈头痛", "偏瘫", "高烧不退", "黑便", "咯血",
]

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

DISCLAIMER = "\n\n提示：以上内容仅用于健康养生科普，不能替代医生诊断；如症状持续、加重或出现急性不适，请及时就医。"


class ChatService:
    def __init__(self) -> None:
        self._llm: ChatTongyi | None = None
        self.sessions: dict[str, list[dict[str, str]]] = defaultdict(list)

    @property
    def llm(self) -> ChatTongyi:
        if self._llm is None:
            self._llm = ChatTongyi(model=settings.llm_model, temperature=0.1)
        return self._llm

    async def answer(self, request: ChatRequest) -> ChatResponse:
        message = (request.message or "").strip()
        if len(message) > settings.max_input_length:
            message = message[:settings.max_input_length]

        session_id = request.session_id or str(uuid.uuid4())
        image_description = None

        if not message and request.image_path:
            message = "请根据我上传的图片提供健康养生相关分析和建议。"

        if not message:
            return ChatResponse(session_id=session_id, answer="请输入问题后再发送。")

        if health_tool_service.is_tool_capability_question(message):
            answer = health_tool_service.capability_response()
            self._append_history(session_id, message, answer)
            return ChatResponse(session_id=session_id, answer=answer)

        if not settings.dashscope_api_key:
            return ChatResponse(
                session_id=session_id,
                answer="后端未设置 DASHSCOPE_API_KEY，无法调用通义千问模型。请先在 .env 中配置密钥。",
            )

        profile_text = self._profile_to_text(request.user_profile)
        cache_key = self._cache_key(message, profile_text) if not request.image_path else None
        if cache_key:
            cached = await cache_service.get_json(cache_key)
            if cached:
                answer = cached["answer"]
                self._append_history(session_id, message, answer)
                return ChatResponse(
                    session_id=session_id,
                    answer=answer,
                    cached=True,
                    sources=[SourceSnippet(**item) for item in cached.get("sources", [])],
                    tool_results=[ToolResult(**item) for item in cached.get("tool_results", [])],
                )

        final_query = message
        image_hint = "本轮未上传图片。"
        if request.image_path:
            image_description = await asyncio.to_thread(analyze_image, request.image_path)
            if image_description and not is_image_analysis_error(image_description):
                image_hint = (
                    "用户上传了图片。你看到的是图像识别后的内容，回答时使用“根据你上传的图片”或“从图片来看”。"
                    "图片识别结果仅作健康养生科普参考，不可替代医生诊断。"
                )
                final_query = self._build_image_augmented_query(message, image_description)
            else:
                final_query = f"{message}\n\n图片识别未成功：{image_description or '未获得有效图片描述'}"

        expanded_query = self._expand_query(final_query)
        raw_tool_results = health_tool_service.run_tools(final_query, request.user_profile)
        tool_results = [ToolResult(**item) for item in raw_tool_results]
        if raw_tool_results and not request.image_path and not self._is_health_query(final_query):
            answer = health_tool_service.direct_answer(raw_tool_results)
            self._append_history(session_id, message, answer)
            return ChatResponse(
                session_id=session_id,
                answer=answer,
                tool_results=tool_results,
            )

        if raw_tool_results and rag_service.db is None:
            top_docs = []
        elif raw_tool_results and not self._is_health_query(final_query) and not request.image_path:
            top_docs = []
        else:
            top_docs = await asyncio.to_thread(rag_service.retrieve_docs, expanded_query)
        context = "\n\n---\n\n".join([doc.page_content for doc in top_docs]) if top_docs else "暂无相关知识库资料。"
        sources = [
            SourceSnippet(index=index + 1, content=doc.page_content[:500])
            for index, doc in enumerate(top_docs)
        ]

        prompt = self._build_prompt(
            final_query=final_query,
            context=context,
            profile_text=profile_text,
            history=self.sessions[session_id][-4:],
            image_hint=image_hint,
            tool_context=health_tool_service.format_for_prompt(raw_tool_results),
            tool_catalog=health_tool_service.format_catalog_for_prompt(),
        )
        answer = await self._call_llm(prompt)

        if (self._is_health_query(final_query) or request.image_path) and "不能替代医生诊断" not in answer:
            answer += DISCLAIMER

        self._append_history(session_id, message, answer)

        if cache_key:
            await cache_service.set_json(
                cache_key,
                {
                    "answer": answer,
                    "sources": [source.dict() for source in sources],
                    "tool_results": [tool.dict() for tool in tool_results],
                },
            )

        return ChatResponse(
            session_id=session_id,
            answer=answer,
            image_description=image_description,
            sources=sources,
            tool_results=tool_results,
        )

    async def stream_answer(self, request: ChatRequest):
        response = await self.answer(request)
        yield f"event: session\ndata: {json.dumps({'session_id': response.session_id}, ensure_ascii=False)}\n\n"
        for index in range(0, len(response.answer), 6):
            chunk = response.answer[index:index + 6]
            yield f"event: token\ndata: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)
        payload = response.dict()
        yield f"event: done\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def list_sessions(self) -> list[dict[str, str | int | None]]:
        result = []
        for session_id, turns in self.sessions.items():
            last_user = next((turn["content"] for turn in reversed(turns) if turn["role"] == "user"), None)
            result.append({"session_id": session_id, "turns": len(turns), "last_message": last_user})
        return result

    def delete_session(self, session_id: str) -> bool:
        return self.sessions.pop(session_id, None) is not None

    async def _call_llm(self, prompt: str) -> str:
        response = await asyncio.to_thread(lambda: self.llm.invoke([HumanMessage(content=prompt)]))
        return response.content

    def _append_history(self, session_id: str, message: str, answer: str) -> None:
        self.sessions[session_id].append({"role": "user", "content": message})
        self.sessions[session_id].append({"role": "assistant", "content": answer})

    def _profile_to_text(self, profile: UserProfile | None) -> str:
        if not profile:
            return ""
        parts = []
        if profile.age:
            parts.append(f"年龄：{profile.age}")
        if profile.gender and profile.gender != "保密":
            parts.append(f"性别：{profile.gender}")
        if profile.health:
            parts.append(f"健康状况/关注点：{profile.health}")
        return "；".join(parts)

    def _cache_key(self, message: str, profile_text: str) -> str:
        raw = f"{message}::{profile_text}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"chat:{digest}"

    def _expand_query(self, user_input: str) -> str:
        expansions = [mapping for symptom, mapping in ENTITY_EXPANSION.items() if symptom in user_input]
        return user_input + " " + " ".join(expansions) if expansions else user_input

    def _is_health_query(self, text: str) -> bool:
        return any(keyword in (text or "") for keyword in HEALTH_KEYWORDS)

    def _has_red_flag(self, text: str) -> bool:
        return any(keyword in (text or "") for keyword in RED_FLAG_KEYWORDS)

    def _build_image_augmented_query(self, message: str, image_desc: str) -> str:
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

    def _build_prompt(
        self,
        final_query: str,
        context: str,
        profile_text: str,
        history: list[dict[str, str]],
        image_hint: str,
        tool_context: str,
        tool_catalog: str,
    ) -> str:
        history_text = ""
        for turn in history:
            speaker = "用户" if turn["role"] == "user" else "助手"
            history_text += f"{speaker}：{turn['content']}\n"

        red_flag_rule = (
            "用户描述包含可能需要及时就医的危险信号。回答开头先提示尽快就医或寻求专业医生帮助，再给一般性养生注意事项。"
            if self._has_red_flag(final_query)
            else "如果用户描述出现胸痛、呼吸困难、昏迷、抽搐、大出血、偏瘫、高烧不退等危险信号，应优先建议就医。"
        )

        return f"""你是一个专业、谨慎的健康养生科普助手。请根据【参考资料】和【用户背景】回答问题。

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


chat_service = ChatService()
