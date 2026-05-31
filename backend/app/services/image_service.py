import base64
import mimetypes
from pathlib import Path

from dashscope import MultiModalConversation

from backend.app.config import settings


def normalize_image_path(image: str | Path | None) -> Path | None:
    if not image:
        return None
    return Path(str(image)).expanduser().resolve()


def is_image_analysis_error(result: str) -> bool:
    if not result:
        return True
    return result.startswith(("图片分析失败：", "图片分析异常：", "图片处理异常："))


def analyze_image(image: str | Path | None) -> str:
    if not settings.dashscope_api_key:
        return "图片分析失败：后端未设置 DASHSCOPE_API_KEY。"

    image_path = normalize_image_path(image)
    if not image_path or not image_path.exists():
        return "图片处理异常：图片路径不存在。"

    try:
        image_bytes = image_path.read_bytes()
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        messages = [{
            "role": "user",
            "content": [
                {"image": f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"},
                {
                    "text": (
                        "请用中文简要描述这张图片的内容，重点关注与健康、养生相关的特征。"
                        "例如舌苔颜色、厚薄、裂纹、齿痕，或食材名称、新鲜度、烹饪状态等。"
                        "不要做医学诊断。"
                    )
                },
            ],
        }]
        response = MultiModalConversation.call(model=settings.vision_model, messages=messages)
        if response.status_code == 200:
            return response.output.choices[0].message.content[0]["text"]
        return f"图片分析失败：多模态 API 返回状态码 {response.status_code}，消息：{response.message}"
    except Exception as exc:
        return f"图片分析异常：{exc}"
