import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any


@dataclass
class HealthToolResult:
    name: str
    title: str
    result: str
    data: dict[str, Any]


class HealthToolService:
    def run_tools(self, text: str, user_profile: dict[str, Any] | Any | None = None) -> list[dict[str, Any]]:
        profile = self._normalize_profile(user_profile)
        text = text or ""
        results: list[HealthToolResult] = []

        bmi = self._maybe_calculate_bmi(text)
        if bmi:
            results.append(bmi)

        water = self._maybe_calculate_water(text)
        if water:
            results.append(water)

        sleep = self._maybe_plan_sleep(text)
        if sleep:
            results.append(sleep)

        exercise = self._maybe_calculate_exercise(text, profile)
        if exercise:
            results.append(exercise)

        return [asdict(item) for item in results]

    def format_for_prompt(self, results: list[dict[str, Any]]) -> str:
        if not results:
            return "本轮未触发健康计算工具。"
        lines = []
        for item in results:
            lines.append(f"- {item['title']}：{item['result']}")
        return "\n".join(lines)

    def format_catalog_for_prompt(self) -> str:
        return "\n".join(
            f"- {tool['title']}（{tool['name']}）：{tool['description']}"
            for tool in self.list_tools()
        )

    def is_tool_capability_question(self, text: str) -> bool:
        text = text or ""
        capability_keywords = (
            "外部工具",
            "工具调用",
            "调用工具",
            "有哪些工具",
            "能用什么工具",
            "能调用什么",
            "可以调用工具",
            "你能调用",
        )
        return any(keyword in text for keyword in capability_keywords)

    def capability_response(self) -> str:
        return (
            "我目前可以调用项目内置的健康养生工具，但还没有接入联网搜索、天气、地图、日历等真正外部平台工具。\n\n"
            "已支持的内置健康工具包括：\n"
            f"{self.format_catalog_for_prompt()}\n\n"
            "例如你可以这样问：\n"
            "- 身高170cm，体重70kg，帮我算一下BMI。\n"
            "- 体重60kg，每天喝多少水比较合适？\n"
            "- 如果我早上7点起床，晚上几点睡比较好？\n"
            "- 我30岁，温和有氧运动心率控制在多少合适？"
        )

    def direct_answer(self, results: list[dict[str, Any]]) -> str:
        if not results:
            return ""
        lines = ["已调用健康计算工具，结果如下："]
        for item in results:
            lines.append(f"- {item['title']}：{item['result']}")
        lines.append("")
        lines.append("提示：以上结果用于健康养生科普和日常参考，不能替代医生或专业营养师的个体化评估。")
        return "\n".join(lines)

    def list_tools(self) -> list[dict[str, str]]:
        return [
            {
                "name": "bmi_calculator",
                "title": "BMI 计算工具",
                "description": "根据身高和体重计算 BMI，并给出基础体重管理参考。",
            },
            {
                "name": "water_intake_estimator",
                "title": "饮水量估算工具",
                "description": "根据体重估算每日温水摄入范围。",
            },
            {
                "name": "sleep_schedule_planner",
                "title": "睡眠作息工具",
                "description": "根据起床时间反推 7.5 小时和 9 小时睡眠对应的建议入睡时间。",
            },
            {
                "name": "exercise_heart_rate_estimator",
                "title": "运动心率工具",
                "description": "根据年龄估算温和有氧运动目标心率区间。",
            },
        ]

    def _normalize_profile(self, user_profile: dict[str, Any] | Any | None) -> dict[str, Any]:
        if not user_profile:
            return {}
        if isinstance(user_profile, dict):
            return user_profile
        data = {}
        for key in ("age", "gender", "health"):
            value = getattr(user_profile, key, None)
            if value is not None:
                data[key] = value
        return data

    def _maybe_calculate_bmi(self, text: str) -> HealthToolResult | None:
        if not any(keyword in text for keyword in ("BMI", "bmi", "体重", "身高", "肥胖", "减重", "超重")):
            return None

        height_cm = self._extract_height_cm(text)
        weight_kg = self._extract_weight_kg(text)
        if not height_cm or not weight_kg:
            return None

        height_m = height_cm / 100
        bmi = round(weight_kg / (height_m * height_m), 1)
        if bmi < 18.5:
            category = "偏瘦"
        elif bmi < 24:
            category = "正常范围"
        elif bmi < 28:
            category = "超重"
        else:
            category = "肥胖"

        return HealthToolResult(
            name="bmi_calculator",
            title="BMI 计算工具",
            result=f"身高 {height_cm:g} cm、体重 {weight_kg:g} kg，对应 BMI 约 {bmi}，属于{category}。",
            data={"height_cm": height_cm, "weight_kg": weight_kg, "bmi": bmi, "category": category},
        )

    def _maybe_calculate_water(self, text: str) -> HealthToolResult | None:
        if not any(keyword in text for keyword in ("喝水", "饮水", "补水", "水量", "温水", "喝多少水", "多少水")):
            return None

        weight_kg = self._extract_weight_kg(text)
        if not weight_kg:
            return None

        low = round(weight_kg * 30)
        high = round(weight_kg * 35)
        return HealthToolResult(
            name="water_intake_estimator",
            title="饮水量估算工具",
            result=f"按体重 {weight_kg:g} kg 估算，每日温水摄入可参考 {low}-{high} ml，分多次饮用更稳妥。",
            data={"weight_kg": weight_kg, "daily_water_ml_low": low, "daily_water_ml_high": high},
        )

    def _maybe_plan_sleep(self, text: str) -> HealthToolResult | None:
        if not any(keyword in text for keyword in ("几点睡", "几点起", "起床", "入睡", "睡眠计划", "作息")):
            return None

        wake_time = self._extract_time(text)
        if not wake_time:
            return None

        base = datetime(2000, 1, 2, wake_time[0], wake_time[1])
        bedtime_75 = (base - timedelta(hours=7, minutes=30)).strftime("%H:%M")
        bedtime_90 = (base - timedelta(hours=9)).strftime("%H:%M")
        wake_label = f"{wake_time[0]:02d}:{wake_time[1]:02d}"

        return HealthToolResult(
            name="sleep_schedule_planner",
            title="睡眠作息工具",
            result=f"如果计划 {wake_label} 起床，可参考 {bedtime_75} 入睡获得约 7.5 小时睡眠，或 {bedtime_90} 入睡获得约 9 小时睡眠。",
            data={"wake_time": wake_label, "bedtime_7_5h": bedtime_75, "bedtime_9h": bedtime_90},
        )

    def _maybe_calculate_exercise(self, text: str, profile: dict[str, Any]) -> HealthToolResult | None:
        if not any(keyword in text for keyword in ("运动", "有氧", "燃脂", "心率", "快走", "跑步", "锻炼")):
            return None

        age = self._extract_age(text) or self._to_number(profile.get("age"))
        if not age or age <= 0:
            return None

        max_hr = max(220 - age, 80)
        low = round(max_hr * 0.5)
        high = round(max_hr * 0.7)
        return HealthToolResult(
            name="exercise_heart_rate_estimator",
            title="运动心率工具",
            result=f"按年龄 {age:g} 岁估算，温和有氧运动目标心率可参考 {low}-{high} 次/分钟，运动中应以能说话但略喘为宜。",
            data={"age": age, "max_heart_rate_estimate": max_hr, "target_low": low, "target_high": high},
        )

    def _extract_height_cm(self, text: str) -> float | None:
        patterns = [
            r"身高\s*(\d+(?:\.\d+)?)\s*cm",
            r"身高\s*(\d+(?:\.\d+)?)\s*厘米",
            r"(\d+(?:\.\d+)?)\s*cm",
            r"(\d+(?:\.\d+)?)\s*厘米",
            r"身高\s*(1\.\d{1,2})\s*米",
            r"(1\.\d{1,2})\s*m",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            value = float(match.group(1))
            return value * 100 if value < 3 else value
        return None

    def _extract_weight_kg(self, text: str) -> float | None:
        patterns = [
            r"体重\s*(\d+(?:\.\d+)?)\s*kg",
            r"体重\s*(\d+(?:\.\d+)?)\s*公斤",
            r"体重\s*(\d+(?:\.\d+)?)\s*千克",
            r"(\d+(?:\.\d+)?)\s*kg",
            r"(\d+(?:\.\d+)?)\s*公斤",
            r"(\d+(?:\.\d+)?)\s*千克",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None

    def _extract_age(self, text: str) -> float | None:
        patterns = [
            r"年龄\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*岁",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
        return None

    def _extract_time(self, text: str) -> tuple[int, int] | None:
        patterns = [
            r"(\d{1,2})[:：](\d{1,2})",
            r"(\d{1,2})\s*点\s*(\d{1,2})?\s*分?",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            hour = int(match.group(1))
            minute = int(match.group(2) or 0)
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return hour, minute
        return None

    def _to_number(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(str(value).strip())
        except ValueError:
            return None


health_tool_service = HealthToolService()
