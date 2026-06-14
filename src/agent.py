"""
GUI Agent - 中兴捧月智能代理比赛
基于 CoAT (Chain-of-Action-Thought) + 防循环 + 历史管理 的 Agent 实现

核心设计思想：
1. CoAT 行动思维链：每步操作前强制模型进行"观察→思考→行动"的结构化推理
2. 历史轨迹管理：滑动窗口 + 语义摘要，保持决策连贯性
3. 防循环检测：基于动作历史的重复检测 + 截图哈希对比
4. 多轮对话：利用模型的上下文窗口，将历史操作作为对话历史传入
5. 动作验证：CLICK/TYPE/OPEN 后置合理性校验
6. 错误恢复：动作修正 + 策略切换 + 历史回退
7. APP名称映射：常见应用别名统一
"""

import re
import json
import hashlib
import logging
from typing import Dict, Any, List, Optional, Tuple

from agent_base import (
    BaseAgent, AgentInput, AgentOutput,
    ACTION_CLICK, ACTION_SCROLL, ACTION_TYPE,
    ACTION_OPEN, ACTION_COMPLETE, VALID_ACTIONS,
    UsageInfo
)

logger = logging.getLogger(__name__)

# ==========================================
#  APP 名称映射表
# ==========================================

# 键：模型可能输出的名称  →  值：ref.json 中的标准名称
APP_NAME_ALIASES: Dict[str, str] = {
    # 美团系列
    "美团外卖": "美团",
    "美团买菜": "美团",
    "美团优选": "美团",
    "meituan": "美团",
    # 淘宝系列
    "手机淘宝": "淘宝",
    "淘宝网": "淘宝",
    "taobao": "淘宝",
    # 抖音系列
    "tiktok": "抖音",
    "抖音短视频": "抖音",
    # 百度系列
    "百度地图": "百度地图",
    "百度": "百度地图",
    "baidumap": "百度地图",
    # B站
    "哔哩哔哩": "B站",
    "bilibili": "B站",
    "b站": "B站",
    # 快手
    "快手短视频": "快手",
    "kuaishou": "快手",
    # 腾讯视频
    "腾讯": "腾讯视频",
    "tengxunshipin": "腾讯视频",
    "视频": "腾讯视频",
    # 其他
    "大众点评": "大众点评",
    "dianping": "大众点评",
    "拼多多": "拼多多",
    "京东": "京东",
    "jd": "京东",
    "爱奇艺": "爱奇艺",
    "iqiyi": "爱奇艺",
    "芒果tv": "芒果TV",
    "芒果": "芒果TV",
    "喜马拉雅": "喜马拉雅",
    "去哪儿": "去哪儿",
    "去哪儿旅行": "去哪儿",
    "qunar": "去哪儿",
    "铁路12306": "铁路12306",
    "12306": "铁路12306",
    "小红书": "小红书",
    "微信": "微信",
    "支付宝": "支付宝",
}

# 构建「指令关键词 → 标准APP名」的二级映射
# 当 OPEN 名称无法在别名表中找到时，从用户指令中推断
APP_KEYWORD_MAP: Dict[str, str] = {
    "美团": "美团",
    "淘宝": "淘宝",
    "抖音": "抖音",
    "百度地图": "百度地图",
    "地图": "百度地图",
    "B站": "B站",
    "b站": "B站",
    "哔哩哔哩": "B站",
    "快手": "快手",
    "腾讯视频": "腾讯视频",
    "爱奇艺": "爱奇艺",
    "芒果TV": "芒果TV",
    "芒果": "芒果TV",
    "喜马拉雅": "喜马拉雅",
    "去哪儿": "去哪儿",
    "大众点评": "大众点评",
    "点评": "大众点评",
    "拼多多": "拼多多",
    "京东": "京东",
    "12306": "铁路12306",
    "小红书": "小红书",
}


class Agent(BaseAgent):
    """
    基于 CoAT 思维链的 GUI Agent

    核心优化：
    - CoAT (Chain-of-Action-Thought): 强制模型进行结构化思考
    - 空间感知增强：System Prompt 包含精确的屏幕分区坐标参考
    - 动作验证：CLICK/TYPE/OPEN 后置合理性校验
    - 错误恢复：动作修正 + 策略切换
    - APP名称映射：常见应用别名统一
    - 多轮对话历史：将历史操作作为对话上下文
    - 防循环检测：连续相同动作检测 + 突破策略
    - 输出解析重试：解析失败时自动重试
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._reset_state()

    def _reset_state(self):
        """重置 Agent 的内部状态"""
        self._action_history: List[Dict[str, Any]] = []
        self._message_history: List[Dict[str, Any]] = []
        self._current_instruction: str = ""
        self._last_screenshot_hash: str = ""
        self._consecutive_same_action: int = 0  # 连续相同动作计数（不含当前步）
        self._last_action_key: str = ""
        self._last_thought: str = ""
        # 新增：错误恢复状态
        self._action_toggle_count: int = 0  # CLICK↔TYPE 摇摆计数
        self._consecutive_click_count: int = 0  # 连续 CLICK 计数
        self._inferred_app: str = ""  # 从指令推断的目标 APP 名
        # 新增：子目标分解（借鉴 MobileAgent）
        self._task_plan: List[str] = []  # 子目标计划列表
        self._current_plan_step: int = 0  # 当前子目标索引
        # 新增：反思机制（借鉴 SeeClick/AppAgent）
        self._last_image: Any = None  # 上一轮截图（用于差异对比）
        self._stuck_count: int = 0  # 连续卡住计数（截图不变次数）
        self._scroll_magnitude: int = 400  # 当前滚动量（自适应）

    def reset(self):
        """每个测试用例开始前由 TestRunner 调用"""
        self._reset_state()

    def _initialize(self):
        """初始化方法"""
        pass

    # ==========================================
    #  核心方法：act()
    # ==========================================

    def act(self, input_data: AgentInput) -> AgentOutput:
        """
        Agent 核心方法：根据输入生成动作

        流程：
        1. 更新内部状态（历史记录、截图哈希等）
        2. 从指令推断目标 APP（首次调用时）
        3. 检测循环，若检测到则触发突破策略
        4. 构建 CoAT 风格的多轮对话 messages
        5. 调用模型 API
        6. 解析输出为标准 action + parameters
        7. 动作验证与修正
        8. 更新历史记录
        """
        # 保存指令
        self._current_instruction = input_data.instruction

        # 首次调用时从指令推断目标 APP + 生成子目标计划
        if not self._action_history:
            self._inferred_app = self._infer_app_from_instruction(input_data.instruction)
            self._task_plan = self._generate_task_plan(input_data.instruction)
            self._current_plan_step = 0

        # 计算当前截图哈希（用于防循环）
        current_hash = self._compute_image_hash(input_data.current_image)

        # 更新 stuck 计数（截图连续不变次数，用于反思机制）
        if self._last_screenshot_hash and current_hash == self._last_screenshot_hash:
            self._stuck_count += 1
        else:
            self._stuck_count = 0

        # 检测循环
        loop_detected = self._detect_loop(current_hash)

        # 构建 messages
        messages = self._build_messages(input_data, loop_detected)

        # 调用 API（带重试）
        max_retries = 2
        raw_output = ""
        action = ""
        parameters = {}
        usage = None

        for attempt in range(max_retries):
            try:
                response = self._call_api(messages)
                usage = self.extract_usage_info(response)
                raw_output = response.choices[0].message.content

                # 解析输出
                action, parameters = self._parse_action(raw_output)

                if action and action in VALID_ACTIONS:
                    break
                else:
                    logger.warning(
                        f"[Agent] 解析失败 (attempt {attempt + 1}/{max_retries}): "
                        f"raw_output={raw_output[:200]}"
                    )

            except Exception as e:
                logger.error(
                    f"[Agent] API 调用异常 (attempt {attempt + 1}/{max_retries}): {e}"
                )

        # 如果所有重试都失败，返回默认动作
        if not action or action not in VALID_ACTIONS:
            logger.warning("[Agent] 所有重试失败，使用默认动作 SCROLL（向下滚动）")
            action = ACTION_SCROLL
            parameters = self._make_scroll_params("down")

        # ===== 动作验证与修正 =====
        action, parameters = self._validate_and_correct_action(
            action, parameters, input_data, current_hash
        )

        # 更新历史记录
        self._update_history(input_data, action, parameters, raw_output, current_hash)

        # 推进子目标（借鉴 MobileAgent 的 plan tracking）
        self._advance_plan_step(action, parameters)

        return AgentOutput(
            action=action,
            parameters=parameters,
            raw_output=raw_output,
            usage=usage
        )

    # ==========================================
    #  动作验证与修正
    # ==========================================

    def _validate_and_correct_action(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: AgentInput,
        current_hash: str = ""
    ) -> Tuple[str, Dict[str, Any]]:
        """
        动作后置验证：在返回前对动作做合理性校验和修正

        验证项：
        1. CLICK 坐标范围校验 + 边界内缩
        2. OPEN 名称映射
        3. CLICK↔TYPE 摇摆检测 + 动作类型强制修正
        4. 连续 CLICK 检测（疑似无效点击 → 改为 SCROLL）
        """
        # ---- 1. OPEN 名称映射 ----
        if action == ACTION_OPEN:
            app_name = parameters.get("app_name", "")
            if app_name:
                mapped = self._resolve_app_name(app_name)
                if mapped != app_name:
                    logger.info(f"[Agent] APP名称映射: '{app_name}' -> '{mapped}'")
                    parameters["app_name"] = mapped

        # ---- 2. CLICK 坐标校验 ----
        if action == ACTION_CLICK:
            point = parameters.get("point", [])
            if len(point) == 2:
                x, y = point
                x, y = self._clamp_click_coordinates(x, y)
                parameters["point"] = [x, y]

        # ---- 3. CLICK↔TYPE 摇摆检测 ----
        if action in (ACTION_CLICK, ACTION_TYPE):
            last = self._action_history[-1] if self._action_history else None
            if last and last["action"] in (ACTION_CLICK, ACTION_TYPE):
                if last["action"] != action:
                    # CLICK↔TYPE 交替，可能是摇摆
                    # 但若当前 TYPE 的文字和最近一次 TYPE 文字不同，说明是有意义的操作
                    is_meaningful = False
                    if action == ACTION_TYPE:
                        curr_text = parameters.get("text", "")
                        for prev in reversed(self._action_history[:-1]):
                            if prev.get("action") == ACTION_TYPE:
                                prev_text = prev.get("parameters", {}).get("text", "")
                                if curr_text and prev_text and curr_text != prev_text:
                                    is_meaningful = True
                                break
                    if is_meaningful:
                        self._action_toggle_count = 0
                    else:
                        self._action_toggle_count += 1
                else:
                    self._action_toggle_count = 0
            else:
                self._action_toggle_count = 0

            # 连续 4 次 CLICK↔TYPE 摇摆 → 强制使用 SCROLL 尝试突破
            if self._action_toggle_count >= 4:
                logger.warning(
                    f"[Agent] 检测到 CLICK↔TYPE 摇摆 {self._action_toggle_count} 次，"
                    f"切换为 SCROLL（自适应滚动量: {self._scroll_magnitude}）"
                )
                self._action_toggle_count = 0
                return ACTION_SCROLL, self._make_scroll_params("down")

        # ---- 4. 连续 CLICK 检测（疑似无效点击） ----
        if action == ACTION_CLICK:
            if all(
                step["action"] == ACTION_CLICK
                for step in self._action_history[-3:]
            ):
                self._consecutive_click_count += 1
            else:
                self._consecutive_click_count = 0

            # 连续 4+ 次 CLICK 且截图未变（无效点击）→ 改为 SCROLL
            if self._consecutive_click_count >= 3:
                if current_hash == self._last_screenshot_hash and self._action_history:
                    logger.warning(
                        f"[Agent] 连续 {self._consecutive_click_count + 1} 次 CLICK 且截图未变，"
                        f"切换为 SCROLL（向下滚动查看更多内容）"
                    )
                    self._consecutive_click_count = 0
                    return ACTION_SCROLL, self._make_scroll_params("down")
        else:
            self._consecutive_click_count = 0

        return action, parameters

    def _make_scroll_params(self, direction: str = "down") -> Dict[str, Any]:
        """
        生成滚动参数（自适应滚动量，借鉴 AppAgent）

        Args:
            direction: "down"=向下(看更多), "up"=向上(回之前)

        滚动量会根据之前的滚动效果自适应调整：
        - 滚动后页面未变 → 减小幅度（可能到边界了）
        - 滚动后页面变化 → 逐渐恢复默认量
        """
        center_x = 500
        mag = self._scroll_magnitude  # 默认 400，自适应调整

        if direction == "down":
            return {
                "start_point": [center_x, 500 + mag // 2],
                "end_point": [center_x, 500 - mag // 2]
            }
        else:  # up
            return {
                "start_point": [center_x, 500 - mag // 2],
                "end_point": [center_x, 500 + mag // 2]
            }

    def _clamp_click_coordinates(self, x: int, y: int) -> Tuple[int, int]:
        """
        坐标范围校验 + 安全边界内缩

        策略：
        - 超出 [0, 1000] 的值裁剪到边界
        - 边界附近（<5 或 >995）内缩 10 个单位，避免点在屏幕边缘被系统判定无效
        """
        x = max(0, min(1000, x))
        y = max(0, min(1000, y))

        # 边界安全内缩
        MARGIN = 10
        if x < 5:
            x = MARGIN
        elif x > 995:
            x = 1000 - MARGIN
        if y < 5:
            y = MARGIN
        elif y > 995:
            y = 1000 - MARGIN

        return int(x), int(y)

    # ==========================================
    #  APP 名称映射
    # ==========================================

    def _resolve_app_name(self, name: str) -> str:
        """
        解析 APP 名称：别名映射 + 指令关键词推断

        优先级：
        1. 精确匹配别名表
        2. 包含匹配（如 "美团外卖" 包含 "美团"）
        3. 指令关键词推断
        4. 返回原名
        """
        # 1. 精确匹配
        name_lower = name.lower().strip()
        if name_lower in APP_NAME_ALIASES:
            return APP_NAME_ALIASES[name_lower]

        # 2. 别名表中的值匹配（反向查找）
        for alias, standard in APP_NAME_ALIASES.items():
            if name_lower == standard.lower() or name_lower == alias.lower():
                return standard

        # 3. 包含匹配（如 "美团外卖" 包含 "美团"）
        for keyword, standard in APP_KEYWORD_MAP.items():
            if keyword.lower() in name_lower or name_lower in keyword.lower():
                return standard

        # 4. 使用指令推断的 APP 名（如果已确定）
        if self._inferred_app and self._inferred_app != name:
            # 检查是否可能是同一个 APP 的不同表述
            for kw in [self._inferred_app, name]:
                if kw.lower() in name.lower() or name.lower() in kw.lower():
                    return self._inferred_app

        return name

    def _infer_app_from_instruction(self, instruction: str) -> str:
        """从用户指令中推断目标 APP 名称"""
        for keyword, app_name in sorted(
            APP_KEYWORD_MAP.items(), key=lambda x: -len(x[0])
        ):
            if keyword in instruction:
                return app_name
        return ""

    # ==========================================
    #  消息构建
    # ==========================================

    def _build_messages(
        self,
        input_data: AgentInput,
        loop_detected: bool
    ) -> List[Dict[str, Any]]:
        """
        构建 CoAT 风格的多轮对话 messages

        结构：
        [system] 角色定义 + 操作规范 + 空间参考
        [user]    任务指令
        [assistant] 历史操作（多轮）
        [user]    当前截图 + 上下文提示（循环警告/错误恢复/APP名称提示）
        """
        messages = []

        # 1. System Prompt
        system_prompt = self._build_system_prompt(input_data.instruction)
        messages.append({"role": "system", "content": system_prompt})

        # 2. 任务指令
        instruction_text = f"请帮我完成以下任务：\n{input_data.instruction}"
        # 首步时附加 APP 名称提示
        if not self._action_history and self._inferred_app:
            instruction_text += (
                f"\n\n[提示] 请打开「{self._inferred_app}」应用来完成此任务。"
                f"注意：打开应用时请使用完全准确的名称「{self._inferred_app}」，"
                f"不要使用类似「{self._inferred_app}外卖」「{self._inferred_app}优选」等子应用名。"
            )
        messages.append({"role": "user", "content": instruction_text})

        # 3. 历史操作
        if self._action_history:
            messages.append({
                "role": "assistant",
                "content": "好的，我来一步步完成这个任务。"
            })

            history_summary = self._build_history_summary()
            messages.append({
                "role": "user",
                "content": history_summary
            })

            messages.append({
                "role": "assistant",
                "content": "我继续执行下一步操作。"
            })

        # 4. 当前截图 + 上下文提示（JPEG 编码减小体积）
        image_url = self._encode_image(input_data.current_image, image_format="JPEG")

        current_prompt = self._build_current_prompt(input_data, loop_detected)

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": current_prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        })

        return messages

    def _build_current_prompt(self, input_data: AgentInput, loop_detected: bool) -> str:
        """
        构建当前步的提示文本，包含子目标进度、反思指导、错误恢复提示
        """
        parts = []

        # 子目标进度（借鉴 MobileAgent 的 Plan-Execute 模式）
        if self._task_plan:
            current_idx = min(self._current_plan_step, len(self._task_plan) - 1)
            parts.append(
                f"[任务进度] 当前子目标 ({current_idx + 1}/{len(self._task_plan)}): "
                f"「{self._task_plan[current_idx]}」"
            )
            # 如果还有后续步骤，简要提示下一步方向
            if current_idx < len(self._task_plan) - 1:
                parts.append(
                    f"后续目标: {' → '.join(self._task_plan[current_idx + 1:current_idx + 3])}"
                    + (" ..." if current_idx + 3 < len(self._task_plan) else "")
                )

        # 反思提示（借鉴 SeeClick 的具体失败分析）
        reflection = self._build_reflection_prompt()
        if reflection:
            parts.append(reflection)

        # 循环警告
        if loop_detected:
            parts.append(
                "[警告] 检测到操作可能陷入循环。请回顾历史操作，换一种策略！"
            )

        # CLICK↔TYPE 摇摆提示
        if self._action_toggle_count >= 2:
            parts.append(
                "[注意] 你在「点击」和「输入」之间反复切换但可能没有取得进展。"
                "请确认当前页面的实际状态——如果搜索框已经激活（光标在闪烁），"
                "请直接输入文字；如果搜索框未激活，请先点击搜索框。"
            )

        # 连续无效点击提示
        if self._consecutive_click_count >= 1 and self._action_history:
            parts.append(
                "[注意] 你连续多次点击了不同位置但页面似乎没有变化。"
                "请仔细观察截图中的文字内容，确认目标元素的位置后再点击。"
            )

        parts.append("请观察当前截图，思考并执行下一步操作。")
        return "\n\n".join(parts)

    def _build_system_prompt(self, instruction: str) -> str:
        """
        构建 System Prompt — 增强空间感知版

        核心改进：
        - 添加屏幕九宫格坐标参考，帮助模型精确定位
        - 添加常见 UI 元素区域提示
        - 强化坐标输出指导
        - 添加操作顺序规则
        """
        return """你是一个手机操作助手，通过观察截图决定下一步操作。

## 任务
{instruction}

## 思考与输出格式
每一步必须按以下格式输出：

### 观察与分析
当前页面和目标元素的位置（用下方坐标估算）

### 思考与决策
当前状态、目标差距、下一步计划

### 操作执行
**点击**：CLICK:[x, y]
**输入**：TYPE:['文字']
**滚动**：SCROLL:[[起x, 起y], [终x, 终y]]
**打开应用**：OPEN:['应用名']
**完成**：COMPLETE:[]

## 坐标规则（0~1000归一化）
左上(0,0) 右下(1000,1000)，手机竖屏(y方向更长)

屏幕分区参考：
- 状态栏 y:0~30 | 顶部导航/搜索框 y:30~120
- Tab栏/分类 y:120~200 | 内容区 y:200~850
- 底部导航栏("首页""搜索""我的"等) y:900~1000
- 水平：左 0~333 | 中 333~667 | 右 667~1000

## 操作规范
1. 点击坐标指向元素中心，不点边缘
2. 先CLICK搜索框激活，再TYPE输入文字
3. 向下滚动(看更多)：start_y > end_y，如 [[500,700],[500,300]]
4. 向上滚动(回之前)：start_y < end_y，如 [[500,300],[500,700]]
5. 任务完成输出 COMPLETE:[]
6. 先处理弹窗，再继续主任务
7. 一步一操作，严格按格式输出""".format(instruction=instruction)

    def _build_history_summary(self) -> str:
        """
        构建历史轨迹摘要

        改进：
        - 最近 5 步完整显示（原 3 步），增加连贯性
        - 增加操作类型标注，帮助模型理解动作意图
        - 添加操作结果提示（成功/失败）
        """
        if not self._action_history:
            return ""

        history = self._action_history
        total_steps = len(history)

        summary_parts = []

        if total_steps <= 5:
            for i, step in enumerate(history, 1):
                summary_parts.append(
                    f"步骤{i}: {self._format_step_description(step)}"
                )
        else:
            # 早期步骤压缩为摘要
            early_steps = history[:-5]
            action_count = {}
            for step in early_steps:
                act = step.get("action", "UNKNOWN")
                action_count[act] = action_count.get(act, 0) + 1

            early_summary = "、".join(
                [f"{act} {count}次" for act, count in action_count.items()]
            )
            summary_parts.append(
                f"步骤1-{len(early_steps)}（已完成）：{early_summary}"
            )

            # 最近 5 步完整显示
            for i, step in enumerate(history[-5:], total_steps - 4):
                summary_parts.append(
                    f"步骤{i}: {self._format_step_description(step)}"
                )

        return "以下是已执行的操作历史：\n" + "\n".join(summary_parts)

    def _format_step_description(self, step: Dict[str, Any]) -> str:
        """格式化单步操作描述（借鉴 CoAT 原论文，附带操作效果）"""
        action = step.get("action", "")
        params = step.get("parameters", {})
        thought = step.get("thought", "")
        changed = step.get("page_changed", True)  # 页面是否发生了变化

        action_desc = {
            ACTION_CLICK: f"点击位置 ({params.get('point', '?')})",
            ACTION_TYPE: f"输入文字 \"{params.get('text', '?')}\"",
            ACTION_SCROLL: f"从 ({params.get('start_point', '?')}) 滚动到 ({params.get('end_point', '?')})",
            ACTION_OPEN: f"打开应用 \"{params.get('app_name', '?')}\"",
            ACTION_COMPLETE: "任务完成",
        }.get(action, f"执行 {action}")

        # 附带操作效果（借鉴 MobileAgent 的屏幕变化感知）
        effect = "" if changed else "（页面未变化）"
        if thought:
            return f"[思考] {thought}\n[操作] {action_desc}{effect}"
        return f"{action_desc}{effect}"

    # ==========================================
    #  输出解析
    # ==========================================

    def _parse_action(self, raw_output: str) -> Tuple[str, Dict[str, Any]]:
        """
        从模型输出中解析 action 和 parameters

        支持的输出格式：
        - CLICK:[x, y]
        - CLICK:[[x, y]]
        - TYPE:['内容']
        - TYPE:["内容"]
        - SCROLL:[[x1, y1], [x2, y2]]
        - OPEN:['应用名']
        - COMPLETE:[]
        - 以及文字描述格式中的操作行
        """
        if not raw_output:
            return "", {}

        # 提取思考内容（用于历史记录）
        thought = self._extract_thought(raw_output)

        # 尝试多种格式解析
        action, parameters = self._try_parse_formats(raw_output)

        # 存储思考内容
        if action:
            self._last_thought = thought

        return action, parameters

    def _extract_thought(self, raw_output: str) -> str:
        """从模型输出中提取思考内容"""
        patterns = [
            r"###\s*观察与分析\s*\n(.*?)(?=###|$)",
            r"###\s*思考与决策\s*\n(.*?)(?=###|$)",
        ]

        thoughts = []
        for pattern in patterns:
            match = re.search(pattern, raw_output, re.DOTALL)
            if match:
                content = match.group(1).strip()
                content = re.sub(r"^\d+\.\s*", "", content, flags=re.MULTILINE)
                if content:
                    thoughts.append(content[:200])

        return "；".join(thoughts) if thoughts else ""

    def _try_parse_formats(self, raw_output: str) -> Tuple[str, Dict[str, Any]]:
        """尝试多种格式解析"""

        # 格式1: ACTION:[params] 格式（标准格式）
        action, params = self._parse_standard_format(raw_output)
        if action:
            return action, params

        # 格式2: Action: ACTION(...) 格式
        action, params = self._parse_function_format(raw_output)
        if action:
            return action, params

        # 格式3: 从"操作执行"部分提取
        action, params = self._parse_section_format(raw_output)
        if action:
            return action, params

        # 格式4: 关键字匹配（兜底）
        action, params = self._parse_keyword_format(raw_output)
        if action:
            return action, params

        return "", {}

    def _parse_standard_format(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        解析标准格式: ACTION:[params]
        如 CLICK:[500, 300], TYPE:['内容'], COMPLETE:[]
        """
        patterns = {
            ACTION_CLICK: r"CLICK\s*:\s*\[?\s*(\d+)\s*,\s*(\d+)\s*\]?",
            ACTION_TYPE: r"TYPE\s*:\s*\[?\s*['\"]?([^'\]\"\]]+)['\"]?\s*\]?",
            ACTION_SCROLL: r"SCROLL\s*:\s*\[?\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*,\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*\]?",
            ACTION_OPEN: r"OPEN\s*:\s*\[?\s*['\"「]([^'\"」\]]+)['\"」]?\s*\]?",
            ACTION_COMPLETE: r"COMPLETE\s*:\s*\[\s*\]",
        }

        for action_type, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                if action_type == ACTION_CLICK:
                    x, y = int(match.group(1)), int(match.group(2))
                    return ACTION_CLICK, {"point": [x, y]}
                elif action_type == ACTION_TYPE:
                    text_val = match.group(1).strip()
                    # 排除纯空白或仅含括号的匹配
                    if text_val:
                        return ACTION_TYPE, {"text": text_val}
                elif action_type == ACTION_SCROLL:
                    return ACTION_SCROLL, {
                        "start_point": [int(match.group(1)), int(match.group(2))],
                        "end_point": [int(match.group(3)), int(match.group(4))]
                    }
                elif action_type == ACTION_OPEN:
                    return ACTION_OPEN, {"app_name": match.group(1).strip()}
                elif action_type == ACTION_COMPLETE:
                    return ACTION_COMPLETE, {}

        return "", {}

    def _parse_function_format(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        解析函数格式: Action: click(point='x y') 或 click(x, y)
        """
        patterns = {
            ACTION_CLICK: r"click\s*\(\s*(?:point\s*=\s*['\"]?)?\s*(\d+)\s*,\s*(\d+)\s*['\"]?\s*\)",
            ACTION_TYPE: r"type\s*\(\s*(?:content\s*=\s*)?['\"]([^'\"]+)['\"]\s*\)",
            ACTION_SCROLL: r"scroll\s*\(\s*(?:start_point\s*=\s*['\"]?)?\s*(\d+)\s*,\s*(\d+)\s*['\"]?\s*,\s*(?:end_point\s*=\s*['\"]?)?\s*(\d+)\s*,\s*(\d+)\s*['\"]?\s*\)",
            ACTION_OPEN: r"open\s*\(\s*(?:app_name\s*=\s*)?['\"]([^'\"]+)['\"]\s*\)",
            ACTION_COMPLETE: r"complete\s*\(\s*(?:content\s*=\s*)?['\"]?[^'\"]*['\"]?\s*\)",
        }

        for action_type, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if action_type == ACTION_CLICK:
                    x, y = int(match.group(1)), int(match.group(2))
                    return ACTION_CLICK, {"point": [x, y]}
                elif action_type == ACTION_TYPE:
                    return ACTION_TYPE, {"text": match.group(1)}
                elif action_type == ACTION_SCROLL:
                    return ACTION_SCROLL, {
                        "start_point": [int(match.group(1)), int(match.group(2))],
                        "end_point": [int(match.group(3)), int(match.group(4))]
                    }
                elif action_type == ACTION_OPEN:
                    return ACTION_OPEN, {"app_name": match.group(1)}
                elif action_type == ACTION_COMPLETE:
                    return ACTION_COMPLETE, {}

        return "", {}

    def _parse_section_format(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """从"操作执行"部分提取操作"""
        section_match = re.search(
            r"###\s*操作执行\s*\n(.*?)(?=###|$)",
            text, re.DOTALL
        )
        if section_match:
            section_text = section_match.group(1).strip()
            action, params = self._parse_standard_format(section_text)
            if action:
                return action, params
            action, params = self._parse_function_format(section_text)
            if action:
                return action, params

        return "", {}

    def _parse_keyword_format(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        关键字匹配兜底：从文本描述中提取操作意图
        """
        # COMPLETE 检测（仅在操作执行部分或明确声明任务完成时才触发）
        if re.search(r"(?:任务|目标)?(?:已经)?完成(?:了吗)?[。，！]?$|COMPLETE:\[\]", text, re.MULTILINE):
            # 排除描述性的"已完成X步"等中间状态
            if not re.search(r"已完成了?\d+步|前\d+步.*完成", text):
                return ACTION_COMPLETE, {}

        # OPEN 检测
        open_match = re.search(r"打开(?:应用)?[：:\"\s]*[「\[]?([^」\]\"\n]+)[」\]]?", text)
        if open_match:
            return ACTION_OPEN, {"app_name": open_match.group(1).strip()}

        # TYPE 检测
        type_match = re.search(r"(?:输入|填写)[：:\"\s]*[「\[]?([^」\]\"\n]+)[」\]]?", text)
        if type_match:
            return ACTION_TYPE, {"text": type_match.group(1).strip()}

        return "", {}

    # ==========================================
    #  防循环检测
    # ==========================================

    def _compute_image_hash(self, image) -> str:
        """计算截图的 MD5 哈希（用于变化检测）"""
        import io
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return hashlib.md5(buffered.getvalue()).hexdigest()

    def _detect_loop(self, current_hash: str) -> bool:
        """
        检测是否陷入循环

        判断条件（满足任一即触发）：
        1. 连续 3 步相同动作 + 相同参数（_consecutive_same_action >= 2）
        2. 连续 2 步截图完全相同（说明操作无效）
        """
        if not self._action_history:
            return False

        # 条件1：连续相同动作
        if len(self._action_history) >= 2:
            last = self._action_history[-1]
            prev = self._action_history[-2]

            if (last["action"] == prev["action"] and
                    last["parameters"] == prev["parameters"] and
                    last["action"] != ACTION_COMPLETE):
                self._consecutive_same_action += 1
            else:
                self._consecutive_same_action = 0

            # 修复：_consecutive_same_action 表示"除了 last 之外还有几步相同"
            # =1 → last 和 prev 相同（连续2步）
            # =2 → 连续3步相同 → 触发
            if self._consecutive_same_action >= 2:
                logger.warning(
                    f"[Agent] 检测到循环！连续 {self._consecutive_same_action + 1} 步相同动作"
                )
                self._consecutive_same_action = 0
                return True

        # 条件2：截图未变化（复用 stuck_count，避免双重计数）
        if self._last_screenshot_hash and self._last_screenshot_hash == current_hash:
            if self._stuck_count >= 1:
                # stuck_count 在 act() 开头已更新，这里 >= 1 表示连续 2 次截图不变
                logger.warning("[Agent] 检测到循环！连续截图未变化")
                return True
        else:
            pass  # stuck_count 已在 act() 开头正确重置

        return False

    # ==========================================
    #  历史管理
    # ==========================================

    def _update_history(
        self,
        input_data: AgentInput,
        action: str,
        parameters: Dict[str, Any],
        raw_output: str,
        screenshot_hash: str
    ):
        """更新历史记录"""
        # 首步时 _last_screenshot_hash 为空，无法判断页面变化，默认 True
        if self._last_screenshot_hash:
            page_changed = (screenshot_hash != self._last_screenshot_hash)
        else:
            page_changed = True  # 首步无法判断，标记为变化（保守策略）

        step_record = {
            "step": len(self._action_history) + 1,
            "action": action,
            "parameters": parameters,
            "thought": getattr(self, '_last_thought', ''),
            "screenshot_hash": screenshot_hash,
            "page_changed": page_changed,  # 借鉴 MobileAgent：记录操作效果
        }

        self._action_history.append(step_record)
        self._last_screenshot_hash = screenshot_hash
        # 保存当前截图用于下次差异对比
        self._last_image = input_data.current_image

        # 自适应滚动量（借鉴 AppAgent）
        if action == ACTION_SCROLL:
            if not page_changed:
                # 滚动后页面未变，可能到达边界，减小滚动量
                self._scroll_magnitude = max(200, self._scroll_magnitude - 100)
            elif self._scroll_magnitude < 400:
                # 滚动有效，逐渐恢复默认量
                self._scroll_magnitude = min(400, self._scroll_magnitude + 50)

    # ==========================================
    #  子目标分解（借鉴 MobileAgent Plan-Execute）
    # ==========================================

    def _generate_task_plan(self, instruction: str) -> List[str]:
        """
        将任务指令分解为子目标计划

        使用轻量级 prompt 让模型快速生成步骤列表。
        如果生成失败，返回空列表（退化为无计划模式）。
        """
        plan_prompt = (
            "你是一个任务规划专家。请将以下手机操作任务分解为简洁的步骤（3-8步），"
            "每步用一行描述，不要编号，不要多余解释。\n\n"
            f"任务：{instruction}\n\n"
            "请直接输出步骤列表："
        )

        try:
            messages = [
                {"role": "system", "content": "你是一个任务分解助手，输出简洁的步骤列表。"},
                {"role": "user", "content": plan_prompt}
            ]
            response = self._call_api(messages)
            content = response.choices[0].message.content.strip()

            # 解析步骤列表
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            # 清理编号前缀（如 "1." "Step 1:" 等）
            cleaned = []
            for line in lines:
                cleaned_line = re.sub(r"^(?:\d+[\.\、\)]\s*|step\s*\d+[\.\：:]\s*)", "", line, flags=re.IGNORECASE)
                cleaned_line = cleaned_line.strip("•-* ")
                if len(cleaned_line) > 2 and len(cleaned_line) < 50:
                    cleaned.append(cleaned_line)

            if len(cleaned) >= 2:
                logger.info(f"[Agent] 生成子目标计划（{len(cleaned)}步）: {cleaned}")
                return cleaned
        except Exception as e:
            logger.warning(f"[Agent] 子目标计划生成失败: {e}")

        return []

    def _advance_plan_step(self, action: str, parameters: Dict[str, Any]):
        """
        根据当前执行的动作自动推进子目标

        策略：当一个 TYPE/OPEN 动作或成功的 CLICK（截图变化）发生后，
        自动将 plan_step 推进一步。
        """
        if not self._task_plan or self._current_plan_step >= len(self._task_plan) - 1:
            return

        # OPEN 动作 → 推进一步（打开APP后进入下一步）
        if action == ACTION_OPEN:
            self._current_plan_step = min(self._current_plan_step + 1, len(self._task_plan) - 1)
            return

        # TYPE 动作 → 推进一步（输入搜索词后进入下一步）
        if action == ACTION_TYPE:
            self._current_plan_step = min(self._current_plan_step + 1, len(self._task_plan) - 1)

    # ==========================================
    #  反思机制（借鉴 SeeClick/AppAgent）
    # ==========================================

    def _build_reflection_prompt(self) -> Optional[str]:
        """
        构建反思提示：当操作无效时，注入具体的失败分析

        借鉴 SeeClick 的 self-correction 和 AppAgent 的经验回放：
        不是泛泛地提醒"换个策略"，而是具体分析失败原因。
        """
        if not self._action_history or self._stuck_count < 1:
            return None

        last_action = self._action_history[-1]
        last_act = last_action.get("action", "")
        last_params = last_action.get("parameters", {})

        # 构建失败分析
        analysis_parts = []

        if self._stuck_count >= 2:
            # 连续 2+ 次截图未变 → 具体分析
            if last_act == ACTION_CLICK:
                point = last_params.get("point", [])
                analysis_parts.append(
                    f"你连续 {self._stuck_count + 1} 次操作后页面没有变化。"
                    f"上次点击位置是 {point}，该位置可能不是可交互元素，"
                    f"或者目标元素在当前视口之外。"
                )
                analysis_parts.append(
                    "建议策略：1) 仔细阅读截图中的文字内容，寻找目标元素 2) "
                    "尝试向下/向上滚动查看更多内容 3) 如果在搜索框中操作，"
                    "确认搜索框已激活（光标闪烁）再输入"
                )
            elif last_act == ACTION_SCROLL:
                analysis_parts.append(
                    f"你连续滚动 {self._stuck_count + 1} 次但页面没有变化，"
                    f"可能已到达页面顶部或底部。"
                )
                analysis_parts.append(
                    "建议策略：1) 尝试反向滚动 2) 停止滚动，仔细查看当前页面内容 "
                    "3) 可能目标元素就在当前页面中，需要点击而非继续滚动"
                )
            elif last_act == ACTION_TYPE:
                analysis_parts.append(
                    "你输入了文字但页面没有变化，可能搜索框未激活或输入内容有问题。"
                )
                analysis_parts.append(
                    "建议策略：1) 先点击搜索框使其获得焦点 2) 检查是否需要点击搜索按钮"
                )

            return "[操作失败分析] " + " ".join(analysis_parts)

        return None
