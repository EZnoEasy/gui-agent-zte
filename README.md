# GUI Agent — 中兴捧月智能代理比赛

> 基于 VLM 的手机自主操作智能体 | 中兴捧月智能代理比赛

## 项目简介

一个基于多模态大模型（VLM）的 GUI Agent，能够自主操控手机 App 完成复杂的跨应用任务（如电商下单、视频搜索、评论发布等）。

深度融合了 CoAT（Chain-of-Action-Thought）、MobileAgent、SeeClick、AppAgent 等前沿论文的核心思想，在 ~1070 行 Python 代码中实现了一套完整的多层防护与自适应推理系统。

## 核心技术

- **CoAT 行动思维链**：每步操作前强制模型进行「观察分析 → 思考决策 → 操作执行」三段式结构化推理，显著提升决策质量
- **空间感知 Prompt**：内嵌屏幕九宫格坐标参考 + 常见 UI 元素区域提示，增强 VLM 的空间定位能力
- **子目标分解与追踪**：首次调用时自动将复杂任务拆解为 3-8 步子目标，执行中自动推进
- **多层防循环检测**：动作重复检测 + 截图哈希对比，有效避免 Agent 陷入无限循环
- **具体化失败反思**：按 CLICK/SCROLL/TYPE 类别生成针对性自校正建议，非泛化警告
- **自适应动作生成**：滚动量动态调节、坐标边界安全内缩、APP 名称三级映射（精确→别名→指令推断）
- **三级历史记忆压缩**：最近 5 步完整保留 + 早期步骤语义摘要 + 操作效果标记（page_changed）

## 项目结构

```
├── README.md
├── GUI_Agent_调研报告.md          # 前沿论文调研
├── doc/
│   └── 算法设计说明文档.md        # 详细算法设计文档
├── src/
│   ├── agent.py                   # 主 Agent 实现 (~1070 行)
│   ├── agent_base.py              # Agent 基类
│   ├── requirements.txt           # 依赖
│   └── utils/
│       ├── __init__.py
│       └── image_utils.py         # 图像处理工具
└── submission/                    # 最终提交版本
    ├── doc/
    │   └── 算法设计说明文档.md
    └── src/
        ├── agent.py
        ├── agent_base.py
        ├── requirements.txt
        └── utils/
            ├── __init__.py
            └── image_utils.py
```

## 设计灵感

| 论文 / 项目 | 借鉴思路 |
|---|---|
| **CoAT** (复旦 DISC) | 行动思维链范式，结构化推理 |
| **MobileAgent** (阿里) | 子目标分解 (Plan-Execute)，屏幕变化感知 |
| **SeeClick** (南京大学) | 具体化失败反思，自校正机制 |
| **AppAgent** (腾讯) | 自适应探索，滚动量动态调节 |

## 许可证

MIT License
