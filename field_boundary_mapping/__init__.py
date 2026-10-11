"""field_boundary_mapping — 农田边界语义理解 Python 包

子包:
    perception  提案源与感知 (SAM3 本地推理 / 云 API VLM / 本地 VLM)
    proposals   区域提案流水 (固定提示词提案 + SAM 细化 + 评估)
    scene       场景理解三阶段 (VLM 场景理解 → SAM 区域细化)
    evidence    证据加工 (实例分组 / 冲突审计 / 时序链 / 可见边界)
    viz         可视化与查看器 (轨线浏览器 / 渲染 / 人工校对)

共享工具在包顶层 semantic_pipeline (调色板 / 掩膜轮廓 / C++ 核心加载)。
C++ 核心由 CMake 编译到 lib/field_boundary_core, 经 load_core() 加载。
"""
