# field_boundary_mapping — 农田边界语义理解

从农机车载图像中提取农田边界与功能区语义 (作物 / 已收获地 / 裸地 /
水域 / 沟渠 / 车辆 / 行人 / 树木 / 电杆等), 输出**可审计**的掩膜
证据、实例分组与时序一致性链, 为地头转向与边界作业提供语义输入。

核心链路为三阶段:

```
车载图像 ──► 场景理解 (VLM 区域候选提案) ──► 掩膜细化 (SAM3)
        ──► 语义证据核验与几何核心 (C++ / pybind11)
        ──► 实例分组 / 冲突审计 / 时序链 / 可见边界提取 / 浏览器查看器
```

提案源两种可选:
- **云 API** (`cloud_vlm_proposer.py`): OpenAI 兼容接口
  (默认 DashScope, 模型如 `qwen-vl-max-latest`);
- **本地 VLM** (`vlm_proposer.py`): 本地部署的 Qwen3-VL。

## 设计纪律

- **只读原始数据**: 所有产物写入 `runs/<新目录>/`, 从不覆盖
  既有图像、模型响应与掩膜; 每个掩膜记录来源提案与核验状态。
- **模型输出不修补**: 解析失败、字段缺失、坐标越界一律记为
  失败并归档原始响应, 不做语义级重试或事后改写。
- **证据可追溯**: 实例级审计 (重复/冲突/时序一致性) 全部有
  脚本产出报告, 人工复核走 `review_results.py` /
  `annotation_review_tool.py`。

## 目录结构

```
field_boundary_mapping/
├── CMakeLists.txt                 # C++ 核心构建 (pybind11, 零第三方依赖)
├── include/field_boundary/core.hpp  # 语义证据核验与几何核心
├── src/bindings.cpp               # pybind11 绑定 (field_boundary_core)
├── lib/                           # 编译产物 (生成)
├── tests/                         # 491 个自包含单元测试
│
│  三阶段流水
├── scene_understanding.py         # 场景理解: VLM 输出解析与核验
├── scene_staging.py               # 分阶段编排 (理解 → 细化 → 渲染)
├── run_staged_scene.py            # 分阶段运行入口
├── run_scene_understanding.py     # 场景理解批量运行
├── refine_scene_regions.py        # 场景区域 SAM 细化
├── run_proposals.py               # 区域候选提案 + SAM 细化 + 评估
│
│  提案源与感知
├── cloud_vlm_proposer.py          # 云 API 提案源 (OpenAI 兼容)
├── vlm_proposer.py                # 本地 VLM 提案源
├── sam3_local.py                  # 本地 SAM3 推理封装
│
│  证据加工
├── group_instances.py             # 实例分组派生层 (跨提案/跨帧)
├── analyze_instance_conflicts.py  # 实例重复与冲突审计
├── analyze_temporal_consistency.py# 时序一致性评分与链构建
├── extract_visible_boundaries.py  # 单帧可见边界提取 (含闭合层)
├── evaluate_offline.py / fuse_metric.py  # 离线评估与融合指标
│
│  可视化与复核
├── serve_track_viewer.py          # 轨线浏览器查看器
├── serve_sam_viewer.py            # SAM 结果查看器
├── semantic_pipeline.py           # 掩膜/渲染工具与调色板
├── render_track_visualization.py / render_ablation_comparison.py
├── rerender_overlays.py           # 叠加图重渲染
├── review_results.py / annotation_review_tool.py  # 人工复核
├── compare_proposal_runs.py       # 提案配置对比
└── run_offline.py                 # SAM 离线批量推理
```

## 构建与测试

C++ 核心 (Windows, VS Build Tools; Linux 同理):

```powershell
uv sync   # 或自行安装 numpy pybind11
$py  = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$pb  = (Resolve-Path ".\.venv\Lib\site-packages\pybind11\share\cmake\pybind11").Path
cmake -S . -B build-core -G "Visual Studio 18 2026" -A x64 `
    -DPython_EXECUTABLE="$py" -Dpybind11_DIR="$pb"
cmake --build build-core --config Release
# 产物自动同步到 lib/field_boundary_core*.pyd
```

单元测试 (491 个, 自包含, 不依赖 runs/ 数据):

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover tests
```

## 运行

**数据**: 本仓库不含任何图像/视频数据, `runs/` 为本地实验输出
目录 (不入库)。请使用自有授权影像。

**云 API 提案源** (密钥只从环境变量读取, 不落盘):

```powershell
$env:DASHSCOPE_API_KEY = "..."   # 或 --api-key-env 指定其他变量名
.\.venv\Scripts\python.exe run_proposals.py --help
```

**本地 SAM3**: 需自备 SAM3 权重 checkpoint:

```powershell
.\.venv\Scripts\python.exe run_offline.py --weights path\to\sam3.pt --output runs\my_run ...
```

**浏览器查看器** (只读叠加, 不修改 run 目录):

```powershell
.\.venv\Scripts\python.exe serve_track_viewer.py runs\my_run
```

## 语义类别

| 角色 | 类别 |
|---|---|
| 地表 | crop (作物) / ground (裸地·已收获) / woody_vegetation (树木灌木) / water (水域) |
| 危险物 | ditch (沟渠) / person (行人) / vehicle (车辆) / tarp (塑料布) / pole (电杆) |

掩膜渲染统一走 `semantic_pipeline.COLORS`; 多功能重叠像素按角色
混合色渲染, 冲突实例用独立冲突色标出。

## 已知边界

- 语义结果目前只报告**格式合法率与产出量**; 精度评估需先建立
  人工标注真值, 真值建立前不引用任何准确率数字。
- 云 API 调用为顺序串行; 网络/限流失败按失败记录, 不自动重试。
- SAM3 细化依赖本地权重与 GPU (CPU 可跑但约 76 s/帧)。
