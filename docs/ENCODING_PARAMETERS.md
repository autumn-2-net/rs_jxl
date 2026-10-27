# 编码参数指南

本页按当前 `rs_jxl` Python 绑定、Rust 参数类型和实际映射整理，不是
`cjxl` 参数全集，也不是旧 C++ `jxlpy` 包的说明。这里的编码均为无损编码；
参数改变工具、搜索成本和输出体积，不改变像素无损的验收要求。
同名 effort 不承诺与官方或优化版 C++ 的体积、速度相同。

输入格式、NumPy/Torch、文件与内存接口、安装和 wheel 见
[Python API](PYTHON_API.md)。显式 Rust 工具见 [Rust Pixel API](PIXEL_API.md)。

## 1. 先选保留对象

| 目的 | Python 入口 | 实际保留内容 |
| --- | --- | --- |
| PNG/数组的像素无损 | `encode(src, ...)` | 解码后的全部样本，包括 alpha=0 下的 RGB；不保证还原原 PNG 文件字节 |
| 原 JPEG 文件归档 | `transcode_jpeg(src, ...)` | JPEG 量化系数及重建信息；`reconstruct_jpeg` 可恢复原文件字节 |
| JPEG 解码后像素无损 | `encode(src, lossless_jpeg=False, ...)` | 输入 JPEG 解码器产生的像素；不能恢复原 JPEG 文件 |
| 等尺寸完整状态序列 | `encode_multiframe(frames, ...)` | 每张完整输入状态；可由 Rust 自动构造 crop/mask/多矩形层 |
| 异尺寸图片或显式层 | `encode_layers(images, ...)` | 原始层样本；画布位置、混合和引用由调用者指定 |
| 只分析多帧复用潜力 | `analyze_multiframe(frames, ...)` | 统计与几何报告，不产生 JXL，也不预测实际压缩率 |

Python `encode` 遇到 JPEG 路径或编码字节默认走 **JPEG 字节转码**，不是先解码成像素。
如果已经传入解码后的数组，则只能做像素无损。多帧和 layers 接口里的 JPEG 输入
也会解码成像素，不能把几个原始 JPEG 的重建数据一起归档。

CLI 有一个重要区别：`rs-jxl encode input.jpg ...` 是像素路径；JPEG 字节归档必须用
`rs-jxl transcode-jpeg ...`。不要拿这两种保留目标的体积当同一种实验结果。

## 2. Python 通用参数与默认值

| 参数 | 默认值 / 范围 | 含义 |
| --- | --- | --- |
| `effort` | `encode` / `encode_multiframe` / `encode_layers` 为 **7**；`transcode_jpeg` 为 **9**；整数 `1..11` | 后端策略等级，不是有损质量等级；更高不保证更小或按固定比例变慢 |
| `threads` | **1**；整数 `0..256` | 1 为调用线程串行；0 在 Python 中解析为 CPU 数量，最多 256；Rust 原始接口要求 `1..256` |
| `lossless` | **True** | 像素接口不提供有损编码；`False` 报错 |
| `distance` | **0** | 非零报错；没有 `distance=1` 等有损模式 |
| `lossless_jpeg` | **True**，仅 `encode` | 控制 JPEG 编码输入是字节转码还是先解码再像素无损 |
| `source_search` | **False**，用于单图或 JPEG 转码 | 额外的候选调度搜索；不等于多帧 `strategy="auto"` |
| `encoder_options` | **None** | 像素工具的字典，键名用下节的 Rust 名称；不是任意透传通道 |
| `return_info` | **False** | True 返回 `(output_result, metadata)`；不同入口的报告结构不同 |
| `strip_metadata` | **False**，像素路径 | 允许丢弃不支持的输入附加元数据；支持的色彩元数据仍保留，不是无条件删去 ICC |
| `borrow_inputs` | **False**，像素/多帧/layers 接口 | 可选借用适合的连续缓冲区；不是端到端零复制，期间调用者不得修改输入及别名 |

`output=None` 返回编码字节；指定路径写文件并返回 `Path`；指定可写二进制流写入后
返回 `None`。这些 Python 编码接口先完成编码再写输出，不是真正逐帧流式编码。
原生流接口另见 [Streaming Output](STREAMING_OUTPUT.md)。

重复处理文件可使用 `Encoder(threads=N)` 复用原生线程池。该 context 的线程数固定，
每次调用的 effort 等选项不会继承上次值。外层并发处理多张图时，避免
“文件 worker 数量 × 每次编码线程数”过度占用 CPU 和内存。

## 3. effort 与搜索不是同一个开关

### 固定 source 参数

```python
import rs_jxl

data, meta = rs_jxl.encode(
    "input.png", effort=9, threads=4,
    source_search=False, return_info=True,
)
print(meta["report"])
```

这里执行一个 source-policy 候选，但候选内部仍可以搜索 RCT、加权预测器参数、
MA 树、调色板和熵模型。**关闭 `source_search` 并不等于后端没有搜索。**

Python 把 effort 映射为 `CandidateParams.speed_tier = 10 - effort`，再投影到
`EncodeOptions`。原始 `EncodeOptions` 本身没有 effort 字段，也没有统一的
“默认 e7”；不能把它的手动工具默认值当成 Python e7 的配置。

| effort | 固定 source 后端的要点 |
| --- | --- |
| 1 | 先进入 tier 9。符合特定输入/参数条件时可走独立 FJXL 快速路线；否则 Modular 回退到 tier 8。不是“所有 e1 都是 FJXL” |
| 2 | Modular 固定 Gradient 树 |
| 3 | Modular 固定 Weighted 树 |
| 4..9 | 默认预测器为 Best；随等级改变学习、颜色变换、熵/LZ77 等策略 |
| 10 | tier 0，默认 Variable 预测器；通常比 e9 承担更多学习和整帧工作 |
| 11 | tier -1，独立的 Tectonic 参数策略；不能当作 e10 的所有设置简单加倍 |

低 effort 的固定树不承诺采用显式 `predictor` 指定的逐像素预测器。
FJXL 也有自己的固定工具和内部 effort，不能用 Modular 参数表解释其全部行为。
确认实际路线应看 `meta["report"]["implementation"]`，其值为 `modular` 或
`fast_lossless`；报告里的 `effective_parameters` 不是每个最终底层工具的展开列表。

### 额外 source 搜索

```python
data, meta = rs_jxl.encode(
    "input.png", effort=9, threads=4,
    source_search=True, return_info=True,
)
```

- effort 1..10 进入额外的像素候选调度；它会分析/试编码，并保留实际获胜输出。
- 该普通像素搜索要求主图和 extra channels 使用最多 16 位整数，并且存储类型为
  U8/U16；U32 存储或浮点不能因为单候选能编码，就推断此搜索也接受。
- effort 11 + `source_search=True` 是单独的帧候选 tournament，资格条件和工作量
  不同于上述普通整数搜索。其完整编码次数可达 23 或 27，不能默认用于批量任务。
- 显式 `group_size_shift`、`rct`、`predictor`、`palette_colors`、`patches`、
  `learning_fraction` 与 `source_search=True` 冲突，即使显式值恰好等于默认值。
  `ma_max_leaves` 是额外资源上限，不属于这组互斥工具覆盖项。
- Python 当前只暴露普通 Auto 搜索；Rust/CLI 的 Scout、Pilot、progressive-group
  实验不是 `source_search=True` 自动启用的能力。

搜索不是穷举所有合法码流，也不保证小于 PNG、每个更低 effort 或独立压缩之和。
体积收益与付出的编码时间都需要测量。

## 4. Python 像素工具参数

下面的名称可直接作为 `encode`、`encode_multiframe`、`encode_layers` 的关键字，
也可以放入 `encoder_options`。多帧调用中是共用的后端配置，不是逐帧参数列表。

| Rust 名称 | source 参数默认值 / 接受范围 | 作用与注意事项 |
| --- | --- | --- |
| `group_size_shift` | 省略为自动；整数 `0..3` | Modular 分组边长分别为 128/256/512/1024；不是线程数。较大组减少部分头开销，也可能增大学习成本和工作集 |
| `rct` | 省略为自动；整数 `0..41` | 可逆颜色变换编号，0 为恒等变换、6 为常用 YCoCg 路线。省略不是 0；灰度等布局仍受工具适用条件约束 |
| `predictor` | 省略时按 effort 解析；整数 `0..15` | 0..13 是具体预测器，14/15 是搜索模式；不是跨帧预测开关 |
| `palette_colors` | **1024**；整数 `0..70911` | 多通道 palette 候选的颜色数上限，实际尝试还受内容/成本门控。不是量化，超过预算不会合并近似颜色 |
| `patches` | 省略为自动；布尔 **True/False** | False 关闭 source patch；True 请求该路线，但仍受适用性及实际发现结果约束 |
| `learning_fraction` | **0.5**；有限、非负的 f32 数值 | MA 学习基准比例，之后按 effort 缩放并截到 1；不是运行轮数或时间上限 |
| `ma_max_leaves` | 省略无额外人为上限；整数 `1..2097152` | 额外的已学习 MA 树叶数资源上限；还受码流/解码器及几何限制约束，不是推荐的提速质量档位 |

### 名称别名

别名只在**关键字参数**中转换；`encoder_options` 内请使用上表规范名称。

| Python 关键字别名 | 转换 |
| --- | --- |
| `modular_group_size` | `group_size_shift` |
| `modular_colorspace` | `rct` |
| `modular_predictor` | `predictor` |
| `modular_palette_colors` | `palette_colors` |
| `iterations` | `learning_fraction = value / 100` |
| `modular_ma_tree_learning_percent` | `learning_fraction = value / 100` |

例如 `iterations=100` 对应 `learning_fraction=1.0`，**不是编码 100 遍**。
不能同时传别名和对应规范名，也不能在字典和关键字中重复同一工具。
`effort`、`threads`、`source_search`、`strip_metadata` 必须使用各自显式参数。

### 预测器编号

| 编号 | 名称 | 编号 | 名称 |
| --- | --- | --- | --- |
| 0 | Zero | 8 | TopLeft |
| 1 | Left | 9 | LeftLeft |
| 2 | Top | 10 | Average1 |
| 3 | Average0 | 11 | Average2 |
| 4 | Select | 12 | Average3 |
| 5 | Gradient | 13 | Average4 |
| 6 | Weighted | 14 | Best，Weighted/Gradient 搜索 |
| 7 | TopRight | 15 | Variable，多预测器搜索 |

Zero 表示预测值为零，不是丢掉图像或不使用熵编码。它与调色板、LZ77 组合可能适合
部分截图，但不代表天然适合所有低色数内容。具体算式见
[predictor.rs](../crates/rs-jxl/src/modular/predictor.rs)。

### 默认值里容易误读的地方

- source 自动 group shift：宽高都不超过 128 时为 0，都不超过 256 时为 1，
  都不超过 400 时为 2，其余为 1。改变分组还可能改变整帧/分区路线，不仅是头大小。
- `learning_fraction` 接受大于 1 的有限值，但有效学习比例会缩放/饱和；不是无限采样。
  正比例还存在每通道约 1024 个期望样本的下限，小图可能全采样。
  设为 0 是零学习采样，不是关闭全部预测、树表示或其他分析。
- `palette_colors=0` 关闭这一项控制的多通道 palette 尝试，**不关闭所有单通道
  palette/通道值紧缩**。source 策略另有前后变换通道百分比，Python 尚未暴露。
- 当前 source patch 门控包括主图位深最多 16、effort 至少 5、非 groupwise 路线等。
  `patches=True` 会影响整帧路线选择，但不会绕过所有工具限制，也不保证检出重复块。
  只有 atlas、字典、残差合计比不使用 patch 的输出更小，才有体积收益；强制开并不承诺获益。
- `patches` 检测当前编码图像内可复用的组件，不是自动寻找上一帧运动或差分的开关。
  帧间复用由第 7 节的原生多帧规划负责。
- `ma_max_leaves` 在 source 路线达到上限时可能报错，不能用它假装“更快但同质量”。
  CLI/Rust 另有 strict/bounded 模式；Python 当前没有 `ma_budget_mode` 选项。

## 5. 截图参数的实际写法

此前讨论的 `cjxl` 配置：

```text
-d 0 -e 9 -g 3 -I 100 -P 0 --modular_palette_colors=10000 --patches 0 -Y 0
```

这些已暴露的控制在 Python 中可表达为：

```python
data, meta = rs_jxl.encode(
    "content.png", effort=9, threads=4,
    source_search=False,
    encoder_options={
        "group_size_shift": 3,
        "learning_fraction": 1.0,
        "predictor": 0,
        "palette_colors": 10000,
        "patches": False,
        "rct": 0,
    },
    return_info=True,
)
```

也可使用 `modular_group_size=3, iterations=100, modular_predictor=0,
modular_palette_colors=10000, patches=False, modular_colorspace=0`。
这里仅保证参数含义的映射，不保证与某版 `cjxl` 产生相同文件或耗时。
该组合是值得比较的截图候选，不是默认配置或自动内容分类器。

比较时至少保留普通 e9，以及需要时单独测 `patches=True` 的候选；不要把大 palette、
全采样、patch 全开视为必然更强。它们会相互改变残差分布和建模成本。
如果输入是 JPEG，只有明确指定 `lossless_jpeg=False` 才能用于像素路线；
这会改变归档目标，不能替代 JPEG 字节无损转码。

## 6. JPEG 字节无损转换

```python
from pathlib import Path
import rs_jxl

original = Path("input.jpg").read_bytes()
data, meta = rs_jxl.transcode_jpeg(
    original, effort=9, threads=4,
    source_search=False, return_info=True,
)
assert rs_jxl.reconstruct_jpeg(data) == original
```

| Python 参数 | 默认值 / 范围 | 含义 |
| --- | --- | --- |
| `effort` | **9**；`1..11` | 直接系数后端把 1 映射到 2、11 映射到 10；不是像素 e11 tournament |
| `source_search` | **False** | True 测试 JPEG 候选前缀并保留最小完整结果；不是 PNG/像素搜索 |
| `brotli_quality` | **None**；整数 `0..11` | 只控制 JPEG 重建元数据的 Brotli；默认 `min(effort + 1, 11)`，不是图像质量 |
| `no_cfl` | **False** | True 禁用系数域 chroma-from-luma；两种设置均须能重建原 JPEG |
| `order` | **"source"**；`"source"/"default"/"adaptive"` | 系数扫描顺序策略：按 source 门控、固定默认顺序、显式自适应顺序 |
| `limits` | **None** | 显式资源上限字典；超限失败，不退化到有损或像素重编码 |

`encode("input.jpg")` 使用的是 `encode` 的默认 **effort=7**；
`transcode_jpeg("input.jpg")` 默认 **effort=9**。需要可比结果时显式传相同 effort。
Rust `TranscodeOptions::default()` 和 CLI `transcode-jpeg` 的默认 effort 则是 **3**。

JPEG 搜索请求等级达到 `1/3/5/7/9/10` 时，依次纳入后端 effort
`3/5/7/8/9/10`；请求 11 与 10 使用同一候选前缀。它复用输入准备和可复用数据，
但不是免费搜索。比较搜索前缀体积时可固定 `brotli_quality`，避免自动元数据质量
随请求 effort 改变而混入结果。

Python 转码的 `limits` 支持 `max_input_bytes`、`max_output_bytes`、
`max_coefficients`、`max_tokens`、`sample_limit`、`max_coefficient_bytes`、
`max_block_visits`。省略表示不增加对应资源策略上限，格式限制仍在；这些不是进程级
内存/超时沙箱。转码没有独立 `max_metadata_bytes`，应使用 `max_input_bytes`。

`patches`、`palette_colors`、`rct` 等像素工具不能用于 JPEG 字节转码，即使传 False
或 0 也会报错。支持的 JPEG 结构和明确不支持的特殊形式见
[JPEG API](JPEG_API.md) 与 [兼容性边界](JPEG_COMPATIBILITY.md)。不支持的输入不会
静默解码为像素后重新压缩。

## 7. 多帧压缩参数

`encode_multiframe` 输入的是同尺寸、同样本布局、兼容色彩/extra-channel 元数据的
**完整帧状态**。差分、矩形划分、参考选择和候选编码都在 Rust 中，不在 Python 中。

| 参数 | Python 默认值 | 含义 |
| --- | --- | --- |
| `strategy` | **"crop"** | `"crop"` 只做单 bbox 规划；`"auto"` 对符合条件的 full/crop/mask/多矩形方案试编码，按实际字节选择 |
| `reference` | **"auto"** | `"previous"` 上一帧；`"first"` 第一帧；`"full"` 不做帧间差分；`"none"` 是 full 别名 |
| `durations` | **1** | 单个 tick 数或每帧列表，`0..2^32-1`；0 是真正零时长 layer，不替换为 1 |
| `tps` | **(1000, 1)** | ticks/second 的分子和分母；分子 `1..2^30`，分母 `1..1024` |
| `loops` | **0** | `0..2^32-1`；0 表示无限循环 |
| `auto_options` | **None** | 仅 `strategy="auto"` 可传，包括空字典；省略成员使用 Rust 默认值 |

`reference="auto"` 有两种不同语义：`strategy="crop"` 时选第一帧/上一帧中较小 bbox；
`strategy="auto"` 时让对应方案按实际编码字节比较。后者不是只做面积判断。
多帧不支持额外 `source_search=True`；其 effort 提供共用的 Modular 后端参数，
`strategy="auto"` 选择的是帧表示方式，不是再跑一层完整静态图搜索。

| `auto_options` 成员 | 默认值 / 范围 | 作用 |
| --- | --- | --- |
| `allow_mask` | **True**，bool | 允许安全条件下的二值 alpha mask + BLEND 候选 |
| `allow_multi_rect` | **True**，bool | 允许多个 REPLACE 矩形层；关闭也关闭 merged 候选 |
| `tile_size` | **32**，正 u32 | 初始网格边长；不是每个最终 crop 都必须这么大 |
| `extra_tile_sizes` | **[]**，正 u32 的 list/tuple | 添加其他网格尺度，例如 `[8, 16, 64]`；相同拆分去重，保留原候选，可能增加耗时 |
| `max_rectangles` | **32**，正 usize | 单个多矩形候选的矩形数上限，不是序列总帧数 |
| `adaptive_rectangles` | **True**，bool | 在传统 tiled 候选之外加入成本估计驱动的合并矩形候选 |
| `prune` | **True**，bool | 只用已完成输出给出的字节下界终止必败候选；不是凭估计删候选 |
| `retained_reference` | **False**，bool | 轮换两个完整参考画布，保留获胜差分的背景；要求 `reference="auto"` |
| `reference_lookahead` | **False**，bool | 看下一输入的精确几何差异，辅助决定旧参考的保留；要求 retained_reference=True |
| `max_frames` | **usize::MAX**，非负 usize | 逻辑输入帧数上限 |
| `max_total_trials` | **usize::MAX**，非负 usize | 搜索候选工作预算 |
| `max_trial_pixels` | **u64::MAX**，非负 u64 | 累计提交给编码器的矩形面积，包括 payload 缓存命中；不是变更像素数 |

```python
data, report = rs_jxl.encode_multiframe(
    ["state0.png", "state1.png", "state2.png"],
    effort=9, threads=4, strategy="auto", reference="auto",
    durations=[100, 100, 100],
    auto_options={
        "retained_reference": True,
        "reference_lookahead": True,
    },
    return_info=True,
)
for frame in report["frames"]:
    winner = frame["candidates"][frame["winner_index"]]
    print(winner["strategy"], winner["reference_index"], winner["encoded_bytes"])
```

保留参考策略可能淘汰第一帧；lookahead 只看下一个输入，不编码未来候选，也不保证
全序列最小体积。它们不是任意历史帧搜索或运动补偿，均默认关闭。
多尺度也是可选的搜索扩展；更细的网格可以拆开同一大 tile 里的分散变化，但独立层开销
可能抵消收益。完整/裁剪/mask 不按尺度重复，重复矩形可复用 payload；实际字节数决定
当前候选胜负。候选报告 `tile_size` 标明来源，完整/裁剪/mask 为 None。
上面的普通示例不启用多尺度；e9 实测中 t2 只缩小 0.83%，耗时却变为 2.30 倍，不推荐
默认加入 `[8, 16, 64]`。分散小改动的合成样本收益较好，不能推广到所有图片。
完整体积、bpp 和速度见 [多尺度对照](MULTIFRAME_MULTISCALE_BENCHMARK.md)。
具体槽位、缓存和报告语义见 [自动多帧](MULTIFRAME_AUTO.md) 与
[lookahead 测量](MULTIFRAME_LOOKAHEAD_BENCHMARK.md)。

安全边界与代价：

- 精确差异比较所有通道，包括 invisible RGB 和 extra planes；主方案是 REPLACE + crop。
- mask 当前要求双方都是不超过 16 位、完全不透明、非预乘的整数 GA/RGBA，且没有额外平面。
  不会自动给 RGB 加 alpha，也不对半透明图套用透明黑技巧。
- subsampled extra channels 的插值与合成顺序限制当前 crop 复用，相关输入保留 full-frame。
- 多矩形通常增加零时长中间层，可能增加头开销和解码成本。网格越细、矩形越多不一定越小。
- 自动选择是当前帧候选内的局部字节最优，不承诺低于所有独立 JXL 之和。
- 预算是硬错误边界，不是超时后静默返回低 effort 结果；默认不额外限额。
- 相同 crop payload 的合法复用由内部处理，无需参数；并非所有 mask/patch 候选都能共享。

### 异尺寸与显式 layers

`encode_layers` 使用相同像素工具参数，但不自动推断差分。`canvas=(width, height)`
可省略以推断非负边界；负 origin 需要显式 canvas。每图一项 `layer_options` 可指定
`origin`、`duration`、`timecode`、`name`、`blend`、`extra_blend`、`save_as_reference`。
默认是零时长 REPLACE 层，使用空白参考槽 1；默认不启用动画元数据。

可指定的 blend mode 为 `replace/add/blend/alpha_weighted_add/multiply`，
但有该枚举不等于自动生成正确残差，也不代表任意混合算术都逐位可逆。
普通 decode 返回画布合成结果，`decode_layer` 读取原始层；具体引用合法性、
不同尺寸和透明度语义见 [Python Frames and Layers](PYTHON_API.md#frames-and-layers)。

### 分析工具不是压缩率预测器

```python
analysis = rs_jxl.analyze_multiframe(
    ["state0.png", "state1.png"], threads=4, reference="auto",
    analysis_options={"tile_size": 32, "max_rectangles": 32,
                      "extra_tile_sizes": [8, 16, 64], "history_window": 8},
)
```

当前 `analysis_options` 接受 `allow_mask`、`allow_multi_rect`、
`adaptive_rectangles`、`tile_size`、`extra_tile_sizes`、`max_rectangles`，默认同上；另有
`rectangle_bytes_per_pixel=1.0`，接受有限非负数，仅用于合并矩形的成本估计。
`history_window` 是非负 usize，默认 0；正值检查往前该距离内、除第一/上一帧之外的历史
输入，返回 `history`，不会假装这些帧已经保存在解码器参考槽中。扫描增加的时间包含在
`analysis_seconds`。`improves_static_bbox` 是几何潜力，不是保证压缩更小。
新增 `geometry` 汇总多矩形最小面积、层数和潜在试编码工作量；重复矩形计数可用于判断
payload 复用机会。原 `reference_modes` 仍然只统计单 bbox，避免悄悄改变旧字段含义。
它不运行编码，不接受 effort、prune 或编码预算；不能据此声称已经选出压缩体积最优方案。
分析 API 当前的第一/上一帧几何汇总也不是 retained/lookahead 的实际编码轨迹。
如何读报告、判断是否值得增加搜索，见 [多帧分析使用指南](MULTIFRAME_ANALYSIS_GUIDE.md)。
完整字段定义见 [原生多帧分析](MULTIFRAME_ANALYSIS.md)。

## 8. Python 之外还有哪些工具

更多工具已经在 Rust 中，不意味着可以直接塞进 Python `encoder_options`。
下面列出主要可调入口；这里的默认值是**手动 Rust 工具**，不是 Python source preset。

| 层次 / 字段 | 手动默认值或可选值 | Python 是否直接暴露 |
| --- | --- | --- |
| `EncodeOptions.group_size_shift` / `shared_tree` | 1 / false；共享与局部树布局 | 只有 group shift；Python source 投影决定其余 |
| `EncodeOptions.patches` / `source_patches` | 均 None；分别为 Replace 组件 atlas 与 source Add patch，不能同时启用 | 仅 source 策略布尔 `patches`，不是完整检测器配置 |
| `EncodeOptions.global_palette` / `responsive` / `passes` | false / false / 单 pass；squeeze 与渐进空间 pass 是不同控制 | 否 |
| `EncodeOptions.max_pixels` / `max_output_bytes` | u64::MAX / usize::MAX，显式资源上限 | 像素编码当前不接受通用 `limits`；不要与 JPEG limits 混用 |
| `ModularOptions.rct` / `predictor` | Some(6) / Best；直接设置 rct=None 是恒等，不是 source 的自动 sentinel | source 规范参数可调，底层结构不透传 |
| `weighted_preset` / `weighted_candidates` / `rct_candidates` | 0 / None / None；范围分别 0..4 / 1..5 / 1..19 | 否 |
| `tree_samples` / `max_tree_leaves` / `split_threshold` | 1024 / 16 / 48.0；正采样目标、1..2097152 叶数、有限非负分裂成本 | 否；不是 learning_fraction/ma_max_leaves 的同义词 |
| `sampling` / `sample_limit` | None / None；source 学习策略与可选硬采样上限 | 仅基准 learning_fraction；无原始采样策略字典 |
| `ma_budget_mode` / `ma_max_leaves` | None / None；source 默认 strict、手动旧路径默认 bounded | 仅 ma_max_leaves |
| `fixed_tree` / `custom_tree` | 均 None，预设树或显式合法树，互斥 | 否 |
| `previous_channels` / `tree_groups` | 0 / None；跨通道属性、独立学习域 | 否 |
| `source_ma_parallel` / `source_gather` / `parallel_tokens` | 原生并行策略；额外实验开关不因 threads>1 自动全部开启 | 否，Python 仅指定池大小 |
| `entropy` / `entropy_options` | Auto / 默认模型策略；可用 Prefix、Ans、Source | 否 |
| `lz77` / `lz77_mode` | true / Lazy；Rle、Greedy、Lazy、Optimal | 否；Python source 策略按 effort/预测器选择 |
| `source_lz77` | None；显式 source parser、匹配与执行配置 | 否 |
| `palette_colors` / `palette_predictor` / `palette_policy` | 均 None；精确绝对或 delta palette，source 阶段/成本策略 | Python 只暴露 source 颜色预算，不暴露完整策略 |
| `squeeze` | 空方向列表，true=横向、false=纵向可逆变换 | 否 |

source 投影会替换其负责的字段，例如预测器、树策略、RCT 和 palette 阶段；
不能认为给基础 `EncodeOptions` 填了字段，就必然在 source 调度里原样生效。
直接手动工具与 `CandidateParams` 路线应分开使用，详情见
[Rust Pixel API](PIXEL_API.md#tool-selection)。

### LZ77 的作用范围

LZ77 处理 Modular token 流，并不是跨整个动画任意回看原始图像的通用字典。
Python 当前没有 `lz77`、`lz77_mode`、`probe_lz77` 参数。
普通 source 策略可选 None、Lz77b1w3t、Lz77b3w3t、Optc256 等路线；
具体由 effort 和解析后的预测器决定，不仅看 effort。

Rust 可通过 `ModularOptions.source_lz77` 进一步控制 source 匹配与解析，
或者通过低层 `tools::lz77` 原语研究其他匹配器；低层原语不是独立完整编码入口。
`source_lz77` 的长度/符号编码配置必须与 `entropy_options` 的码流配置一致，
不允许只改一边。可用实现与限制见 [LZ77 移植记录](LZ77_PORT_STATUS.md)。

### CLI 与 Python 不同

CLI `encode --source-effort 9` 对应 source 候选；加 `--source-search` 才是额外搜索。
不指定 source effort 的 CLI `encode` 使用手动 `Tools`，**不是默认 Python e7**。
source effort 与手动工具参数互斥，不能把所有 CLI 开关拼在一起。

手动 CLI 已有 `--rct`、`--predictor`、`--weighted-preset`、`--rct-candidates`、
`--weighted-candidates`、`--palette-colors`、`--palette-predictor`、`--squeeze`、
`--entropy auto|prefix|ans|source`、`--lz77 off|rle|greedy|lazy|optimal`、
`--tree-samples`、`--tree-leaves`、`--tree-split-threshold`、`--previous-channels`、
`--group-size-shift`、`--shared-tree`、`--patches`、`--source-patches`、
`--global-palette`、`--responsive`、`--progressive`。
这些开关不因此成为 Python 参数；手动 CLI `--patches` 是 Replace 工具，
与 Python source `patches` 不能机械等同。

CLI/Rust 另有 [MA 预算](MA_TREE_BUDGET.md)、[精确剪枝](MA_SEARCH_PRUNING.md)
以及 [progressive-group 调度](PROGRESSIVE_GROUP_SCHEDULER.md)。
实验参数可以改变搜索空间或成本；没有测量不能把“更多工具”说成稳定压缩收益。

## 9. 错误与性能判断

以下不是忽略的选项，而是错误：

- `distance>0`、`lossless=False`：未实现有损编码。
- `encoder_options={"lz77_mode": "optimal"}`：没有该 Python/FFI 字段。
- `encoder_options={"modular_predictor": 0}`：字典不转换关键字别名，应写 `predictor`。
- `source_search=True, patches=False`：额外 source 搜索与显式工具覆盖冲突。
- `encode("input.jpg", patches=False)`：默认 JPEG 字节转码不接受像素 patch 控制。
- `strategy="crop", auto_options={}`：自动候选参数不能放到 crop-only 调用。
- `reference_lookahead=True` 但未开启 retained reference，或 retained reference 配合
  `reference="previous"`：引用策略冲突。
- 未知选项、原始 `cjxl` 字符串、数字 libjxl frame-setting ID：没有跨编码器透传。

Python 参数检查通常抛 `TypeError`/`ValueError`；native 解析、资源或编码错误经 FFI
抛 `CodecError`，它是 `RuntimeError` 的子类，并提供 `kind`。
应检查完整错误信息，不把失败当作自动回退或“参数没有效果”。

做压缩比较时同时记录：输出字节、`bpp = 8 * bytes / pixels`、编码时间、解码时间、
`MPix/s = pixels / seconds / 1e6`。等尺寸完整多帧用逻辑画布像素总数，不能用候选
试编码面积或裁剪面积把吞吐量放大；异尺寸 layers 则注明使用原始层面积还是画布面积。

单图 `return_info=True` 的统计在 `meta["report"]`；JPEG 的报告有 `backend_effort`
和 `trials`，多帧的报告则直接提供 `frames`，不要假定三者字段完全一样。
单图 `codec_seconds` 不含外部输入解码、Python/FFI 传输、线程池创建和文件 I/O，
但包含所请求的搜索；JPEG 编码时间还包含系数输入准备。端到端时间应另行测量。

验收目标是解码后的原始样本一致，或 JPEG 重建字节一致，**不是新旧 JXL 文件 SHA256
相同**。颜色、透明度、extra channels 和逻辑帧时序也必须保持。压缩参数都应在这些
约束下比较；本页不承诺任意输入的最佳体积、固定速度提升或 effort 单调性。

## 10. 实现依据

本页依据以下代码核对；后续新增字段应同时更新绑定/原生校验及本页，而不是只添示例：

- [Python 编码入口](../python/rs_jxl/encode.py) 与 [参数归一化](../python/rs_jxl/_options.py)。
- [Python 多帧](../python/rs_jxl/multiframe.py) 与 [显式 layers](../python/rs_jxl/layers.py)。
- [FFI 请求与操作校验](../crates/rs-jxl-ffi/src/request.rs) 与 [编码映射](../crates/rs-jxl-ffi/src/encode.rs)。
- [EncodeOptions](../crates/rs-jxl/src/encode.rs)、[ModularOptions](../crates/rs-jxl/src/modular/encode.rs)。
- [CandidateParams](../crates/rs-jxl/src/scheduler/params.rs)、[source 投影](../crates/rs-jxl/src/scheduler/adapter.rs)、[统一调度入口](../crates/rs-jxl/src/codec.rs)。
- [JPEG 转码参数](../crates/rs-jxl/src/jpeg/archival.rs) 与 [CLI 手动工具](../crates/rs-jxl-cli/src/options.rs)。
