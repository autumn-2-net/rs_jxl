# 原生多帧分析使用指南

这份指南回答两个问题：一组图片是否值得做多帧复用，以及是否值得增加搜索。
`analyze_multiframe` 在 Rust 中比较样本、规划矩形；Python 只负责输入和报告。
它不编码、不预测压缩率，也不会自动替你选择最终编码参数。
完整字段定义见 [分析 API](MULTIFRAME_ANALYSIS.md)，参数范围见
[编码参数](ENCODING_PARAMETERS.md)。不要因为接口多就把工具全部打开。

## 1. 输入和环境

- 使用当前 `rs_jxl` Python 包和匹配的原生库，基础依赖为 `cffi`、`numpy`。
  以下路径输入不需要 Pillow 或 Torch。安装方式见 [Python API](PYTHON_API.md)。
- 示例从 `rs_jxl` 仓库根目录运行。开发环境可设置以下变量，使用已有构建：

```powershell
$env:PYTHONPATH = "$PWD/python"
$env:RS_JXL_NATIVE_LIB = "$PWD/target/release/rs_jxl_ffi.dll"
```

已安装 wheel 时不需要上述设置。示例各节沿用前文变量；编码步骤是单独的可选操作。
输入必须是按真实顺序排列的完整状态，不是提前抠出的残差；不要依赖文件枚举顺序。
路径、编码文件字节、NumPy/Torch 数组均可输入，但画布尺寸、样本类型、位深、
通道布局、色彩及 extra-channel 元数据必须匹配。不会自动缩放、补边或统一通道。
异尺寸图片使用显式 `encode_layers`，不是把不匹配输入强塞给自动差分。

这里的 JPEG 会先解码为像素，再做像素无损处理，**不保留原 JPEG 文件字节**。
要原文件可逆归档，请单独使用 `transcode_jpeg`，不要混入本指南的体积比较。
视觉相似的 JPEG 也可能有大量不同像素；不能用肉眼相似度代替精确差异。

## 2. 第一遍只做默认分析

先不增加历史窗口和网格尺度。默认网格为 32、每个多矩形候选最多 32 个矩形，
已有安全 mask 和自适应矩形候选；不是只检查一个 bbox。

```python
from pathlib import Path
from time import perf_counter
import rs_jxl

root = Path('../jxl_python/test_img/mt_lay/t1')
frames = [
    root / '146845733_p0.png',
    root / '146845733_p1.png',
    root / '146845733_p2.png',
    root / '146845733_p3.png',
]
start = perf_counter()
analysis = rs_jxl.analyze_multiframe(frames, reference='auto', threads=4)
print('call_seconds', perf_counter() - start)
print('native_analysis_seconds', analysis['analysis_seconds'])
print('crop_supported', analysis['crop_supported'])

for frame in analysis['frames'][1:]:
    print('frame', frame['index'])
    for name in ('previous_stats', 'first_stats'):
        stats = frame[name]
        print(name, 'changed', f"{stats['changed_ratio']:.2%}",
              'bbox', f"{stats['bbox_ratio']:.2%}",
              'fill', f"{stats['bbox_fill_ratio']:.2%}",
              'mask', stats['mask_eligible'])
    geometry = frame['geometry']
    best = frame['candidates'][geometry['best_area_candidate_index']]
    print('minimum_area', best['strategy'], best['reference_index'],
          f"{best['layer_area_ratio']:.2%}", 'layers', best['layer_count'])
```

首帧没有比较对象，两个 `*_stats` 为 `None`，所以示例从第二帧读差异。
`call_seconds` 包含输入读取/解码、传输及上下文等开销；`analysis_seconds` 不包含这些。
重复调用可用 `Encoder(threads=4)` 复用线程池，但不会跨调用保留参考画布。

## 3. 先读差异，再读候选

| 指标 | 应如何理解 |
| --- | --- |
| `changed_ratio` | 真正发生变化的位置数 / 完整画布像素数，任意通道变化都计入 |
| `bbox_ratio` | 能包住全部变化的一个矩形面积 / 画布面积 |
| `bbox_fill_ratio` | 变化位置 / bbox 面积；越低，bbox 内未变化区域越多 |
| 候选 `layer_area_ratio` | 该方案实际提交的矩形总面积 / 画布面积，不是文件大小比例 |
| 候选 `layer_count` | 普通图像层数；更多小层可能增加头开销、解码和重建成本 |

变化少且集中，单个 crop 通常值得先试。变化少但分布远，bbox 可能接近整图，
此时再看多矩形最小面积是否明显下降。变化本来就多，分割不一定能省下编码工作。
`reference_modes` / `best_geometric_reference` 只比较单 bbox；顶层 `geometry`
还考虑多矩形。两者结果不同不是 bug，也都不是压缩率预测。

`geometry.best_area_candidate_index` 选最小面积，面积相同优先少层，再按稳定顺序。
实际编码可能选择面积更大、层更少或更易压缩的方案。mask 提交的是整个 bbox，
其零值内容可能更好压缩，但面积统计不会显示这个收益。
完全相同的状态差异面积为零，编码仍需要合法的 1x1 重复层，并非零字节。

`suitability` 是几何提示：后续帧平均提交面积低于 30% 为 `strong`，低于 70%
为 `moderate`，其余为 `limited`；分类排除首帧。它不是编码器开关或收益承诺。
总面积和 bpp 的计算则必须包括首帧，不能只统计很便宜的后续差分。

### 透明度与 extra channels

精确比较包括 alpha=0 下的 RGB、所有 extra channels，以及浮点样本位模式。
当前 mask 只适用于双方完全不透明、非预乘、最多 16 位整数的 GA/RGBA，
且没有额外平面。RGB 不会自动加 alpha；半透明图不会被套用透明黑技巧。
`mask_eligible=False` 不表示不能压缩，REPLACE crop 仍可适用。
有子采样 extra channels 时 `crop_supported=False`，当前编码保留完整帧，
避免插值边界破坏精确重建；不能忽略该标志强行裁剪。

## 4. 用工作量判断是否该停止加搜索

```python
g = analysis['geometry']
pixels = analysis['width'] * analysis['height'] * analysis['frame_count']
print('minimum_area_ratio', g['layer_area_ratio'], 'layers', g['layer_count'])
print('proposed_candidates', g['candidate_count'])
print('proposed_trial_pixels/canvas_pixels', g['trial_pixels'] / pixels)
print('proposed_trial_layers', g['trial_layers'])
print('potential_repeated_work', g['unmasked_rectangles'])
```

`trial_pixels` / `trial_layers` 累加所有候选，是剪枝前的潜在工作，不是已执行的编码。
如果最小面积只降低一点，候选层数却成倍增加，就没有理由默认扩大搜索。
这些指标不是秒数，也不是能安全淘汰某候选的字节下界。

`unmasked_rectangles` 统计同一输入内完全相同矩形的潜在 payload 复用。
`unique_pixels` 是不同矩形面积之和，不是空间并集；部分重叠不算相同。
mask 不计入这项复用，实际 patch、剪枝等也会影响缓存命中。
重复量大不等于免费搜索，更不表示不同输入帧之间已经共享压缩数据。

## 5. 出现旧状态回访，再检查历史

```python
history = rs_jxl.analyze_multiframe(
    frames, reference='auto', threads=4,
    analysis_options={'history_window': 8},
)
for frame in history['frames']:
    for item in frame['history']:
        if item['improves_static_bbox']:
            print('opportunity', frame['index'], '<-', item['reference_index'],
                  'bbox', item['stats']['bbox_ratio'])
```

窗口是向前的输入位置距离，跳过已比较的第一/上一帧，不是额外比较次数。
`improves_static_bbox` 要求面积严格小于第一和上一帧两者，仍不保证更省字节。
这里的 `reference_index` 是逻辑输入编号，**不是当前可用的解码器槽位**。
旧状态可能已经被覆盖；增加窗口不会改变编码器的参考保留策略。
窗口很大时扫描会变贵，短序列也不必一开始就开启它。

确有回访机会，再单独评估 `retained_reference=True` 和 `reference_lookahead=True`。
它们放在 `auto_options`，要求 `strategy='auto', reference='auto'`，后者依赖前者。
两者默认关闭；只看下一输入的 lookahead 也不是无限历史搜索或全序列最优。

## 6. 分散变化明显时，才做网格第二遍分析

下面只增加一个尺度作诊断，不启动编码，也不推荐直接加遍所有尺度。

```python
grid_analysis = rs_jxl.analyze_multiframe(
    frames, reference='auto', threads=4,
    analysis_options={'extra_tile_sizes': [16]},
)
for name, result in [('base', analysis), ('grid16', grid_analysis)]:
    g = result['geometry']
    print(name, 'minimum_pixels', g['layer_pixels'],
          'trial_pixels', g['trial_pixels'], 'trial_layers', g['trial_layers'])
```

仅当潜力和工作量值得进一步验证，才在一次受控编码中传入同样的 `extra_tile_sizes`。
不要把 `history_window` 或整个 `analysis_options` 字典直接传给 `auto_options`。
默认合并估价 `rectangle_bytes_per_pixel=1.0` 只是规划参数，不是预测 bpp；
实际编码从首帧测量合并成本，因此即使网格相同，两个报告的矩形也可能不同。

## 7. 可选：编码一次，查看真正的赢家

本节会真正执行 e9 编码，可能需要几十秒；不属于前面的轻量快速开始。
只选代表序列试一次，不在分析脚本中嵌入自动大规模参数遍历。

```python
start = perf_counter()
data, encoded = rs_jxl.encode_multiframe(
    frames, effort=9, threads=4, strategy='auto', reference='auto',
    durations=1, return_info=True,
)
encode_call_seconds = perf_counter() - start
for frame in encoded['frames']:
    winner = frame['candidates'][frame['winner_index']]
    print('winner', frame['index'], winner['strategy'], winner['tile_size'],
          'reference', winner['reference_index'], 'bytes', winner['encoded_bytes'])
    print('completed', len(frame['candidates']), 'pruned', len(frame['pruned']),
          'slots_before', frame['references_before'])
pixels = analysis['width'] * analysis['height'] * analysis['frame_count']
print('bytes', len(data), 'bpp', 8 * len(data) / pixels)
print('encode_call_seconds', encode_call_seconds,
      'MPix/s', pixels / encode_call_seconds / 1e6)
```

输出留在内存，不写文件。报告直接是 `encoded['frames']`，不是单图的 `meta['report']`。
获胜候选的 `elapsed_seconds` 不是整次编码时间；失败候选的搜索也消耗时间。
完整体积用 `len(data)`，不能用某帧赢家的 payload 字节代表整个文件。
这里路径输入的计时包括读取、输入解码、FFI 和编码，不能称为纯后端吞吐量。
要排除输入解码，需预先准备相同样本和元数据，保持各方案计时边界一致。

分析的 `auto` 只做第一/上一帧几何比较；编码的 `strategy='auto'` 实际试编码选字节。
默认 `strategy='crop'` 则只按 bbox 规划，搜索更少，不保证同等体积。
保留历史参考会改变后续状态，当前帧最小也不保证整段最小。

最终用同 effort、线程、像素和元数据对比独立 JXL 总和、full 多帧及 crop/auto。
节省率为 `1 - 多帧字节 / 独立JXL总字节`。bpp 和 MPix/s 都用完整逻辑画布像素数，
不使用裁剪面积、候选试编码面积或通道数量放大吞吐量。解码时间另记，尤其是多小层。
验收比较解码后的全部原始样本和帧时序，不比较新旧 JXL 字节或哈希必须相等。

## 8. 已有结果说明什么

[e9 实测](MULTIFRAME_MULTISCALE_BENCHMARK.md) 的 base 使用单一 32 网格，
另外开启 retained reference 和 lookahead，**不是上节默认 auto 的配置**。
四个原生线程、同版本，以下为一次测量，不能据此宣称稳定加速。

| 序列 | 独立 JXL 总字节 | base 字节 | 相对独立节省 | base / 独立编码秒 |
| --- | ---: | ---: | ---: | ---: |
| t1，四张 PNG | 45,714 | 12,478 | 72.70% | 20.764 / 26.240 |
| t2，四张 JPEG 的像素 | 4,098,261 | 3,623,903 | 11.57% | 39.851 / 10.821 |

再加 `[8, 16, 64]`：t1 仅省 12 B，20.764 -> 22.063 秒；t2 省 0.833%，
39.851 -> 91.696 秒，耗时 2.30 倍。**这些增量不足以支持采用额外搜索，保持默认关闭。**
t2 基础多帧本身也比独立编码慢得多，应按归档需求评估那 11.57%，而非只看文件变小。
t3 同尺寸两张只比独立总和省约 0.16%，额外网格不省字节且更慢，应优先独立编码。
完整 t3 尺寸不一致，被拒绝是正确行为，不应偷偷缩放后报告为无损结果。

## 9. 还能提升，但不靠盲目加候选

当前同位置 crop/mask/多矩形路径在这些样本上的额外网格收益已小，不能据此宣布
多帧或 JXL 到达极限。首帧占比高时继续抠差分收益有限；更值得减少重复搜索成本。
滚动截图、平移区域可能需要跨帧异位置精确块匹配，而不只是更细的同位置 bbox。
这是尚未完成的方向，现有 `patches` 开关不等于自动运动匹配；参考保存阶段、槽位、
字典成本和所有通道的精确还原都要处理，见 [复用路线图](MULTIFRAME_REUSE_ROADMAP.md)。

实践原则：先用便宜分析排查机会，再验证少数代表方案；只省一点却成倍变慢的方案不采用。
能显著缩小且时间合理才值得保留，时间已很短时也不为微小提速主动牺牲体积。
没有实际测量就不声称更小、更快；分析提示用于决策，不用于跳过正确性验收。

已有的 [排序复测](MULTIFRAME_ORDER_BENCHMARK.md) 展示了不增加候选、通过更早取得
有效字节阈值减少 mask 编码工作的尝试，也记录了被否决方案和未获益的输入。
