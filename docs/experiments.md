# 屏幕人头检测与辅助瞄准研究

共用屏幕采集、目标选择、鼠标控制和计时流程，比较不同检测算法在相同输入下的表现。当前提供 Haar/OpenCV CPU、YOLO ONNX/OpenCV DNN CPU，以及可选的 YOLO ONNX/ONNX Runtime CPU 路径。YOLO 是模型算法族，OpenCV 是图像处理和推理工具；这里可以分别比较 Haar 与 YOLO 算法，以及相同 YOLO 模型的不同推理运行时。

## 准备模型

先安装可选的 CPU ONNX Runtime，再下载一个固定版本的动漫人头参考模型：

```bash
python -m pip install -r requirements-yolo.txt
python download_model.py
```

下载的是 [deepghs/anime_head_detection](https://huggingface.co/deepghs/anime_head_detection) 的 `head_detect_v2.0_n`，保存为 `models/anime-head.onnx`。下载器固定模型仓库版本并校验文件长度及 SHA-256；权重来源和许可证以原模型仓库为准。模型、数据集和实验结果均保留在本地忽略目录中。

这个模型训练的是动漫人头，类别 0 为 `head`；它对写实、低多边形或其他游戏画风的适配效果需要实测。该参考模型采用动态尺寸导出，示例使用 `--backend onnxruntime`；OpenCV DNN 路径请使用自己已验证兼容的固定尺寸 ONNX 导出。两条后端共享相同预处理与后处理，不会静默切换运行时。Haar 基线检测正面人脸，也不能直接视为通用人头检测。通用 COCO 模型中的 `person` 是人体框，不可直接当作头部框使用。

替换模型时，传入自己训练或选择的人头检测 ONNX 文件：

```bash
python screen_detection.py --detector yolo --model models/my-head.onnx --class-id 0 --input-size 640
```

支持以下 **batch=1、FP32、检测框以输入图像像素表示** 的输出协议：

| `--yolo-format` | 输出形状 | 含义 |
| --- | --- | --- |
| `v8`（默认） | `[1, 4+C, N]` | `cx, cy, w, h` 和 C 个类别分数；图内不包含 NMS |
| `v5` | `[1, N, 5+C]` | `cx, cy, w, h, objectness` 和 C 个类别分数；图内不包含 NMS |
| `end2end` | `[1, N, 6]` | `x1, y1, x2, y2, confidence, class_id` |

预处理为居中等比例 letterbox、114 填充、BGR 转 RGB、除以 255。模型输入大小由 `--input-size` 指定，必须与导出配置一致。后处理包含置信度过滤、按类别 NMS、去填充、坐标映射和边界裁剪。不支持分割、姿态、旋转框或坐标已归一化的其他输出协议。

例如，在独立的模型训练环境中导出 YOLOv8 检测模型：

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640 batch=1 dynamic=False simplify=True opset=12 nms=False
```

导出细节参考 [Ultralytics 导出文档](https://docs.ultralytics.com/modes/export/) 和 [OpenCV YOLO DNN 文档](https://docs.opencv.org/4.10.0/da/d9d/tutorial_dnn_yolo.html)。不同 YOLO 版本应选择与实际输出一致的协议。

## 实时屏幕实验

```bash
python screen_detection.py --list-monitors
python screen_detection.py --detector haar --monitor 1 --frames 300 --log results/haar-live.csv
python screen_detection.py --detector yolo --backend onnxruntime --model models/anime-head.onnx --monitor 1 --frames 300 --log results/yolo-live.csv
```

默认最多采集 60 帧/秒；推理较慢时按实际速度运行。`--frames 0` 持续运行。`--region LEFT TOP WIDTH HEIGHT` 可限制采集区域，坐标相对于所选显示器左上角。例如 1920×1080 显示器中央的 640×640 区域：

```bash
python screen_detection.py --detector yolo --backend onnxruntime --model models/anime-head.onnx --region 640 220 640 640
```

小目标实验可以比较“原生分辨率中心裁剪”和“全屏缩小”，并在报告中保留采集配置。鼠标目标参考点始终是所选显示器的中心，区域偏移会正确换算。

默认不显示预览，避免预览遮挡实验画面。`--preview` 可显示检测框、目标选择范围和软件延迟。Windows 会将自己的预览窗口排除在截图外；macOS/Linux 请将预览放到未采集的显示器上。屏幕采集使用 [MSS](https://python-mss.readthedocs.io/latest/usage.html)，macOS 需要系统屏幕录制权限；Linux 需要 MSS 支持的桌面采集环境。

## 鼠标接管实验（Windows）

```bash
python screen_detection.py --detector yolo --backend onnxruntime --model models/anime-head.onnx --aim --window-title "ResearchScene" --aim-key right --radius 180 --smoothing 5 --max-step 25
```

将 `ResearchScene` 替换为目标窗口标题中的一段文字。只有目标窗口处于前台、该帧采集时和发送输入时的前台窗口相同、启用按键仍按住，且帧龄未超过上限时，才发送相对鼠标位移。右键松开后停止；Windows 下 **F8 全局退出**，终端内 **Ctrl+C 退出**。

每帧从 `--radius` 像素半径内选择最靠近显示器中心的人头框中心，误差除以 `--smoothing`，每轴位移不超过 `--max-step`，距离小于 `--deadzone` 时停止移动。`--max-frame-age` 默认 100 毫秒，超过后丢弃该帧的控制指令。无目标时不沿用上一次位置。

输入通过标准 [Windows SendInput](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput) 发送相对位移。相对输入计数与游戏画面像素不是固定 1:1，实际响应受游戏灵敏度、加速度、采样率以及应用是否接收标准输入影响，需要分别校准和记录。目标选择目前不区分敌我。

## 延迟日志

每次实时实验生成 CSV 和同名 JSON。指定日志文件已存在时会报错，避免覆盖记录；未指定时使用带时间戳的路径。

| 字段 | 测量内容 |
| --- | --- |
| `capture_ms` | 从该帧循环开始到获得 BGR 图像，包含前台窗口读取 |
| `detect_ms` | 预处理、模型推理和检测后处理总耗时 |
| `decision_ms` | 目标选择、鼠标步长计算及控制条件检查 |
| `send_ms` | 发送输入阶段耗时；未移动时仅有调用框架开销 |
| `pipeline_ms` | 循环开始到控制阶段结束的软件流水线耗时，不含预览、日志写入和帧率等待 |
| `input_latency_ms` | 发出鼠标位移的帧才记录，测量到 SendInput 返回；没有输入时为空 |

同时记录目标坐标、目标数量、相对位移和启用状态。JSON 保存参数、模型 SHA-256、OpenCV/NumPy/Python 版本、实际推理运行时及其版本、CPU 后端及采集区域。屏幕画面不会保存或上传。

这些数据是软件阶段耗时，**不等于游戏中可见准星的端到端响应延迟**。后者还包含游戏输入处理、渲染、显示刷新等，需要同步录像、受控实验场景或外部测量。实时首帧保留冷启动开销；离线评测提供预热。

## 跨游戏、跨画风离线评测

准备截图及相同语义的人头标注，使用清单文件（图像路径相对于清单所在目录，框为原图像素 `x, y, width, height`）：

```json
{
  "schema_version": 1,
  "images": [
    {"file": "realistic/frame001.png", "game": "game_a", "style": "realistic", "boxes": [[300, 200, 40, 50]]},
    {"file": "anime/frame001.png", "game": "game_b", "style": "anime", "boxes": [[600, 250, 60, 70]]},
    {"file": "negative/frame001.png", "game": "game_a", "style": "realistic", "boxes": []}
  ]
}
```

在完全相同的清单上分别运行：

```bash
python benchmark_detection.py --manifest datasets/eval.json --detector haar --input-size 640 --repeat 10 --warmup 3 --output results/haar
python benchmark_detection.py --manifest datasets/eval.json --detector yolo --backend onnxruntime --model models/anime-head.onnx --input-size 640 --repeat 10 --warmup 3 --output results/yolo
```

报告包含每图 CSV 和汇总 JSON，按 `game/style` 分组报告 Precision、Recall、F1、TP/FP/FN、平均耗时、P50/P95 和推理吞吐。使用置信度排序、IoU≥0.5 的一对一贪心匹配；`--iou` 可调整。每张图只计一次准确率，重复次数只用于延迟采样。图像读取、模型加载和首次图像预热不计入推理耗时。

JSON 还保存标注清单及图像 SHA-256。比较时保持硬件、截图尺寸、CPU 线程数和标注定义一致；默认 `--threads 1`，同时用于 OpenCV 和 ONNX Runtime 的单次推理线程配置。Haar 不使用 YOLO 的置信度/NMS/后端参数，其分数不是校准概率。当前统计是固定阈值下的 Precision/Recall/F1，不是 mAP，也不声称已验证特定游戏画风。

自动化测试中的 `tests/fixtures/synthetic_yolov8.onnx` 只有 401 字节，只验证数据协议和推理链路，会固定输出一个框，**不得用于检测效果实验**。
