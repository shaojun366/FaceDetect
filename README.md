# FaceDetect

基于 OpenCV 的实时视觉检测与实验工具，支持摄像头人脸检测、屏幕人头检测、可选的 Windows 鼠标跟随，以及跨游戏画风的准确率和延迟评测。

摄像头模式定位人脸，屏幕模式定位模型训练的头部类别。这里的“识别”指位置检测，不包含姓名识别、人脸注册或身份比对。

屏幕研究模式支持 Haar/OpenCV 基线、YOLO ONNX/OpenCV DNN，以及可选的 ONNX Runtime，详见 [屏幕检测、鼠标跟随与算法对比实验](docs/experiments.md)。摄像头仍使用原来的 `face_detection.py` 命令。

```bash
python -m pip install -r requirements-yolo.txt
python download_model.py
python screen_detection.py --detector yolo --backend onnxruntime --model models/anime-head.onnx --frames 300
```

参考模型用于动漫风格人头检测；其他游戏画风需要用对应数据集验证或替换模型。默认只观察和记录数据，鼠标跟随通过 `--aim` 显式启用。

## 安装

使用 Python 3.10 或更高版本，在有图形桌面的 Windows、macOS 或 Linux 上运行。需要可用的摄像头及系统摄像头权限。

```bash
git clone https://github.com/shaojun366/FaceDetect.git
cd FaceDetect
python -m venv .venv
```

Windows PowerShell 无需激活环境，直接运行：

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe face_detection.py
```

macOS / Linux：

```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python face_detection.py
```

后续示例中的 `python` 指上述虚拟环境中的 Python。请自行创建环境；旧版本仓库中的 `bin/`、`lib/` 和根目录 `pyvenv.cfg` 已停止跟踪，不应复用。

## 使用

默认请求摄像头 0 的 640×480 画面，在宽度最多为 320 像素的等比例缩小图上检测，将检测框映射回原始画面。预览默认镜像显示。实际采集分辨率取决于摄像头支持情况。

```bash
python face_detection.py --help
python face_detection.py --camera 1
python face_detection.py --width 1280 --height 720 --process-width 640 --no-mirror
python face_detection.py --scale-factor 1.1 --min-neighbors 5 --min-size 30
```

按 **Esc**、**Q** 或 **Ctrl+C** 退出；Windows、macOS 和 X11 桌面也支持关闭预览窗口退出。Wayland 下请使用退出按键。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--camera` | `0` | 非负摄像头编号 |
| `--width` / `--height` | `640` / `480` | 请求的采集分辨率 |
| `--process-width` | `320` | 检测图像宽度，保持长宽比，不放大原图 |
| `--scale-factor` | `1.2` | 大于 1；减小可增加检测密度，但处理更慢 |
| `--min-neighbors` | `3` | 非负；增大通常可减少误检，也可能漏检 |
| `--min-size` | `20` | 检测图像中最小人脸边长，单位像素 |
| `--no-mirror` | 关闭 | 指定后取消镜像显示 |
| `--cascade` | OpenCV 自带分类器 | 可指定其他 Haar 分类器 XML 路径 |

FPS 每 30 帧更新一次，测量值持续显示；首次测量完成前显示 `0.00`。程序不会保存或上传摄像头画面。

## 常见问题

- **无法打开摄像头**：检查系统摄像头权限，关闭占用摄像头的程序，或尝试 `--camera 1`。
- **无法读取画面**：检查设备是否断开或被其他程序占用。程序会返回非零退出码并释放设备。
- **缺少 cv2 或图形界面错误**：确认安装依赖和运行脚本使用同一个 Python。本工具需要桌面版 `opencv-python`；不要在同一环境混装 `opencv-python-headless`、`opencv-contrib-python` 等共享 `cv2` 名称的包。参见 [OpenCV 安装说明](https://pypi.org/project/opencv-python/)。
- **Linux 缺少系统图形库**：根据发行版安装 OpenCV 所需的图形运行库（例如 Ubuntu 的 `libgl1`、`libglib2.0-0`），并在图形桌面会话中启动。
- **检测漏检或误检**：正面、光线充足的人脸更适合此分类器。侧脸、遮挡和较小人脸的效果有限，可调整处理宽度和检测参数。

## 开发与验证

安装运行依赖后执行：

```bash
python -m unittest discover -s tests -v
```

测试覆盖摄像头异常清理、FPS 显示、屏幕区域坐标、YOLO 后处理、鼠标启用条件和评测报告。小型合成 ONNX 测试模型用于真实推理集成验证，不是已训练的人头模型。测试不截取实际桌面、不移动真实鼠标，也不需要物理摄像头。GitHub Actions 在 Windows、macOS、Linux 上执行这些测试。

如需使用 PyInstaller 打包，可另外安装 `requirements-build.txt` 中的可选依赖。跨平台发布时应在目标系统上打包，并确认包含 OpenCV 的 `data/` 分类器目录。

## 许可证

[MIT](LICENSE)
