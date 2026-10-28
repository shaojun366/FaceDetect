# FaceDetect

基于 OpenCV Haar 级联分类器的实时人脸检测工具。打开摄像头后，在画面中标出人脸并显示 FPS 和人脸数量。

当前功能是**人脸检测**：定位画面中的人脸，不包含姓名识别、人脸注册或身份比对。

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

测试使用模拟摄像头覆盖异常清理、FPS 显示、退出行为和参数校验，并使用真实 OpenCV 验证等比例缩放与自带分类器，不需要物理摄像头。GitHub Actions 在 Windows、macOS、Linux 上执行这些测试。

如需使用 PyInstaller 打包，可另外安装 `requirements-build.txt` 中的可选依赖。跨平台发布时应在目标系统上打包，并确认包含 OpenCV 的 `data/` 分类器目录。

## 许可证

[MIT](LICENSE)
