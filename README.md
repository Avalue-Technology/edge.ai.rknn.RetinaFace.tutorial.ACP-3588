# edge.ai.rknn.RetinaFace.tutorial.ACP-3588
About RKNN Model Zoo - RetinaFace Model with Live USB Camera Tutorial.

# Requirements
This application has been installed in the following environment:
- Debian 11 aarch64
- Python 3.9.2
- librknnrt version: 1.6.0
- RKNN Toolkit2 Lite

# Reference
[RKNPU2 Runtime for Linux aarch64 (v1.5.2)](https://github.com/rockchip-linux/rknpu2/tree/master/runtime/RK356X/Linux/librknn_api/aarch64 "RKNPU2 Runtime for Linux aarch64")

**/usr/lib/librknn_api.so**

**/usr/lib/librknn_api.so**

**RKNN Toolkit2 Lite**

Please run: pip install rknn_toolkit_lite2-1.6.0-cp39-cp39-linux_aarch64.whl to install RKNN Toolkit2 Lite on the target device.

# Application Dependency
Please run: pip install -r requirements.txt

# Find Camera Index
please type: ./ListCamera.sh to find Camera Index.

# Update Application Camera Index
please use text editor to moidfy: RetinaFace_Live_RK3588.py to change Camera Index to match running environment.

# Run Applicaton
Before you start to run, please make sure Retinaface_Live_RK3568.py Camera Index is correct, then you can type:
python3 Retinaface_Live_RK3568.py to run application.

# Execution Screenshot
![RetinaFace.Tutorial.01.png](https://raw.githubusercontent.com/Avalue-Technology/edge.ai.rknn.RetinaFace.tutorial.ACP-3588/refs/heads/main/MarkdownDocumentImages/RetinaFace.Tutorial.01.png "RetinaFace.Tutorial.01.png")
![RetinaFace.Tutorial.02.png](https://raw.githubusercontent.com/Avalue-Technology/edge.ai.rknn.RetinaFace.tutorial.ACP-3588/refs/heads/main/MarkdownDocumentImages/RetinaFace.Tutorial.02.png "RetinaFace.Tutorial.02.png")
![RetinaFace.Tutorial.03.png](https://raw.githubusercontent.com/Avalue-Technology/edge.ai.rknn.RetinaFace.tutorial.ACP-3588/refs/heads/main/MarkdownDocumentImages/RetinaFace.Tutorial.03.png "RetinaFace.Tutorial.03.png")
![RetinaFace.Tutorial.04.png](https://raw.githubusercontent.com/Avalue-Technology/edge.ai.rknn.RetinaFace.tutorial.ACP-3588/refs/heads/main/MarkdownDocumentImages/RetinaFace.Tutorial.04.png "RetinaFace.Tutorial.04.png")