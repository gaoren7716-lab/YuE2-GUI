# NOTICE — 第三方许可与归属

本仓库（`YuE2-GUI`）的**界面与封装代码**以 MIT License 发布（见 `LICENSE`）。
但本工具依赖的 **YuE2 模型权重** 与 **官方推理包** 受各自许可证约束，**不随本仓库分发**，需按 `install.py` 另行下载。

## 模型权重（YuE2-3B / YuE2-Vae）

- 许可：**Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)**
- 来源：
  - https://huggingface.co/m-a-p/YuE2-3B
  - https://huggingface.co/m-a-p/YuE2-Vae
- 关键限制：**非商用（NonCommercial）**。任何基于该权重生成的音乐产出物，
  均须遵守其非商用许可，**不能用于商业用途**。
- 归属：使用 YuE2 时须注明模型名称与来源仓库（见上）。

## 推理包（yue2_infer wheel）

由 m-a-p 随 YuE2 官方推理套件发布，受其自身许可证约束，由 `install.py` 从
`hf-mirror.com` 下载，不在本仓库内。

## 本仓库不含的内容

- `models/` —— 模型权重（约 7.8 GB），安装时下载。
- `.venv/` —— Python 运行环境。
- `outputs/` —— 用户生成结果。
- `yue2_infer-*.whl` —— 第三方推理包。

## 国内下载镜像（安装脚本已内置）

- 模型：`https://hf-mirror.com`
- PyPI：`https://pypi.tuna.tsinghua.edu.cn/`
- PyTorch 轮子：上海交大 / 阿里镜像
