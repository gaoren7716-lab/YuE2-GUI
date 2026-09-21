# 更新日志

## v1.0.0（2026-09）

首个可用版本，把 m-a-p/YuE2 封装成傻瓜式中文图形界面。

### 核心功能
- 五张卡片界面：选风格 / 填歌词 / 规划方式 / 创作风格 / 时长上限。
- 一键安装（`install.bat` / `install.sh`）：自动建环境、装 CUDA 版 PyTorch、从国内镜像下模型。
- 双击启动（`start.bat` / `start.sh`），浏览器自动打开 `http://127.0.0.1:7861/`。

### 小显存支持
- 内置「内存扩展模式」（`ram_mode.py`）：按阶段把闲置权重在内存/显存间搬运。
- 实测 RTX 4070 笔记本 8 GB 可生成 3 分钟以上完整歌曲，峰值显存约 5.5 GiB。

### 创作自由度
- 13 套风格预设（中文流行、粤语抒情、中国风、日系 City Pop、爵士、民谣……）。
- 26 种乐器，含 8 种中国民族乐器（古琴、古筝、琵琶、二胡、竹笛、洞箫、笙、阮）。
- 合奏编制下拉：琴箫合奏 / 筝笛合奏 / 筝琵合奏 / 丝竹乐合奏。
- 组合器：语言 × 流派 × 人声 × 情绪 × 乐器 × 调性 × 拍号 × BPM 自由拼。
- 完整创作参数：采样（temperature/top_p/top_k/repetition_penalty/penalty_window）、
  ODE 步数（16/32/48）、CFG 强度、随机种子、规划档位（off/melody/full）、外部 ABC 乐谱回填。

### 批量与体验
- 「一次出 1 / 2 / 4 首」：参数相同、种子递增，方便挑版本。
- 无损 FLAC 输出；装了 ffmpeg 额外提供 MP3 下载。
- 历史作品回听（最近 8 首）。

### 工具与部署
- `selftest.py` 环境自检、`benchmark.py` 长歌压测。
- `pack.py` 打包成上传云 GPU 的精简 zip；`start.sh --cloud` 远程部署。
- 验证脚本：`verify_params.py` / `verify_plan_score.py` / `verify_ui.js`。
