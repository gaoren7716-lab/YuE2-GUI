#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YuE2 内存扩展模式 —— 把用不到的权重放回内存，小显存也能出完整歌曲。

问题在哪
--------
YuE2 是 Mixture-of-Transformers：每一层里有两条互不相干的路径。

    layer.self_attn + layer.mlp                  → AR 路径（出语义令牌）
    layer.nar_self_attn + layer.nar_mlp          → NAR 路径（出音频隐变量）

权重实测（BF16）：

    AR 路径          2.625 GiB
    NAR 路径         2.625 GiB
    embed_tokens     0.705 GiB
    lm_head          0.705 GiB
    其他             0.103 GiB
    ------------------------------------
    合计             6.763 GiB

官方实现在 pipeline.py 里写死了 `budget = 总显存 - 2 GiB`，然后调用
`torch.cuda.set_per_process_memory_fraction()` 把 PyTorch 的分配上限锁死。
8 GB 卡只有 6 GiB 可用，6.763 GiB 的权重必然溢出——这就是"显存不足"的来源。

怎么绕过去
----------
两条路径**从不同时使用**：

  * AR 生成阶段：只用 AR 路径 + embed + lm_head
  * NAR 合成阶段：只用 NAR 路径 + embed
  * NAR 的 prefill 要用 AR 路径，但那是逐层的一次性计算

所以按阶段把闲置的那条路径挪回内存即可。峰值显存降到约 5.7 GiB，
峰值内存只有 2.625 GB。全程 BF16，不做任何量化。

顺带还能用上内存的另一个好处：Windows 的页面文件可以兜底，实在挤不下
也只是变慢，不会像显存那样直接崩。

本模块用运行期补丁实现，不改动 site-packages 里的官方代码。
"""

from __future__ import annotations

import contextlib
import os
import sys
import time

import torch

GIB = 1024 ** 3

# 每层里属于 AR / NAR 路径的子模块
AR_ATTRS = ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")
NAR_ATTRS = ("nar_input_layernorm", "nar_self_attn", "nar_pre_mlp_layernorm", "nar_mlp")

# 默认给桌面留多少显存。0.93 × 8 GiB ≈ 7.4 GiB
DEFAULT_GPU_FRACTION = float(os.environ.get("YUE2_GPU_FRACTION", "0.93"))

_APPLIED = False
_ORIGINAL_PREFILL = None


# ---------------------------------------------------------------------------
# 权重文件映射
# ---------------------------------------------------------------------------

class WeightFile:
    """把 safetensors 映射进地址空间，按需取零拷贝视图。

    这是内存扩展模式的关键一步。搬回内存的权重如果放在匿名内存里，
    Windows 只能写进页面文件，是实打实的 2.6 GB 硬占用。
    换成文件映射之后，这些页是"干净"的：内存吃紧时系统直接丢弃，
    下次要用再从磁盘读回来。占用从硬占用变成可回收的页缓存。

    实测 8 GB 显存 + 16 GB 内存的机器上，这一步把物理内存可用量
    从 0.5 GB 抬到 3 GB 以上。
    """

    DTYPES = {
        "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
        "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
        "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
    }

    def __init__(self, path):
        import json
        import mmap
        import struct

        self.path = str(path)
        self._file = open(self.path, "rb")
        # ACCESS_COPY 给的是可写缓冲区，但只有真正写入才会产生私有页。
        # 我们从不写权重，所以页始终保持文件支撑、可被系统直接回收。
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_COPY)
        header_size = struct.unpack("<Q", self._mm[:8])[0]
        self.header = json.loads(self._mm[8:8 + header_size].decode("utf-8"))
        self.base = 8 + header_size
        self._cache = {}
        self.hits = 0
        self.misses = 0

    def get(self, name):
        """返回该权重的零拷贝视图；拿不到就返回 None（调用方退回普通拷贝）。"""
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        meta = self.header.get(name)
        if not isinstance(meta, dict):
            self.misses += 1
            return None
        dtype = self.DTYPES.get(meta.get("dtype"))
        if dtype is None:
            self.misses += 1
            return None
        start, end = meta["data_offsets"]
        offset = self.base + start
        itemsize = torch.empty(0, dtype=dtype).element_size()
        if offset % itemsize:
            # torch.frombuffer 要求按元素大小对齐，不满足就放弃零拷贝
            self.misses += 1
            return None
        try:
            flat = torch.frombuffer(self._mm, dtype=dtype,
                                    count=(end - start) // itemsize, offset=offset)
            view = flat.view(meta["shape"]) if meta["shape"] else flat
        except Exception:
            self.misses += 1
            return None
        self._cache[name] = view
        self.hits += 1
        return view

    def close(self):
        self._cache.clear()
        for closer in (self._mm.close, self._file.close):
            try:
                closer()
            except Exception:
                pass


def open_weight_file(model_dir):
    """模型目录里如果是单个 model.safetensors 就映射它，否则放弃。"""
    if os.environ.get("YUE2_MMAP_OFFLOAD", "1") != "1":
        return None
    from pathlib import Path
    candidate = Path(model_dir) / "model.safetensors"
    if not candidate.is_file():
        return None
    try:
        return WeightFile(candidate)
    except Exception as exc:
        print(f"  [内存扩展] 权重文件映射失败（{exc}），退回普通内存模式。")
        return None


# ---------------------------------------------------------------------------
# 阶段调度器
# ---------------------------------------------------------------------------

class RamPlanner:
    """在显存和内存之间按阶段搬运权重。

    常驻显存：embed_tokens / lm_head / norm / rotary / latent_pos_embed / 几个小投影
    来回搬  ：AR 路径整条 <-> NAR 路径整条
    """

    def __init__(self, model, device, weight_file=None):
        self.model = model
        self.device = torch.device(device)
        self.weight_file = weight_file
        self.layers = list(model.model.layers)

        self.ar_groups = [tuple(getattr(layer, name) for name in AR_ATTRS)
                          for layer in self.layers]
        self.nar_groups = [tuple(getattr(layer, name) for name in NAR_ATTRS)
                           for layer in self.layers]

        # 常驻显存的小件。lm_head 单独拎出来：它只在 AR 阶段用得上，
        # NAR 阶段占着 0.7 GiB 纯属浪费。
        self.head_group = (model.lm_head,)
        self.resident = [
            model.model.embed_tokens,
            model.model.norm,
            model.model.rotary_emb,
            model.vae2llm,
            model.llm2vae,
            model.time_embedder,
            model.latent_pos_embed,
        ]

        # 模块 -> 在 state_dict 里的前缀，用来查权重文件
        self._prefix = {id(module): prefix for prefix, module in model.named_modules()}

        self.stage = None
        self.swap_count = 0
        self.swap_bytes = 0
        self.rebound = 0

    # -- 基础搬运 ----------------------------------------------------------

    @staticmethod
    def _size_of(modules):
        total = 0
        for module in modules:
            for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
                if isinstance(tensor, torch.Tensor):
                    total += tensor.numel() * tensor.element_size()
        return total

    def _place(self, modules, device):
        for module in modules:
            module.to(device=device)

    def _place_groups(self, groups, device):
        for group in groups:
            self._place(group, device)

    def _rebind(self, module):
        """把刚搬回内存的权重换成文件映射视图。

        必须逐模块做：.to("cpu") 会先分配一份匿名内存，如果整条路径一起搬，
        峰值会多出 2.6 GB；逐模块做，峰值只有单个模块（最大 75 MB）。
        """
        if self.weight_file is None:
            return
        prefix = self._prefix.get(id(module), "")
        for name, param in module.named_parameters(recurse=True):
            key = f"{prefix}.{name}" if name else prefix
            view = self.weight_file.get(key)
            if view is None:
                continue
            if tuple(view.shape) != tuple(param.shape) or view.dtype != param.dtype:
                continue
            if param.data is view:
                continue
            param.data = view
            self.rebound += 1

    def _to_cpu(self, modules):
        for module in modules:
            module.to(device="cpu")
            self._rebind(module)

    def _to_cpu_groups(self, groups):
        for group in groups:
            self._to_cpu(group)

    def _reclaim(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _switch(self, on_gpu_groups, off_gpu_groups, label):
        started = time.perf_counter()
        self._to_cpu_groups(off_gpu_groups)
        self._reclaim()
        self._place(self.resident, self.device)
        self._place_groups(on_gpu_groups, self.device)
        self.stage = label
        self.swap_count += 1
        self.swap_bytes += self._size_of([m for g in on_gpu_groups for m in g])
        return time.perf_counter() - started

    # -- 对外接口 ----------------------------------------------------------

    def probe(self):
        """查真实设备分布，而不是相信 self.stage。

        pipeline.decode() 会直接对整模型调用 .to("cpu")，绕过本调度器；
        以真实状态为准才不会出现"以为还在卡上、其实已经被搬走"的错判。
        """
        ar = next(self.ar_groups[0][1].parameters()).device.type
        nar = next(self.nar_groups[0][1].parameters()).device.type
        want = self.device.type
        if ar == want and nar != want:
            return "ar"
        if nar == want and ar != want:
            return "nar"
        if ar != want and nar != want:
            return "cpu"
        return "both"

    def ar_stage(self):
        """AR 生成阶段：AR 路径 + lm_head 上卡，NAR 路径下卡回内存。"""
        self.stage = self.probe()
        if self.stage == "ar":
            return 0.0
        return self._switch(self.ar_groups + [self.head_group], self.nar_groups, "ar")

    def nar_stage(self):
        """NAR 合成阶段：NAR 路径上卡，AR 路径和 lm_head 下卡回内存。"""
        self.stage = self.probe()
        if self.stage == "nar":
            return 0.0
        return self._switch(self.nar_groups, self.ar_groups + [self.head_group], "nar")

    def cpu_stage(self):
        """全部下卡。"""
        self.stage = self.probe()
        if self.stage == "cpu":
            return 0.0
        started = time.perf_counter()
        self._to_cpu_groups(self.ar_groups + self.nar_groups + [self.head_group])
        self._to_cpu(self.resident)
        self.stage = "cpu"
        self._reclaim()
        return time.perf_counter() - started

    @contextlib.contextmanager
    def one_ar_layer_on_gpu(self, index):
        """把第 index 层的 AR 权重临时搬上卡，用完立刻搬回去。

        NAR 的 prefill 每层只需要一次前向，所以显存里同时只放一层的
        AR 权重（约 100 MB），而不是整条路径的 2.6 GiB。
        """
        group = self.ar_groups[index]
        self._place(group, self.device)
        try:
            yield
        finally:
            self._to_cpu(group)

    # -- 诊断 --------------------------------------------------------------

    def report(self):
        on_gpu = 0
        on_cpu = 0
        for group in self.ar_groups + self.nar_groups:
            for module in group:
                for tensor in module.parameters(recurse=True):
                    if tensor.device.type == "cuda":
                        on_gpu += tensor.numel() * tensor.element_size()
                    else:
                        on_cpu += tensor.numel() * tensor.element_size()
        return {
            "stage": self.stage,
            "swap_count": self.swap_count,
            "swap_gib": self.swap_bytes / GIB,
            "layer_weights_on_gpu_gib": on_gpu / GIB,
            "layer_weights_on_cpu_gib": on_cpu / GIB,
            "weight_file_mapped": self.weight_file is not None,
            "weights_rebound_to_file": self.rebound,
        }


def attach(model, device, weight_file=None):
    planner = RamPlanner(model, device, weight_file)
    model._yue2_ram = planner
    return planner


def planner_of(model):
    return getattr(model, "_yue2_ram", None)


# ---------------------------------------------------------------------------
# 补丁 1：拿掉显存分配上限
# ---------------------------------------------------------------------------

def patch_memory_fraction(fraction=DEFAULT_GPU_FRACTION):
    """官方把 PyTorch 分配上限锁死在 `总显存 - 2 GiB`，这里改成留一小块给桌面。

    不是简单删掉这个限制：完全不设限会让 PyTorch 有机会吃满整张卡，
    把桌面渲染挤出去。0.93 对 8 GB 卡约等于 7.4 GiB，够用又安全。
    """
    original = torch.cuda.set_per_process_memory_fraction

    def capped(_value=None, device=None):
        return original(fraction, device)

    capped._yue2_ram_fraction = fraction
    torch.cuda.set_per_process_memory_fraction = capped
    return fraction


# ---------------------------------------------------------------------------
# 补丁 2：NAR 的 prefill 改成逐层流式
# ---------------------------------------------------------------------------

def make_streamed_prefill(original_prefill):
    """prefill 只用 AR 路径，所以一层一层搬，显存里只驻留一层。"""

    def _prefill(self):
        planner = planner_of(self.model)
        if planner is None:
            return original_prefill(self)

        backbone = self.model.model
        ids = torch.tensor([self.chunk.ar_tokens], dtype=torch.long, device=self.device)
        positions = torch.arange(self.ar_length, device=self.device)[None]
        cos, sin = backbone.rotary_emb(positions)
        x = backbone.embed_tokens(ids)

        for index, layer in enumerate(backbone.layers):
            with planner.one_ar_layer_on_gpu(index):
                q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos, sin)
                # 可见性受限时必须 clone：切片会连带持有全部不可见令牌的存储
                cached = (k[0, :self.visible_length], v[0, :self.visible_length])
                if self.visible_length != self.ar_length:
                    cached = tuple(t.clone() for t in cached)
                self.cache.append(cached)
                h = self._attention(q[0], k[0], v[0], causal=True)
                x = x + layer.self_attn.o_proj(h.flatten(1)[None])
                x = x + layer.mlp(layer.post_attention_layernorm(x))

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    _prefill.__name__ = "_prefill"
    _prefill.__doc__ = "内存扩展模式：逐层流式 prefill，显存只驻留一层的 AR 权重。"
    return _prefill


# ---------------------------------------------------------------------------
# 补丁 3：注意力分块按序列长度自适应
# ---------------------------------------------------------------------------

def auto_query_block(key_length):
    """按 key 长度反推安全的 query 分块。

    官方 attention() 在 CUDA 上默认把分块设成「整个 query 长度」。长序列下
    PyTorch 找不到可用的 flash 内核（这里的张量是转置出来的非标准步长），
    会退回会物化完整打分矩阵的实现——8000×8000 的 bf16 矩阵就是 2 GiB。
    这正是长歌曲在 8 GB 卡上 prefill 爆显存的根因。

    分块只改变临时存储，不改变可见的 key 集合，结果与不分块完全一致。
    """
    budget = int(os.environ.get("YUE2_ATTN_BUDGET_MB", "192")) * 2 ** 20
    per_key_row = 16 * (2 + 4)          # 16 个 query 头：bf16 打分 + fp32 softmax
    block = budget // max(1, per_key_row * max(1, key_length))
    return int(max(64, min(512, block)))


def fast_cuda_attention(q, k, v, causal=False):
    """CUDA 快速路径：手动扩展 KV 头，让 flash 内核真正被用上。

    官方实现走 `enable_gqa=True`。实测 PyTorch 在这种情况下找不到 flash
    内核，会退回物化完整打分矩阵的实现。微基准（q=[1,16,4500,128]、
    k=[1,8,11000,128]、bf16）：

        官方写法     189.6 ms   峰值临时 7138 MB
        手动扩 KV     18.4 ms   峰值临时  139 MB

    快 10.3 倍、省 51 倍内存。代价只是把 K/V 从 8 头复制到 16 头
    （每层约 43 MB，用完即释放），完全不值一提。

    注意 causal 的语义：官方在分块时会按 query 块截断可见 key。
    本函数只在 Q/K 等长（prefill 场景）时启用，此处不做截断，
    与官方"可见 key 集合"完全一致。
    """
    import torch.nn.functional as F

    query = q.transpose(0, 1).unsqueeze(0).contiguous()
    key = k.transpose(0, 1).unsqueeze(0)
    value = v.transpose(0, 1).unsqueeze(0)
    groups = query.shape[1] // key.shape[1]
    if groups > 1:
        key = key.repeat_interleave(groups, 1).contiguous()
        value = value.repeat_interleave(groups, 1).contiguous()
    else:
        key = key.contiguous()
        value = value.contiguous()
    out = F.scaled_dot_product_attention(query, key, value, is_causal=causal)
    return out[0].transpose(0, 1)


def make_adaptive_attention():
    from yue2.nar import attention as official_attention

    def _attention(self, q, k, v, causal=False):
        # 快速路径：CUDA + SDPA + 没被显式指定分块 + causal 时 Q/K 等长。
        # 官方在 causal 分块时会按 query 块截断可见 key，Q/K 不等长时
        # 语义不同，所以那种情况交回官方实现。
        if (self.query_chunk_size is None and q.device.type == "cuda"
                and self.backend == "sdpa" and not (causal and len(q) != len(k))):
            return fast_cuda_attention(q, k, v, causal=causal)
        block = self.query_chunk_size
        if block is None and q.device.type == "cuda":
            # 回退路径同样要限住分块，否则长序列必然爆显存
            block = auto_query_block(len(k))
        return official_attention(q, k, v, causal=causal, backend=self.backend,
                                  query_chunk_size=block)

    return _attention


# ---------------------------------------------------------------------------
# 补丁 4：synthesize 前后切换阶段
# ---------------------------------------------------------------------------

def make_ram_synthesize(original_synthesize, song_chunks, CachedNAR):
    def _synthesize(model, prefix, codec, seed, steps=32, context=24576, attention="sdpa",
                    offload_ar=False, cancelled=None, query_chunk_size=None,
                    on_progress=None):
        planner = planner_of(model)
        if planner is None:
            return original_synthesize(model, prefix, codec, seed, steps=steps, context=context,
                                       attention=attention, offload_ar=offload_ar,
                                       cancelled=cancelled, query_chunk_size=query_chunk_size,
                                       on_progress=on_progress)
        if model.training:
            raise ValueError("synthesize requires model.eval()")

        # AR 路径下卡、NAR 路径上卡，然后才开始建 chunk
        planner.nar_stage()
        chunks = song_chunks(prefix, codec, seed, context)
        output = []
        for chunk_index, chunk in enumerate(chunks):
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled before acoustic prefill")
            engine = CachedNAR(model, chunk, attention, query_chunk_size)
            try:
                progress = None
                if on_progress is not None:
                    def progress(completed, total, _index=chunk_index, _count=len(chunks)):
                        on_progress(_index * total + completed, total * _count)
                output.append(engine.solve(steps, cancelled, on_progress=progress))
            finally:
                engine.close()
            del engine
        return torch.cat(output, dim=0)

    return torch.inference_mode()(_synthesize)


# ---------------------------------------------------------------------------
# 补丁 4：加载时不要整模型上卡
# ---------------------------------------------------------------------------

def make_ram_load_model(original_load_model):
    def _load_model(self, for_nar=False):
        if self.quantization != "none":
            raise RuntimeError(
                "内存扩展模式不做量化（FP8 会额外在内存里留一份 BF16 原始权重）。"
                "请把 quantization 设为 none。")
        loading = self._model is None
        with (self._status("Loading model") if loading else contextlib.nullcontext()):
            if self._model is None:
                from yue2.modeling_yue2 import YuE2ForCausalLM
                start = time.perf_counter()
                self._model = YuE2ForCausalLM.from_pretrained(
                    self.model_dir, local_files_only=True,
                    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
                self.load_timing["mot_load_seconds"] = time.perf_counter() - start
                # 权重文件映射：让搬回内存的权重变成可回收的页缓存，
                # 而不是硬占着物理内存。拿不到就自动退回普通模式。
                attach(self._model, self.device, open_weight_file(self.model_dir))

        if not for_nar:
            self._model._yue2_ram.ar_stage()
        return self._model

    return _load_model


# ---------------------------------------------------------------------------
# 安装
# ---------------------------------------------------------------------------

def install(device="cuda", fraction=None):
    """打上全部补丁。必须在构造 YuE2Pipeline 之前调用。"""
    global _APPLIED, _ORIGINAL_PREFILL
    if _APPLIED:
        return planner_of

    from yue2 import nar as nar_module
    from yue2 import pipeline as pipeline_module

    frac = patch_memory_fraction(fraction if fraction is not None else DEFAULT_GPU_FRACTION)

    _ORIGINAL_PREFILL = nar_module.CachedNAR._prefill
    nar_module.CachedNAR._prefill = make_streamed_prefill(_ORIGINAL_PREFILL)
    nar_module.CachedNAR._attention = make_adaptive_attention()
    nar_module.synthesize = make_ram_synthesize(
        nar_module.synthesize, nar_module.song_chunks, nar_module.CachedNAR)
    pipeline_module.YuE2Pipeline._load_model = make_ram_load_model(
        pipeline_module.YuE2Pipeline._load_model)

    # decode() 会绕过调度器直接对整模型 .to("cpu")，那样搬回内存的权重
    # 就变回匿名内存了。改成走调度器，顺带重新绑定到文件映射。
    original_decode = pipeline_module.YuE2Pipeline.decode

    def _decode(self, latents, *args, **kwargs):
        planner = planner_of(self._model) if self._model is not None else None
        if planner is not None:
            planner.cpu_stage()
        return original_decode(self, latents, *args, **kwargs)

    pipeline_module.YuE2Pipeline.decode = _decode

    # 关掉 pipeline 时一并释放权重文件的映射，不然模型文件会一直被占着，
    # 重新安装模型时会删不掉。
    original_close = pipeline_module.YuE2Pipeline.close

    def _close(self):
        model = self._model
        planner = planner_of(model) if model is not None else None
        if planner is not None and planner.weight_file is not None:
            planner.weight_file.close()
            planner.weight_file = None
        return original_close(self)

    pipeline_module.YuE2Pipeline.close = _close

    _APPLIED = True
    return frac


def installed():
    return _APPLIED


# ---------------------------------------------------------------------------
# 内存体检
# ---------------------------------------------------------------------------

def system_memory():
    """返回 (总物理内存, 可用物理内存, 提交上限剩余)，单位 GB。失败返回 None。

    Linux 走 /proc/meminfo，Windows 走 GlobalMemoryStatusEx。两边都不通就返回 None，
    调用方会跳过内存提示，不影响主流程。
    """
    info = _system_memory_linux()
    if info is not None:
        return info
    return _system_memory_windows()


def _system_memory_linux():
    """Linux：读 /proc/meminfo。比调系统 API 简单，而且容器里也读得到。"""
    try:
        fields = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                fields[key.strip()] = rest.strip()

        def kb(name):
            return int(fields[name].split()[0]) * 1024

        scale = 1000 ** 3
        total = kb("MemTotal") / scale
        avail = kb("MemAvailable") / scale
        # CommitLimit - Committed_AS 才是"系统还能再提交多少虚拟内存"，
        # 这才是 Windows 页面文件剩余量的对等物。字段缺失就退回可用物理内存。
        try:
            commit = (kb("CommitLimit") - kb("Committed_AS")) / scale
        except Exception:
            commit = avail
        return (total, avail, commit)
    except Exception:
        return None


def _system_memory_windows():
    """Windows：GlobalMemoryStatusEx。"""
    try:
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = _Status()
        status.dwLength = ctypes.sizeof(_Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        scale = 1000 ** 3
        return (status.ullTotalPhys / scale, status.ullAvailPhys / scale,
                status.ullAvailPageFile / scale)
    except Exception:
        return None


def memory_advice(need_gb=2.0):
    """跑之前先看一眼内存够不够，不够就给出人话建议。

    阈值 3.5 GB，与界面里的提示线保持一致。
    实测生成过程中物理可用最低会掉到 0.45 GB（靠 mmap 回填 + 页面文件兜住），
    所以 3.5 GB 这个门槛是"启动前建议值"，不是硬性下限 —— 低于它照样能跑，
    只是生成期间系统会明显变卡，提前关掉浏览器和聊天软件体验好很多。
    """
    info = system_memory()
    if info is None:
        return {"ok": True, "message": ""}
    total, avail, commit = info
    if avail >= need_gb + 1.5:
        return {"ok": True, "message": "", "total": total, "available": avail}
    hint = (f"内存偏紧：可用 {avail:.1f} GB，建议至少留 {need_gb + 1.5:.1f} GB。"
            "关掉浏览器、聊天软件等程序再生成会稳很多。")
    return {"ok": False, "message": hint, "total": total, "available": avail}


def trim_memory(weight_file=None):
    """把本进程的工作集尽量还给系统，缓解生成结束后的内存压力。

    重点是 mmap 回填的那些权重页：它们是文件支撑的干净页，丢掉零成本
    （下次用到时从磁盘重读），但能立刻把物理内存交还给系统和其他程序。
    生成完成后调一次，桌面就不会一直卡着。

    返回 True 表示调用成功，不代表具体释放了多少。
    """
    try:
        if sys.platform == "win32":
            return _trim_windows()
        return _trim_linux(weight_file)
    except Exception:
        return False


def _trim_windows():
    """Windows：EmptyWorkingSet。失败则退回 SetProcessWorkingSetSize(-1, -1)。"""
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    handle = kernel32.GetCurrentProcess()

    try:
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
        psapi.EmptyWorkingSet.restype = ctypes.c_int
        if psapi.EmptyWorkingSet(handle):
            return True
    except Exception:
        pass

    # 官方文档：SetProcessWorkingSetSize 传 -1/-1 等价于清空工作集
    kernel32.SetProcessWorkingSetSize.argtypes = [ctypes.c_void_p,
                                                  ctypes.c_size_t, ctypes.c_size_t]
    kernel32.SetProcessWorkingSetSize.restype = ctypes.c_int
    return bool(kernel32.SetProcessWorkingSetSize(
        handle, ctypes.c_size_t(-1), ctypes.c_size_t(-1)))


def _trim_linux(weight_file=None):
    """Linux：对 mmap 区域发 MADV_DONTNEED，内核会把干净页直接丢掉。"""
    import mmap as _mmap

    target = getattr(weight_file, "_mm", None) if weight_file is not None else None
    if target is None:
        return False
    try:
        target.madvise(_mmap.MADV_DONTNEED)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 命令行自检
# ---------------------------------------------------------------------------

def _selftest():
    import sys
    from pathlib import Path

    root = Path(__file__).parent.resolve()
    sys.path.insert(0, str(root))
    install()
    from yue2 import YuE2Pipeline

    models = root / "models"
    info = system_memory()
    if info:
        print(f"[内存] 总 {info[0]:.1f} GB / 可用 {info[1]:.1f} GB / 页面文件可用 {info[2]:.1f} GB")
    free, total = torch.cuda.mem_get_info()
    print(f"[显存] 空闲 {free / GIB:.2f} GiB / 共 {total / GIB:.2f} GiB")

    print("[加载] 读取模型（首次较慢）…")
    t0 = time.perf_counter()
    pipe = YuE2Pipeline.from_pretrained(
        str(models / "YuE2-3B"), vae=str(models / "YuE2-Vae"),
        local_files_only=True, progress=False, device="cuda",
        memory_budget_gib=8, backend="torch-eager",
        quantization="none", offload_ar=True, verify_hashes=False,
    )
    print(f"[加载] 完成，{time.perf_counter() - t0:.1f} 秒")

    print("[AR] 只跑语义阶段，验证阶段切换…")
    t0 = time.perf_counter()
    plan = pipe.plan(style="Mandarin pop ballad, female vocal, piano, 80 BPM",
                     lyrics="[Verse]\n测试一下\n[Chorus]\n看看能不能行")
    semantic = pipe.generate_semantic(plan, sampling={"max_tokens": 400, "min_tokens": 100})
    print(f"[AR] 输出 {len(semantic.tokens)} 个令牌，{time.perf_counter() - t0:.1f} 秒")
    print(f"[AR] 计划 ABC {len(plan.abc or '')} 字符")

    print("[NAR] 合成音频（含逐层流式 prefill）…")
    t0 = time.perf_counter()
    latents = pipe.synthesize(semantic)
    print(f"[NAR] 隐变量 {tuple(latents.shape)}，{time.perf_counter() - t0:.1f} 秒")

    print("[VAE] 解码…")
    t0 = time.perf_counter()
    audio = pipe.decode(latents)
    print(f"[VAE] 音频 {audio.shape}，{time.perf_counter() - t0:.1f} 秒")
    print(f"[结果] 时长 {len(audio) / 48000:.1f} 秒，峰值显存 {torch.cuda.max_memory_allocated() / GIB:.2f} GiB")
    print(f"[调度] {pipe._model._yue2_ram.report()}")


if __name__ == "__main__":
    _selftest()
