# Windows CUDA 探针脚本

这些脚本在开发机（Windows 11 + RTX 4070）上通过 SSH 以 base64 上传执行，
开发机没有 bash，所以统一用 `python <file>` 跑。

## 按顺序

| 脚本 | 作用 |
|---|---|
| `env_check.py` | 网络/磁盘/Python 基线 |
| `install_torch.py` | 建主 venv + 装 torch cu128 |
| `install_rest.py` | diffusers/transformers/bnb/gguf |
| `t17_install_fast.py` | **triton-windows + flash-linear-attention**（关键） |
| `t48_tts_isolation.py` | 建 TTS 独立 venv（transformers 4.57.3） |
| `t50_tts_cuda.py` | TTS venv 换 CUDA torch |

## 各阶段验证

| 脚本 | 验证内容 |
|---|---|
| `cuda_check.py` | CUDA 可用 + 算力基准 |
| `bnb_test.py` | bitsandbytes 4bit |
| `t1_storyboard.py` / `t1b_storyboard.py` | 分镜 LLM |
| `t36_thinking.py` | **enable_thinking=False 修复** |
| `t39_flux2.py` | FLUX.2 Klein 4B 出图 |
| `t53_tts_ok.py` | Qwen3-TTS 合成 |
| `t41_music.py` | MusicGen |
| `t30_forward_ok.py` / `t31_final.py` | LTX transformer 前向 |
| `t33_q2k.py` | Q2_K 显存实测 |
| `t43_vae_stage.py` / `t44_vae2.py` | **分阶段编排（transformer→释放→VAE）** |

## 已知失败路线（保留作记录）

- `dl_ltx25.py` / `verify_23_compat.py`：LTX-2.5 架构不兼容
- `t21_triton.py`：triton 官方 Windows 无 wheel
- `t46_tts_alt.py` / `t47_tts_conflict.py`：transformers 版本冲突定位
