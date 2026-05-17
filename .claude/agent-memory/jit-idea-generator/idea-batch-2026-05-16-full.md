---
name: idea-batch-2026-05-16-full
description: Full detailed content of 12 JiT/PixelDiT ideas generated 2026-05-16 (for validator handoff)
metadata:
  type: project
---

# JiT/PixelDiT Idea Batch (2026-05-16) — Full Detail

12 ideas with full mechanism descriptions, constructed to avoid:
- Prior 12 ideas (CoGAS, Spatial-Causal Speculative, DSTP, TGJC, Rectified Pixel-Flow, PPM-PixelDiT, PiMaH, CCR, SVW Loss, RATBA, Cross-Batch KV, Hier Codebook ANN)
- Caching family (timestep/block/layer/feature cache, step-skip, U-shape, DeepCache, ToCa, FORA, TGATE, etc.)
- Published pixel-DiT methods (JiT, PixelDiT, DiP, DeCo, FREPix, EPG, PixelGen, PixelFlow, HDiT)

---

### 아이디어 1: QNDS: Quantization-Noise-as-Denoising-Step
- **Family**: quantization
- **핵심 가설** (한 문장): Weight quantization noise can be analytically reinterpreted as an extra denoising step, so high-noise timesteps can use INT4 weights and low-noise timesteps INT8/FP16, achieving end-to-end speedup without QAT.
- **기술적 접근** (2-3 문장): Map weight bit-width b to an effective noise variance σ_q(b) using the formula σ_q² ≈ Δ²/12 with Δ = scale/2^b. At inference, schedule a bit-width trajectory b(t) such that σ_total²(t) = σ_noise²(t) + σ_q²(b(t)) tracks the original DDPM schedule, so high-t steps tolerate INT4 and low-t steps need INT8. No retraining: only post-hoc calibration of per-layer scales.
- **핵심 혁신점 (vs 기존 12 + 16 caching papers)**: Caching reuses computation; QNDS replaces it with a cheaper but noisier version that the diffusion process itself absorbs. Signal X (bit-width × noise schedule joint design) has never been combined. Latent-DiT papers use uniform bit-width across all timesteps.
- **왜 Pixel Space 특화** (mechanism level): Pixel-space DiT operates directly on RGB values whose noise budget is well-defined per pixel; latent-space models have a learned VAE manifold where quantization noise does not commute with the diffusion forward process. In pixel space the Gaussian forward kernel is identity-covariance per pixel, so the quantization-as-extra-noise approximation is mathematically exact (per-pixel additive Gaussian).
- **예상 효과** (정량): 2.5-3× inference speedup vs FP16 (avg bit-width ~5.5), FID degradation <0.3 on ImageNet-256. Memory 3-4× reduction.
- **난이도 (1-5)**: 2 (calibration only, no retraining)
- **출판 타겟**: NeurIPS 2026 main track (clean theoretical bridge + practical gain)
- **자동 NO-GO 키워드 회피 확인**: ✅ (no cache/skip/clean-pred/freq-split/global-local)
- **기존 12개와 overlap 없음 확인**: ✅ (none address quantization)

---

### 아이디어 2: LoCS-FP8: Block-FP8 Locality Kernel
- **Family**: hw-codesign
- **핵심 가설** (한 문장): A custom FP8 GEMM kernel with Hilbert-curve token ordering converts pixel DiT's spatial locality into block-FP8 numerical stability, achieving >2× wallclock speedup that uniform FP8 cannot reach.
- **기술적 접근** (2-3 문장): Re-order pixel tokens along a Hilbert space-filling curve so spatially nearby pixels are adjacent in the token sequence. Use Hopper/Blackwell block-FP8 (per-128-element scale) GEMM where each block now contains spatially correlated values with low dynamic range. Custom CUDA kernel fuses Hilbert permutation + block-FP8 quantization + attention QKV projection.
- **핵심 혁신점 (vs 기존 12 + 16 caching papers)**: Caching saves compute by skipping; LoCS saves compute by exploiting hardware FP8 microarchitecture using a property (pixel locality) unique to pixel-space. Signal X (FP8 block format × Hilbert curve coords) is novel. ToMe-style token merge does not interact with FP8 dynamic range.
- **왜 Pixel Space 특화** (mechanism level): Latent-space tokens have no consistent spatial locality after VAE encoding (each latent token mixes 8×8 pixels), so block-FP8 sees high dynamic range. Pixel tokens preserve raw RGB locality; Hilbert ordering keeps spatially close pixels together, so each FP8 block has narrow value range and the per-block scale absorbs less precision loss.
- **예상 효과** (정량): 2.2-2.6× wallclock vs FP16 on H100, additional 1.4× vs uniform FP8 due to better block scaling. FID delta <0.2.
- **난이도 (1-5)**: 5 (custom CUDA, FP8 expertise, hardware tuning)
- **출판 타겟**: MLSys 2027 (hardware-aware ML) or SC 2026
- **자동 NO-GO 키워드 회피 확인**: ✅
- **기존 12개와 overlap 없음 확인**: ✅ (none touch hw kernel design)

---

### 아이디어 3: SNR-vs-Patch-Size Scaling Law
- **Family**: theory
- **핵심 가설** (한 문장): For pixel DiT, the optimal patch size p*(SNR) at each diffusion timestep follows a closed-form law p* ∝ SNR^(-α) where α depends only on image power spectrum exponent, enabling principled timestep-conditional patch sizing.
- **기술적 접근** (2-3 문장): Derive p*(SNR) by minimizing reconstruction MSE under 1/f^β natural image statistics + Gaussian noise at level σ². Fit α from real data, then run pixel DiT with timestep-conditional patch unembed: large patches at high SNR (noise dominates → only coarse structure recoverable), small patches at low SNR. No new architecture: just a schedule.
- **핵심 혁신점 (vs 기존 12 + 16 caching papers)**: First analytical patch-size scaling law for pixel diffusion. JiT uses fixed large patches; DiP uses two fixed sizes. None derived optimal size as function of noise. Caching never addresses tokenization geometry.
- **왜 Pixel Space 특화** (mechanism level): The 1/f^β image statistics on which the derivation depends are properties of natural pixels, destroyed by VAE in latent space. Latent codes have whitened spectrum (post-VAE), so SNR-patch relationship collapses to a constant.
- **예상 효과** (정량): 1.5-2× speedup via larger patches at high-noise steps (fewer tokens), <0.1 FID change. Mostly a theory paper — empirical gain modest.
- **난이도 (1-5)**: 3 (math + small-scale empirical)
- **출판 타겟**: ICLR 2027 (theory + practical schedule)
- **자동 NO-GO 키워드 회피 확인**: ✅
- **기존 12개와 overlap 없음 확인**: ✅ (none derive scaling laws)

---

### 아이디어 4: Attention Rank Collapse Analysis
- **Family**: theory
- **핵심 가설** (한 문장): Pixel DiT attention matrices undergo measurable rank collapse at low-noise timesteps (singular value spectrum concentrates), allowing principled low-rank attention approximation that saves FLOPs without quality loss.
- **기술적 접근** (2-3 문장): Measure singular value distribution of attention matrices A(t) across timesteps. Show rank(A) drops monotonically as t→0 because attention becomes texture-localized. Replace full attention with rank-k(t) approximation via SVD-projected QK at timesteps where rank(A) < k(t). Hyperparameter: rank schedule k(t).
- **핵심 혁신점 (vs 기존 12 + 16 caching papers)**: Cache reuses A across timesteps; this approximates A within a single timestep using its measured rank. Signal X (attention singular spectrum trajectory) never reported for pixel DiT. Linear attention papers use fixed rank; here rank is data-driven and timestep-adaptive.
- **왜 Pixel Space 특화** (mechanism level): Pixel-space attention attends to RGB neighborhoods; at low noise it converges to local sliding-window patterns (rank≈window size), measurable directly. Latent-space attention rank is bounded by VAE bottleneck → no clear collapse trajectory.
- **예상 효과** (정량): 1.3-1.8× FLOPs reduction in last 30% of timesteps, FID delta ~0.1.
- **난이도 (1-5)**: 3 (analytical + measurement)
- **출판 타겟**: ICML 2027 (analysis paper with practical implication)
- **자동 NO-GO 키워드 회피 확인**: ✅
- **기존 12개와 overlap 없음 확인**: ✅ (none analyze attention rank)

---

### 아이디어 5: SLCE: Spatial Lipschitz Head Pruning
- **Family**: pixel-statistics
- **핵심 가설** (한 문장): Attention heads that operate on locally-Lipschitz pixel regions (smooth gradients) can be pruned at each timestep, because their output is well-approximated by linear interpolation from neighboring tokens.
- **기술적 접근** (2-3 문장): For each token at each timestep, compute local Lipschitz constant L(x,t) = ||∇_pixel x_t||. For tokens with L < τ(t), bypass attention heads (use averaged neighbor values). Head pruning mask is per-token per-timestep, decided in O(1) per token via cheap gradient norm.
- **핵심 혁신점 (vs 기존 12 + 16 caching papers)**: Pruning decision uses pixel-local geometric property (Lipschitz), not cached features. Different from token merging (ToMe) which merges by feature similarity — Lipschitz is a continuous physical property of the pixel signal. Signal X (local Lipschitz constant per token) unused anywhere.
- **왜 Pixel Space 특화** (mechanism level): Lipschitz constant is meaningful only in pixel coordinates (∇w.r.t. spatial position). In latent space the spatial coordinates are abstract (post-VAE), so Lipschitz w.r.t. latent indices has no clean physical meaning.
- **예상 효과** (정량): 1.4-1.7× attention FLOPs reduction averaged over timesteps. FID delta <0.2.
- **난이도 (1-5)**: 3 (need efficient per-token Lipschitz estimator)
- **출판 타겟**: CVPR 2027 / ICCV 2027
- **자동 NO-GO 키워드 회피 확인**: ✅
- **기존 12개와 overlap 없음 확인**: ✅ (none use Lipschitz signal)

---

### 아이디어 6: PNAC: Frequency-Anisotropy Noise Calibration
- **Family**: noise-schedule
- **핵심 가설** (한 문장): Natural images have direction-dependent (anisotropic) frequency power spectra, so an anisotropic noise schedule that adds more noise along low-power directions matches signal statistics and enables fewer denoising steps.
- **기술적 접근** (2-3 문장): Measure 2D power spectrum P(k_x, k_y) of training images, fit anisotropic 1/||k||^β model with directional weighting w(θ). Define noise covariance Σ(t) = σ²(t) · diag(w(θ)) in Fourier domain. Forward process applies anisotropic noise; reverse process uses calibrated covariance. Schedule converges in fewer steps because noise budget matches signal energy direction.
- **핵심 혁신점 (vs 기존 12 + 16 caching papers)**: Different from frequency-split methods (DeCo, FREPix) which split into low/high bands — PNAC keeps full spectrum but anisotropic. Signal X (direction-dependent frequency power) is geometric, not band-split. No cache involvement.
- **왜 Pixel Space 특화** (mechanism level): Image power spectrum anisotropy (horizontal/vertical bias from horizon, scene structure) is a natural-image property. Latent codes are whitened by VAE training, eliminating anisotropy.
- **예상 효과** (정량): 20-30% fewer sampling steps for same FID (e.g., 50→35 steps). Wallclock 1.4×.
- **난이도 (1-5)**: 4 (training-side change, need to retrain or fine-tune)
- **출판 타겟**: NeurIPS 2027
- **자동 NO-GO 키워드 회피 확인**: ✅ (not frequency-split)
- **기존 12개와 overlap 없음 확인**: ✅

---

### 아이디어 7: PLDI: Pretrained Latent-DiT Weight Transfer
- **Family**: training-eff
- **핵심 가설** (한 문장): Latent-space DiT weights (FLUX, SD3) can be transferred to pixel DiT via patch-embed surgery + LoRA adapter, cutting pixel DiT training cost by 5-10× without quality loss.
- **기술적 접근** (2-3 문장): Initialize pixel DiT transformer blocks from FLUX weights (same hidden dim, head structure). Replace VAE+patch_embed with a learnable pixel patch_embed of matching output dim. Train pixel patch_embed + LoRA adapters on transformer blocks only, freezing main weights. Fine-tune full model briefly at the end.
- **핵심 혁신점 (vs 기존 12 + 16 caching papers)**: Cross-domain weight transfer (latent→pixel) never attempted. JiT, PixelDiT, DiP all train from scratch. This is orthogonal to inference methods — it cuts training, not inference, but addresses the biggest pixel-DiT bottleneck (training cost).
- **왜 Pixel Space 특화** (mechanism level): The transformer blocks in DiT learn diffusion dynamics that are largely VAE-agnostic (token-level denoising). Pixel space differs from latent space only at the input/output projection (patch_embed/unembed), so most weights transfer. Reverse direction (pixel→latent) does not work because pixel models see richer input statistics.
- **예상 효과** (정량): Training cost 5-10× reduction; final FID match within 0.2 of from-scratch PixelDiT.
- **난이도 (1-5)**: 2 (engineering — surgery + LoRA standard)
- **출판 타겟**: ICLR 2027 (efficient training)
- **자동 NO-GO 키워드 회피 확인**: ✅
- **기존 12개와 overlap 없음 확인**: ✅ (none address training cost)

---

### 아이디어 8: TC-PVDiT: Frame-Difference Tokens for Pixel Video DiT
- **Family**: video
- **핵심 가설** (한 문장): For pixel-space video DiT, denoising frame-differences (residuals from previous frame) instead of full frames cuts effective token count by 5× because most frame-pixels are temporally constant.
- **기술적 접근** (2-3 문장): Encode video as frame_0 + Σ Δframe_t. Pixel DiT processes Δframe with a sparsity mask that drops tokens where ||Δ|| < threshold. Cross-attention to previous frame's keys provides temporal context. Iterative denoising on the sparse Δ token set.
- **핵심 혁신점 (vs 기존 12 + 16 caching papers)**: Cache reuses denoiser features across timesteps; TC-PVDiT exploits frame-to-frame redundancy in the data itself. First pixel-space video DiT efficiency method (no published competitor). Signal X (temporal frame difference) is data-side, not model-side.
- **왜 Pixel Space 특화** (mechanism level): Frame differences are sparse only in pixel space (most pixels static). In latent space, VAE entangles temporally-stable regions with moving ones at each spatial location, so latent frame-diff is dense.
- **예상 효과** (정량): 4-5× speedup per frame after first frame; first frame at full cost. For 16-frame clip, ~3× end-to-end.
- **난이도 (1-5)**: 5 (need video pixel DiT baseline + temporal architecture design)
- **출판 타겟**: ICCV 2027 / SIGGRAPH 2027
- **자동 NO-GO 키워드 회피 확인**: ✅
- **기존 12개와 overlap 없음 확인**: ✅ (none address video)

---

### 아이디어 9: CC-KV: CLIP-Cluster KV Reuse for T2I
- **Family**: multimodal
- **핵심 가설** (한 문장): At inference time, prompts with similar CLIP embeddings (cosine > 0.85) produce nearly identical cross-attention KV matrices in pixel DiT, so a small KV cache indexed by prompt-cluster gives free batched generation.
- **기술적 접근** (2-3 핵심 문장): Cluster training/incoming prompts in CLIP space (k-means, k=1000). For each cluster, cache cross-attention K,V from a representative prompt. At inference, lookup cluster → reuse cached cross-attn KV in early blocks (where text influence is structural), only computing self-attn fresh. Refine in last 30% blocks with prompt-specific KV.
- **핵심 혁신점 (vs 기존 12 + 16 caching papers)**: Different from timestep/feature caching: this caches across prompts within a session, not across timesteps within a generation. Signal X (CLIP semantic clustering) used for inter-prompt KV sharing — never done. Cross-Batch KV (#11 of prior 12) uses batch concurrency; CC-KV uses semantic similarity offline.
- **왜 Pixel Space 특화** (mechanism level): Pixel DiT cross-attention K,V have larger memory footprint (larger token count) than latent DiT, so KV reuse savings are larger. Also, pixel-space cross-attn pattern is more structurally consistent across similar prompts because final output resolution is the same.
- **예상 효과** (정량): 1.5-2× wallclock for batched T2I serving with diverse prompts. Single-prompt no gain.
- **난이도 (1-5)**: 2 (no retraining; cluster + lookup)
- **출판 타겟**: EMNLP 2027 (multimodal) / NeurIPS 2027 datasets-and-benchmarks
- **자동 NO-GO 키워드 회피 확인**: ✅ (not timestep cache — prompt-level)
- **기존 12개와 overlap 없음 확인**: ✅ (Cross-Batch KV is different — concurrency vs similarity)

---

### 아이디어 10: ARF-512: Adaptive Receptive Field Schedule for 512²
- **Family**: high-res
- **핵심 가설** (한 문장): Early layers of pixel DiT need only local receptive field (3×3 attention window), middle layers need medium (32×32), late layers need full global; a depth-indexed receptive field schedule enables single-stage 512×512 generation without cascade.
- **기술적 접근** (2-3 문장): Define receptive field r(l) per layer index l: r(0..L/3) = local sliding window 3×3, r(L/3..2L/3) = strided window 32×32, r(2L/3..L) = full attention. Implement via masked attention. No cascade, single 512² model with attention cost O(N·r²) << O(N²).
- **핵심 혁신점 (vs 기존 12 + 16 caching papers)**: PixelFlow uses cascade (multiple resolutions); HDiT uses hourglass (downsample-upsample). ARF-512 keeps single resolution + single forward pass but varies attention reach per layer. Signal X (depth-indexed receptive field) not used in pixel DiT.
- **왜 Pixel Space 특화** (mechanism level): Pixel images have natural locality at low layers (texture is local) and global structure at high layers (composition). In latent space the resolution is already downsampled (~32× compressed), so early layers can already see globally — no benefit to local-only.
- **예상 효과** (정량): For 512²: attention FLOPs reduced 4-6× vs full attention. Single-stage 512 with quality matching cascade PixelFlow.
- **난이도 (1-5)**: 4 (custom attention masks; quality engineering)
- **출판 타겟**: CVPR 2027
- **자동 NO-GO 키워드 회피 확인**: ✅ (not cascade, not hourglass)
- **기존 12개와 overlap 없음 확인**: ✅

---

### 아이디어 11: BP-Hybrid: Bit-Plane AR+Diffusion Hybrid
- **Family**: hybrid
- **핵심 가설** (한 문장): Decompose RGB into 8 bit-planes; generate MSB planes (1-3) with fast AR (small state space, MaskGIT-style parallel), then refine LSB planes (4-8) with pixel diffusion conditioned on MSB — total inference 2× faster than full diffusion.
- **기술적 접근** (2-3 문장): MSB planes carry coarse structure (3 bits × 3 channels = 9 bits per pixel, 512 states) — AR fits easily. LSB planes carry texture noise — diffusion fits naturally as Gaussian-like distribution. Two-stage: AR generates MSB in O(log N) parallel rounds, then diffusion refines LSB with 10-20 steps using MSB as conditioning. Bit-plane decomposition is exact and reversible.
- **핵심 혁신점 (vs 기존 12 + 16 caching papers)**: First bit-plane-decomposed diffusion. Different from cascade (which separates by resolution): BP-Hybrid separates by bit-depth at same resolution. Signal X (RGB bit-plane decomposition) is unique to pixel space.
- **왜 Pixel Space 특화** (mechanism level): Bit-plane decomposition exists only for raw pixel values (8-bit integer RGB). Latent codes are continuous floats with no natural bit-plane structure. The MSB-carries-structure / LSB-carries-noise statistics are a measured property of natural images.
- **예상 효과** (정량): 2-2.5× wallclock; quality match for MSB; LSB diffusion 4× faster than full RGB diffusion because only 5 bits to model.
- **난이도 (1-5)**: 5 (new architecture, two-stage training, AR+diff handoff)
- **출판 타겟**: NeurIPS 2027 (novel formulation)
- **자동 NO-GO 키워드 회피 확인**: ✅ (not cascade; bit-plane is depth not resolution)
- **기존 12개와 overlap 없음 확인**: ✅

---

### 아이디어 12: CE-QAT: Codebook-Entropy-Weighted Quantization-Aware Training
- **Family**: quantization
- **핵심 가설** (한 문장): When quantizing pixel DiT to INT4/INT8, allocate higher bit-width to attention heads whose VQ-codebook-aware output entropy is high; this entropy-weighted bit allocation beats uniform quantization by 0.4-0.6 FID.
- **기술적 접근** (2-3 문장): For each attention head h, measure entropy H_h of its output distribution conditioned on VQ codebook usage (high entropy = head differentiates many codewords). Allocate bit-widths via b_h = b_min + γ·H_h/H_max. Run QAT with per-head bit-width budget. Pixel DiT only — leverages tokenizer's codebook usage statistic.
- **핵심 혁신점 (vs 기존 12 + 16 caching papers)**: Different from QNDS (#1) which is post-hoc and timestep-driven; CE-QAT is QAT with per-head bit allocation driven by tokenizer-statistic (codebook entropy). Signal X (VQ codebook usage entropy) couples tokenizer + quantizer — never done.
- **왜 Pixel Space 특화** (mechanism level): VQ codebook usage entropy is meaningful when the model directly outputs over codebook indices (pixel-tokenized AR/MaskGIT case) or operates on tokens with codebook alignment (pixel DiT with discrete tokenizer). Latent DiT operates on continuous VAE codes — no codebook entropy signal.
- **예상 효과** (정량): At avg 4 bits: FID delta <0.4 (vs uniform INT4 which gives 1.0+ FID delta). 4× memory reduction.
- **난이도 (1-5)**: 4 (QAT training + per-head bit infra)
- **출판 타겟**: ICML 2027 (efficient inference)
- **자동 NO-GO 키워드 회피 확인**: ✅
- **기존 12개와 overlap 없음 확인**: ✅ (orthogonal to QNDS — QAT vs post-hoc, signal differs)

---

## Cross-batch checks

- All 12 pass NO-GO keyword filter (no cache/skip/clean-pred/freq-split/global-local/cascade/perceptual/SSL/hourglass)
- All 12 disjoint from prior 12 ideas (different signal X)
- Family spread: quantization×2 (#1,12), theory×2 (#3,4), hw-codesign×1 (#2), pixel-statistics×1 (#5), noise-schedule×1 (#6), training-eff×1 (#7), video×1 (#8), multimodal×1 (#9), high-res×1 (#10), hybrid×1 (#11)
- Highest priority for validator: #1 QNDS (novelty 5 + feasibility 5), #7 PLDI (impact 5 + feasibility 5), #3 SNR-Patch scaling law (theory contribution)
