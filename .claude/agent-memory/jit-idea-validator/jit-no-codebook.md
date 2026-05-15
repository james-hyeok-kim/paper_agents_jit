---
name: jit-no-codebook
description: JiT (2511.13720) and PixelDiT (2511.20645) are tokenizer-free / no VQ codebook — any idea premised on "JiT codebook" is invalid
metadata:
  type: project
---

JiT (arXiv 2511.13720) explicitly states "no tokenizer, no pre-training, and no extra loss" — it operates directly on raw continuous pixel patches with large-patch Transformers. PixelDiT (2511.20645) "eliminates the need for the autoencoder and learns the diffusion process directly in the pixel space." DiP (2511.18822) "bypasses VAEs" and uses patches directly.

**Why:** Multiple incoming idea proposals assume "JiT has a discrete codebook" (e.g., codebook-guided attention, ANN lookup, codebook consistency regularization). This premise is false — there is no VQ codebook to cluster, regularize, or accelerate lookup on.

**How to apply:** Any idea whose core mechanism depends on a JiT/PixelDiT "codebook," "discrete tokens," "VQ lookup," or "cluster IDs" must be flagged as 🔴 NO-GO on premise grounds before even scoring novelty/feasibility. The fix would require introducing a codebook (which would defeat the tokenizer-free advantage that is JiT's whole point) or pivoting to continuous-patch analogues (k-means on patch embeddings, etc.), which is a substantially different idea.

Related: [[validation-2026-05-15-batch12]]
