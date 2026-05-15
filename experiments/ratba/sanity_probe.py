"""
RATBA STEP 1: Sanity Probe
jit-b-16 (256 tokens) vs jit-b-32 (64 tokens) 속도·품질 격차 확인

- 각 모델로 ImageNet class-conditional 256^2 샘플 1000개 생성
- 시간 측정 + FID-proxy (torch_fidelity vs fid_stats/jit_in256_stats.npz)
- 성공 기준: jit-b-32 latency >= 1.5x 빠름, FID 차이 > 1.0
"""
import sys
import os
import time
import copy
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2
import shutil

# JiT 소스 경로 추가
JIT_SRC = "/home/jovyan/workspace/Workspace_JiT"
sys.path.insert(0, JIT_SRC)
sys.path.insert(0, os.path.join(JIT_SRC, "src/torch-fidelity"))

from model_jit import JiT_models
from denoiser import Denoiser
import torch_fidelity

# ─────────────────────────────────────────────────────────────
# Denoiser 로드 헬퍼 (EMA1 가중치 사용 - canonical eval과 동일)
# ─────────────────────────────────────────────────────────────
def load_jit_model(model_name, ckpt_path, device, cfg=3.0):
    """
    model_name: e.g. 'JiT-B/16'
    ckpt_path:  checkpoint-last.pth 파일 경로
    """
    class FakeArgs:
        pass

    args = FakeArgs()
    args.model          = model_name
    args.img_size       = 256
    args.in_channels    = 3
    args.class_num      = 1000
    args.attn_dropout   = 0.0
    args.proj_dropout   = 0.0
    args.label_drop_prob = 0.1
    args.P_mean         = -1.1
    args.P_std          = 2.0
    args.t_eps          = 5e-2
    args.noise_scale    = 1.0
    args.ema_decay1     = 0.9999
    args.ema_decay2     = 0.9999
    args.sampling_method = "euler"
    args.num_sampling_steps = 10
    args.cfg            = cfg
    args.interval_min   = 0.1
    args.interval_max   = 1.0

    model = Denoiser(args)
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # EMA1 파라미터로 swap (canonical eval과 동일)
    ema_state_dict = copy.deepcopy(checkpoint['model'])
    ema1_sd        = checkpoint['model_ema1']
    for name in ema_state_dict.keys():
        if name in ema1_sd:
            ema_state_dict[name] = ema1_sd[name]

    # @torch.compile 비활성화 (타이밍 공정성 확보: 첫 warmup 이후 재컴파일 방지)
    import types
    for block in model.net.blocks:
        if hasattr(block.forward, '__wrapped__'):
            raw_fn = block.forward.__wrapped__
            block.forward = types.MethodType(raw_fn, block)
    if hasattr(model.net.final_layer.forward, '__wrapped__'):
        raw_fn = model.net.final_layer.forward.__wrapped__
        model.net.final_layer.forward = types.MethodType(raw_fn, model.net.final_layer)

    model.load_state_dict(ema_state_dict, strict=False)
    model.eval()
    model.to(device)

    # ema_params1 초기화 (generate() 호출에는 불필요, 직접 weights 로드했으므로)
    model.ema_params1 = list(model.parameters())
    model.ema_params2 = list(model.parameters())

    return model, args


def generate_samples(model, num_images, batch_size, device, save_dir, seed=42):
    """
    ImageNet class-balanced 샘플 생성, PNG 저장
    반환: 소요 시간(초)
    """
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(seed)

    class_num = 1000
    assert num_images % class_num == 0 or num_images < class_num, \
        "num_images must be divisible by class_num or < class_num"

    if num_images >= class_num:
        labels_all = np.arange(0, class_num).repeat(num_images // class_num)
    else:
        labels_all = np.arange(0, num_images)

    total = len(labels_all)
    generated = 0
    t_start = time.perf_counter()

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for i in range(0, total, batch_size):
                batch_labels = labels_all[i:i+batch_size]
                labels_t = torch.tensor(batch_labels, dtype=torch.long, device=device)
                imgs = model.generate(labels_t)

                # denormalize [-1,1] → [0,255]
                imgs = (imgs + 1) / 2
                imgs = imgs.float().cpu()

                for b_id in range(imgs.size(0)):
                    img_np = np.round(
                        np.clip(imgs[b_id].numpy().transpose(1, 2, 0) * 255, 0, 255)
                    ).astype(np.uint8)
                    img_bgr = img_np[:, :, ::-1]
                    img_id = i + b_id
                    cv2.imwrite(os.path.join(save_dir, f"{img_id:05d}.png"), img_bgr)
                    generated += 1

                print(f"  [{generated}/{total}] 생성 완료", flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start
    return elapsed


def compute_fid_and_is(sample_dir, fid_stats_file):
    """
    FID: 사전계산된 .npz 통계 vs 생성 이미지 Inception feature 통계 비교
    IS: torch_fidelity로 계산
    """
    import numpy as np
    import scipy.linalg
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Dataset
    from torchvision.datasets import ImageFolder
    from PIL import Image
    import glob

    # IS 계산 (input2=None이므로 fid 비활성화)
    metrics = torch_fidelity.calculate_metrics(
        input1=sample_dir,
        input2=None,
        cuda=True,
        isc=True,
        fid=False,
        kid=False,
        prc=False,
        verbose=False,
    )
    inception_score = metrics['inception_score_mean']

    # FID: Inception feature 추출 후 직접 계산
    from torchvision.models import inception_v3
    import torch.nn.functional as F

    transform = T.Compose([
        T.Resize(299),
        T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class ImageDirDataset(Dataset):
        def __init__(self, path, transform):
            self.files = sorted(glob.glob(os.path.join(path, "*.png")) +
                                glob.glob(os.path.join(path, "*.jpg")))
            self.transform = transform
        def __len__(self): return len(self.files)
        def __getitem__(self, idx):
            img = Image.open(self.files[idx]).convert('RGB')
            return self.transform(img)

    device = torch.device("cuda")
    import torchvision
    incep_weights = torchvision.models.Inception_V3_Weights.DEFAULT
    inception = torchvision.models.inception_v3(weights=incep_weights, transform_input=False)
    inception.fc = nn.Identity()
    inception.eval().to(device)

    loader = DataLoader(ImageDirDataset(sample_dir, transform), batch_size=64,
                        num_workers=4, pin_memory=True)
    feats = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            f = inception(batch)
            feats.append(f.cpu().float().numpy())
    feats = np.concatenate(feats, axis=0)

    mu_gen    = np.mean(feats, axis=0)
    sigma_gen = np.cov(feats, rowvar=False)

    # 사전계산 통계 로드
    ref = np.load(fid_stats_file)
    mu_ref, sigma_ref = ref['mu'], ref['sigma']

    # FID 계산
    eps = 1e-6
    diff = mu_gen - mu_ref
    covmean, _ = scipy.linalg.sqrtm(sigma_gen.dot(sigma_ref), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_gen.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma_gen + offset).dot(sigma_ref + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff.dot(diff) + np.trace(sigma_gen) + np.trace(sigma_ref) - 2 * np.trace(covmean))
    return fid, inception_score


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    CKPT_BASE  = "/data/jameskimh/james_jit_pretrained/jit-h-16"
    FID_STATS  = os.path.join(JIT_SRC, "fid_stats/jit_in256_stats.npz")
    OUT_BASE   = "/data/jameskimh/ratba"
    NUM_IMAGES = 1000   # FID-proxy (≥1K 권장)
    BATCH_SIZE = 16
    CFG        = 3.0

    results = {}

    for variant, model_name, n_tokens in [
        ("jit-b-16", "JiT-B/16", 256),
        ("jit-b-32", "JiT-B/32", 64),
    ]:
        print(f"\n{'='*60}")
        print(f"모델: {model_name}  |  tokens/image: {n_tokens}")
        ckpt_path = os.path.join(CKPT_BASE, variant, "checkpoint-last.pth")
        save_dir  = os.path.join(OUT_BASE, f"sanity_{variant}")

        # 이미 생성된 샘플이 있으면 재사용
        already_done = os.path.exists(save_dir) and len(os.listdir(save_dir)) >= NUM_IMAGES
        if already_done:
            print(f"기존 샘플 사용: {save_dir} ({len(os.listdir(save_dir))}장)")
            # 생성 시간은 로그 파일에서 추출하거나 재측정
            # 재생성으로 시간 측정 (FID만 계산 시에는 스킵)
            # 여기서는 빠른 1회 측정으로 elapsed 추정
            print(f"시간 측정을 위해 소규모 재생성 중 (16장)...")
            model, args = load_jit_model(model_name, ckpt_path, device, cfg=CFG)
            tmp_dir = save_dir + "_timing"
            t_start = time.perf_counter()
            elapsed_small = generate_samples(model, 16, 16, device, tmp_dir)
            torch.cuda.synchronize()
            elapsed = elapsed_small / 16 * NUM_IMAGES  # 외삽
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
            del model; torch.cuda.empty_cache()
        else:
            print(f"체크포인트 로드: {ckpt_path}")
            model, args = load_jit_model(model_name, ckpt_path, device, cfg=CFG)

            print(f"샘플 생성 시작 ({NUM_IMAGES}장, batch={BATCH_SIZE})...")
            elapsed = generate_samples(model, NUM_IMAGES, BATCH_SIZE, device, save_dir)
            del model; torch.cuda.empty_cache()

        sec_per_img = elapsed / NUM_IMAGES
        imgs_per_sec = NUM_IMAGES / elapsed
        print(f"생성 완료: {elapsed:.1f}s ({imgs_per_sec:.2f} img/s, {sec_per_img*1000:.1f} ms/img)")

        print("FID 계산 중...")
        fid, inception_score = compute_fid_and_is(save_dir, FID_STATS)
        print(f"FID: {fid:.4f}  IS: {inception_score:.4f}")

        results[variant] = {
            "model_name": model_name,
            "n_tokens": n_tokens,
            "num_images": NUM_IMAGES,
            "elapsed_sec": round(elapsed, 2),
            "imgs_per_sec": round(imgs_per_sec, 3),
            "ms_per_img": round(sec_per_img * 1000, 2),
            "fid": round(fid, 4),
            "inception_score": round(inception_score, 4),
            "cfg": CFG,
            "num_steps": 10,
        }

        # 메모리 해제 (이미 위에서 해제됨)
        torch.cuda.empty_cache()

    # ─── 결과 분석 ───
    print(f"\n{'='*60}")
    print("SANITY PROBE 결과")
    print(f"{'='*60}")

    b16 = results["jit-b-16"]
    b32 = results["jit-b-32"]

    speedup = b16["elapsed_sec"] / b32["elapsed_sec"]
    fid_gap = abs(b32["fid"] - b16["fid"])

    print(f"jit-b-16:  FID={b16['fid']:.4f}, {b16['ms_per_img']:.1f} ms/img")
    print(f"jit-b-32:  FID={b32['fid']:.4f}, {b32['ms_per_img']:.1f} ms/img")
    print(f"Speedup (b32 vs b16): {speedup:.3f}x")
    print(f"FID gap:              {fid_gap:.4f}")

    success_speed = speedup >= 1.5
    success_fid   = fid_gap > 1.0

    verdict = "GO" if (success_speed and success_fid) else "NO-GO"
    print(f"\n성공 기준 - 속도>=1.5x: {'PASS' if success_speed else 'FAIL'}  FID gap>1.0: {'PASS' if success_fid else 'FAIL'}")
    print(f"VERDICT: {verdict}")

    results["analysis"] = {
        "speedup_b32_vs_b16": round(speedup, 4),
        "fid_gap": round(fid_gap, 4),
        "success_speed": success_speed,
        "success_fid":   success_fid,
        "verdict": verdict,
    }

    out_json = "/home/jovyan/workspace/paper_agents_jit/experiments/ratba/sanity_probe_results.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n결과 저장: {out_json}")


if __name__ == "__main__":
    main()
