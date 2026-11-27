"""Fine-tune one model (one seed) in two stages: head only, then the whole network."""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from joiv.data import ImageTable, eval_transform, load_split, train_transform  # noqa: E402
from joiv.model import CamNet  # noqa: E402


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(net, loader, dev):
    net.eval()
    ys, ps = [], []
    for x, y in loader:
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            out = net(x.to(dev, non_blocking=True))
        ps.append(out.argmax(1).cpu())
        ys.append(y)
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return accuracy_score(y, p), f1_score(y, p, average="macro")


def run_stage(net, params, epochs, lr, tl, vl, dev, patience, log, best, ckpt_fn, tag):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    bad = 0
    for ep in range(epochs):
        net.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = crit(net(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(y)
            n += len(y)
        sched.step()
        acc, f1 = evaluate(net, vl, dev)
        rec = {"stage": tag, "epoch": ep, "loss": tot / n, "val_acc": acc, "val_f1": f1, "sec": time.time() - t0}
        log.append(rec)
        print(json.dumps(rec))
        if f1 > best["f1"]:
            best.update(f1=f1, acc=acc, stage=tag, epoch=ep)
            ckpt_fn()
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop in {tag}")
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--arch", required=True, help="mnv2 | mnv3l | effl0 | effb0 (or any timm name)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--head-epochs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="use only N images per split (smoke test)")
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()

    set_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
    tr, va = load_split(a.split, "train"), load_split(a.split, "val")
    if a.limit:
        tr, va = tr.sample(min(a.limit, len(tr)), random_state=0), va.sample(min(a.limit, len(va)), random_state=0)
    classes = sorted(load_split(a.split, "train")["label"].unique())

    net = CamNet(a.arch, len(classes), pretrained=not a.no_pretrained).to(dev)
    tl = DataLoader(ImageTable(tr, classes, train_transform(net.mean, net.std)), a.bs, shuffle=True,
                    num_workers=a.workers, pin_memory=True, drop_last=len(tr) > a.bs, persistent_workers=a.workers > 0)
    vl = DataLoader(ImageTable(va, classes, eval_transform(net.mean, net.std)), a.bs * 2,
                    num_workers=a.workers, pin_memory=True)

    out = Path(a.out) / f"{a.arch}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    log, best = [], {"f1": -1.0}

    def save():
        torch.save({"arch": a.arch, "timm_name": net.timm_name, "classes": classes, "mean": net.mean,
                    "std": net.std, "seed": a.seed, "state_dict": net.state_dict()}, out / "best.pt")

    for p in net.backbone.parameters():
        p.requires_grad = False
    run_stage(net, net.fc.parameters(), a.head_epochs, a.lr_head, tl, vl, dev, a.patience, log, best, save, "head")
    for p in net.backbone.parameters():
        p.requires_grad = True
    run_stage(net, net.parameters(), a.epochs, a.lr, tl, vl, dev, a.patience, log, best, save, "full")

    (out / "train_log.json").write_text(json.dumps({"args": vars(a), "best": best, "log": log}, indent=1))
    print("best:", best, "->", out / "best.pt")


if __name__ == "__main__":
    main()
