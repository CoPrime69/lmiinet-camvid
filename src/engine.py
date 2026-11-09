"""Evaluation loop and checkpoint loading, shared by train.py, evaluate.py and the figure code."""
import torch
import torch.nn.functional as F

from .data import NUM_CLASSES
from .models import build_model
from .utils import ConfusionMatrix


@torch.no_grad()
def evaluate_model(model, loader, device, use_amp=False):
    """mIoU, pixel accuracy, per-class IoU and the raw confusion matrix over a loader.

    Logits are resized to the label resolution, so a model evaluated on up-sampled inputs is
    still scored against the native 360x480 labels.
    """
    model.eval()
    cm = ConfusionMatrix(NUM_CLASSES, device)
    amp_dev = "cuda" if device.type == "cuda" else "cpu"
    for imgs, masks, _ in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.autocast(device_type=amp_dev, enabled=use_amp and device.type == "cuda"):
            logits = model(imgs)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits.float(), size=masks.shape[-2:], mode="bilinear",
                                   align_corners=False)
        cm.update(logits.argmax(1), masks)
    return cm.compute()


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["model_name"], ck.get("num_classes", NUM_CLASSES)).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), ck


class GraphTrainer:
    """One optimisation step = forward + loss + backward + Adam, optionally replayed as a CUDA graph.

    LMIINet is made of hundreds of small kernels, so in eager PyTorch the step time is dominated by
    the CPU launching them (about 85 us each on this Windows laptop), not by GPU arithmetic.
    Capturing the whole step once and replaying it removes that overhead while executing exactly
    the same kernels. Requirements met here: fixed shapes (drop_last crops), bf16 autocast (no loss
    scaler, so no host sync), Adam with capturable=True and the learning rate held in a GPU tensor
    that is updated in place.

    The first ``warmup`` steps run eagerly on a side stream (they are real training steps); the next
    step is captured and immediately replayed. A capture failure raises (the CUDA context cannot be
    reused afterwards); run.py then repeats that run with --no-cuda-graph.
    """

    def __init__(self, model, criterion, params, lr, weight_decay, aux_weight=0.3,
                 amp_dtype=torch.bfloat16, use_graph=True, warmup=3, device="cuda"):
        self.model, self.criterion, self.aux_weight = model, criterion, aux_weight
        self.device = torch.device(device)
        self.amp_dtype = amp_dtype
        self.use_graph = use_graph and self.device.type == "cuda"
        self.lr = torch.tensor(float(lr), device=self.device)
        cuda = self.device.type == "cuda"
        self.opt = torch.optim.Adam(params, lr=self.lr if cuda else float(lr), betas=(0.9, 0.999),
                                    weight_decay=weight_decay, capturable=cuda)
        self.warmup = warmup
        self.steps = 0
        self.graph = None
        self.static_x = self.static_y = self.static_loss = None
        self.loss_sum = torch.zeros((), device=self.device)
        self.loss_count = 0
        self.mode = "graph" if self.use_graph else "eager"
        if self.use_graph:
            # cuDNN autotuning benchmarks kernels on first use of a new configuration, which is not
            # allowed inside a capture; heuristic kernel choice keeps capture valid.
            torch.backends.cudnn.benchmark = False

    def _loss(self, x, y):
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                            enabled=self.amp_dtype is not None, cache_enabled=False):
            out = self.model(x)
            if isinstance(out, tuple):
                return self.criterion(out[0], y) + self.aux_weight * self.criterion(out[1], y)
            return self.criterion(out, y)

    def _eager_step(self, x, y):
        self.opt.zero_grad(set_to_none=True)
        loss = self._loss(x, y)
        loss.backward()
        self.opt.step()
        return loss.detach()

    def set_lr(self, lr):
        for g in self.opt.param_groups:
            if isinstance(g["lr"], torch.Tensor):
                g["lr"].fill_(lr)            # in place: the captured graph reads this tensor
            else:
                g["lr"] = lr

    def step(self, x, y):
        self.model.train()
        if not self.use_graph:
            loss = self._eager_step(x, y)
        elif self.graph is None and self.steps < self.warmup:
            if self.static_x is None:
                self.side = torch.cuda.Stream()
            self.side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.side):
                loss = self._eager_step(x, y)
            torch.cuda.current_stream().wait_stream(self.side)
            self.static_x = torch.empty_like(x)
            self.static_y = torch.empty_like(y)
        elif self.graph is None:
            try:
                self.static_x.copy_(x)
                self.static_y.copy_(y)
                self.opt.zero_grad(set_to_none=True)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self.static_loss = self._loss(self.static_x, self.static_y)
                    self.static_loss.backward()
                    self.opt.step()
                self.graph = g
                g.replay()                      # capture records only; replay actually trains on x
                loss = self.static_loss.detach()
            except Exception as e:  # noqa: BLE001
                # A failed capture leaves the CUDA context unusable, so stop; run.py retries the run
                # with --no-cuda-graph.
                raise RuntimeError(f"CUDA graph capture failed: {type(e).__name__}: {e}. "
                                   "Re-run with --no-cuda-graph.") from e
        else:
            self.static_x.copy_(x, non_blocking=True)
            self.static_y.copy_(y, non_blocking=True)
            self.graph.replay()
            loss = self.static_loss.detach()
        self.steps += 1
        self.loss_sum += loss.float()
        self.loss_count += 1

    def pop_mean_loss(self):
        v = (self.loss_sum / max(self.loss_count, 1)).item()
        self.loss_sum.zero_()
        self.loss_count = 0
        return v
