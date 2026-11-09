"""Model registry. ``build_model(name, num_classes, **overrides)``."""
from .lmiinet import LMIINet
from .unet import UNet

# Named configurations. Ablations of LMIINet are expressed as kwarg overrides.
MODEL_ZOO = {
    "lmiinet":              (LMIINet, {}),
    "lmiinet_no_cc":        (LMIINet, {"cc_mode": "off"}),       # no branch interaction
    "lmiinet_fixed_cc":     (LMIINet, {"cc_mode": "fixed"}),     # interaction, not learned
    "lmiinet_no_cru":       (LMIINet, {"use_cru": False}),
    "lmiinet_no_tf":        (LMIINet, {"tf_blocks": 0}),         # no Flatten Transformer
    "lmiinet_no_cab":       (LMIINet, {"use_cab": False}),       # plain Flatten Transformer
    "lmiinet_no_fe":        (LMIINet, {"use_fe": False}),
    "lmiinet_no_aux":       (LMIINet, {"aux_head": False}),
    "unet":                 (UNet, {"base": 32}),
    "unet_pm":              (UNet, {"base": 20}),
}


def build_model(name: str, num_classes: int = 11, **overrides):
    if name not in MODEL_ZOO:
        raise KeyError(f"Unknown model '{name}'. Choose from: {', '.join(MODEL_ZOO)}")
    cls, kw = MODEL_ZOO[name]
    kw = {**kw, **overrides}
    return cls(num_classes=num_classes, **kw)


def has_aux(name: str) -> bool:
    cls, kw = MODEL_ZOO[name]
    return cls is LMIINet and kw.get("aux_head", True)
