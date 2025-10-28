import torch
import torch.nn as nn
import torchvision.models as ptmodels
from model import snoutNet
import os


class EnsembleNet(nn.Module):
    """Ensemble of AlexNet, VGG16 and snoutNet.

    - Each member returns 2 outputs (x, y coordinates).
    - Ensemble combines outputs by (weighted) average.

    Usage:
        ensemble = EnsembleNet(device=device)
        ensemble.load_member_weights('alex', 'models/alex_BOTH_AUG.pth')
        ensemble.load_member_weights('vgg',  'models/vgg_BOTH_AUG.pth')
        ensemble.load_member_weights('snout','models/sw_BOTH_AUG.pth')
        out = ensemble(batch)

    Notes on checkpoints:
    - The helper `load_member_weights` tries several common shapes:
      - a direct state_dict saved by `torch.save(model.state_dict())`
      - a dict with a nested 'state_dict' or 'model_state' key
      - otherwise it will attempt to load the file and pass it to load_state_dict with strict=False

    """

    def __init__(self, device='cpu', combine='mean', weights=None):
        super().__init__()
        self.device = device
        self.combine = combine  # 'mean' or 'weighted'

        # AlexNet (architecture)
        self.alex = ptmodels.alexnet(weights=None)
        # replace final classifier layer -> 2 outputs
        self.alex.classifier[6] = nn.Linear(self.alex.classifier[6].in_features, 2)

        # VGG16
        self.vgg = ptmodels.vgg16(weights=None)
        self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, 2)

        # snoutNet (custom)
        self.snout = snoutNet()

        # put on device
        self.alex = self.alex.to(device)
        self.vgg = self.vgg.to(device)
        self.snout = self.snout.to(device)

        # ensemble weights (if weighted average)
        if weights is None:
            self.register_buffer('weights_tensor', torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))
        else:
            w = torch.tensor(weights, dtype=torch.float32)
            if w.numel() != 3:
                raise ValueError('weights must be length 3')
            self.register_buffer('weights_tensor', w)

    def _forward_member(self, model, x):
        # Ensure evaluation/train mode is consistent with parent
        return model(x)

    def forward(self, x):
        # x: batch of images, shape (B,3,H,W)
        out1 = self._forward_member(self.alex, x)
        out2 = self._forward_member(self.vgg, x)
        out3 = self._forward_member(self.snout, x)

        # stack outputs: (3, B, 2)
        stacked = torch.stack([out1, out2, out3], dim=0)

        if self.combine == 'mean':
            out = torch.mean(stacked, dim=0)
        elif self.combine == 'weighted':
            w = self.weights_tensor.view(-1, 1, 1)  # (3,1,1)
            ws = w * stacked
            out = torch.sum(ws, dim=0) / (w.sum())
        else:
            raise ValueError('Unknown combine method')

        return out

    def load_member_weights(self, member_name, path, strict=False):
        """Load a checkpoint into one ensemble member.

        member_name: 'alex' | 'vgg' | 'snout'
        path: path to the .pth file
        strict: pass strict to load_state_dict
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Weights file not found: {path}')

        model = None
        if member_name == 'alex':
            model = self.alex
        elif member_name == 'vgg':
            model = self.vgg
        elif member_name == 'snout':
            model = self.snout
        else:
            raise ValueError('member_name must be alex, vgg or snout')

        ckpt = torch.load(path, map_location=self.device)

        # If the checkpoint is a direct state_dict, use it.
        if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            try:
                model.load_state_dict(ckpt, strict=strict)
                return
            except RuntimeError:
                # fallthrough to other heuristics
                pass

        # Common conventions: {'state_dict': ...}, {'model_state': ...}, {'model': ...}
        for key in ('state_dict', 'model_state', 'model', 'weights'):
            if isinstance(ckpt, dict) and key in ckpt:
                sd = ckpt[key]
                if isinstance(sd, dict):
                    try:
                        model.load_state_dict(sd, strict=strict)
                        return
                    except RuntimeError:
                        # try strict=False
                        model.load_state_dict(sd, strict=False)
                        return

        # As a last resort try to load assuming checkpoint is a full model or has unexpected prefixes.
        # We'll try load_state_dict with strict=False and let PyTorch ignore mismatches.
        try:
            if isinstance(ckpt, dict):
                model.load_state_dict(ckpt, strict=False)
            else:
                # maybe a saved full model - try to load attributes
                model.load_state_dict(ckpt.state_dict(), strict=False)
            return
        except Exception as e:
            raise RuntimeError(f'Failed to load weights from {path}: {e}')


# Convenience factory
def build_ensemble_from_pths(alex_path=None, vgg_path=None, snout_path=None, device='cpu', combine='mean', weights=None):
    """Create an EnsembleNet and load provided checkpoints (if any).

    Recommended files in this repo (examples):
      - models/alex_BOTH_AUG.pth
      - models/vgg_BOTH_AUG.pth
      - models/sw_BOTH_AUG.pth    # snout weights named 'sw_*' in repo

    Pass None for any path to leave that member using randomly initialized weights.
    """
    ensemble = EnsembleNet(device=device, combine=combine, weights=weights)

    if alex_path:
        ensemble.load_member_weights('alex', alex_path)
    if vgg_path:
        ensemble.load_member_weights('vgg', vgg_path)
    if snout_path:
        ensemble.load_member_weights('snout', snout_path)

    ensemble.to(device)
    ensemble.eval()
    return ensemble


if __name__ == '__main__':
    # Quick local test example (requires dataset and images available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # default repo model paths (adjust if you have different names)
    base = os.path.join(os.path.dirname(__file__), 'models')
    alex_pth = os.path.join(base, 'alex_BOTH_AUG.pth')
    vgg_pth = os.path.join(base, 'vgg_BOTH_AUG.pth')
    snout_pth = os.path.join(base, 'sw_BOTH_AUG.pth')

    print('Building ensemble and loading weights (if files exist)...')
    e = build_ensemble_from_pths(alex_path=alex_pth if os.path.exists(alex_pth) else None,
                                vgg_path=vgg_pth if os.path.exists(vgg_pth) else None,
                                snout_path=snout_pth if os.path.exists(snout_pth) else None,
                                device=device)

    print('Ensemble ready. Example shapes:')
    x = torch.randn(2, 3, 227, 227, device=device)
    with torch.no_grad():
        y = e(x)
    print('Input batch:', x.shape)
    print('Ensemble output:', y.shape)
