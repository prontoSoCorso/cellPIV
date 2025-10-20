# --- Adapter to normalize incoming tensor shapes before forwarding to original model ---
import torch

class ModelAdapter(torch.nn.Module):
    """
    Wrap a model so that inputs with extra singleton dims (or 5D) are collapsed
    into shape (B, C, T) expected by ConvTran.forward (which itself does x.unsqueeze(1)).
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        # x is a torch.Tensor (batched)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)

        # If 5D (e.g. [B, 1, 1, 1, T]) collapse middle singleton dims into channels:
        if x.dim() == 5:
            B = x.shape[0]
            T = x.shape[-1]
            # collapse dims 1..-2 into channels -> (B, C, T)
            x = x.view(B, -1, T)

        # If 4D but has an extra spatial singleton (e.g. [B, 1, 1, T]) -> squeeze that axis
        elif x.dim() == 4:
            # cases like (B, C, 1, T) are fine (C may already be >1) -> do nothing
            # if second dim is 1 and third dim is 1 -> (B,1,1,T) -> squeeze to (B,1,T)
            if x.shape[1] == 1 and x.shape[2] == 1:
                x = x.squeeze(2)  # result (B,1,T)
            # if x is (B,1,H,W) and H>1 -> leave it; model might not expect that

        # If 3D (B, C, T) OK; if 2D (C, T) convert to batch (1,C,T)
        elif x.dim() == 3:
            pass
        elif x.dim() == 2:
            x = x.unsqueeze(0)

        # Finally forward to base model
        return self.base_model(x)
