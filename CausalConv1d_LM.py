import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────
# 1. 저수준 레이어
# ─────────────────────────────────────────────────────────────
class CausalConv1d(nn.Module):
    """
    Left-padded causal 1-D convolution.
    입력  (B, C, T) → 출력 동일 shape.
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        # TODO: conv 정의 + left padding 계산
        self.left_pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: forward 로직
        x = F.pad(x, pad=(self.left_pad, 0))
        out = self.conv(x)
        return out


class CausalConvBlock(nn.Module):
    """
    [CausalConv1d → ReLU → LayerNorm] (+ Residual)
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        residual: bool = False,
    ):
        super().__init__()
        # TODO: 내부 모듈 선언
        self.residual = residual
        self.causalconv1d = CausalConv1d(channels, kernel_size)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(normalized_shape=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 블록 연산 (필요 시 residual)
        out = self.causalconv1d(x)
        out = self.act(out)
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2)
        return x + out if self.residual else out


# ─────────────────────────────────────────────────────────────
# 2. 상위 모델
# ─────────────────────────────────────────────────────────────
class CausalConvLanguageModel(nn.Module):
    """
    Autoregressive char-level LM using stacked causal convolutions.

    Workflow
    --------
    tokens (B, T) →
        Embedding →
        transpose →
        CausalConvBlock x N →
        transpose →
        Linear → logits (B, T, V)
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 8,
        kernel_size: int = 3,
        residual: bool = False,
    ):
        super().__init__()
        # TODO: embedding, blocks, head 정의
        self.embed = nn.Embedding(vocab_size, d_model)
        self.conv_blocks = nn.ModuleList(
            [
                CausalConvBlock(d_model, kernel_size=kernel_size, residual=residual) 
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Linear(d_model, vocab_size)
        pass

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : LongTensor, shape (B, T)

        Returns
        -------
        logits : Tensor, shape (B, T, vocab_size)
        """
        # TODO: forward 구현
        x = self.embed(tokens) # B T C
        x = x.transpose(1, 2) # B C T
        for block in self.conv_blocks:
            x = block(x)
        x = x.transpose(1, 2) # B T C
        logits = self.head(x)
        return logits

    # ── Convenience method ───────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_steps: int = 256,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressive sampling loop.

        Parameters
        ----------
        prompt : LongTensor, shape (B, t0)
            Initial context tokens (t0 ≥ 1).
        max_steps : int
            Number of tokens to generate *after* the prompt.
        """
        
        self.eval()                           # 드롭아웃 등 비활성화
        tokens = prompt.clone()               # (B, t0)

        for _ in range(max_steps):
            # 1) 전체 시퀀스를 넣어 마지막 step의 logits만 추출
            logits = self(tokens)             # (B, T, V)
            logits = logits[:, -1, :]         # (B, V) – 가장 최근 시점

            # 2) Temperature scaling → 확률 분포
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)  # (B, V)

            # 3) 범주형 샘플링
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # 4) 시퀀스 뒤에 붙여 다음 루프로
            tokens = torch.cat([tokens, next_token], dim=1)       # (B, T+1)

        return tokens


if __name__=="__main__":
    BATCH      = 2          # 배치 크기
    SEQ_LEN    = 64         # 학습/검증용 입력 길이
    VOCAB_SIZE = 65         # Tiny Shakespeare 예시 (a~z, etc.)
    D_MODEL    = 256
    N_LAYERS   = 8
    KERNEL     = 3
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

    dummy_tokens = torch.randint(
        low=0, high=VOCAB_SIZE, size=(BATCH, SEQ_LEN), dtype=torch.long
    ).to(DEVICE)

    model = CausalConvLanguageModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        kernel_size=KERNEL,
        residual=True,
    )

    MAX_GEN    = 128
    generated = model.generate(dummy_tokens, max_steps=MAX_GEN)
    print(f"[generate] generated shape : {generated.shape}")