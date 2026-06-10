import os
import io
import copy
import math
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# =========================
# 0. 基础配置
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

out_dir = "flow_matching_gif_outputs"
os.makedirs(out_dir, exist_ok=True)

print("Using device:", device)


# =========================
# 1. 构造真实数据分布：2D 圆环
# =========================
def sample_data(batch_size, device):
    """
    生成一个 2D 圆环分布。
    这就是我们的真实数据 x_0。
    """
    theta = torch.rand(batch_size, device=device) * 2 * math.pi
    radius = 1.5 + 0.05 * torch.randn(batch_size, device=device)

    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)

    return torch.stack([x, y], dim=1)


# =========================
# 2. 时间步正弦编码 + 速度预测网络
# =========================
def get_timestep_embedding(t, embed_dim):
    """
    将标量时间步 t ∈ [0,1] 映射为高维正弦编码。
    类似 Transformer 位置编码，让网络更好地区分不同时间步。
    """
    half_dim = embed_dim // 2
    emb = math.log(10000.0) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
    emb = t * emb
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


class ResBlock(nn.Module):
    """带 LayerNorm 的残差块"""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class VelocityNet(nn.Module):
    """
    输入:
        x_t: 当前中间状态，shape = (B, 2)
        t:   当前时间步，shape = (B, 1)

    输出:
        velocity，shape = (B, 2)

    架构:
        - 128 维正弦时间编码
        - 6 个残差块，512 隐藏维度
        - LayerNorm 稳定训练
    """

    def __init__(self, hidden_dim=512, num_blocks=6, time_embed_dim=128):
        super().__init__()
        in_dim = 2 + time_embed_dim

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x_t, t):
        t_emb = get_timestep_embedding(t, self.input_proj[0].in_features - 2)
        h = torch.cat([x_t, t_emb], dim=1)
        h = self.input_proj(h)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


# =========================
# 3. 从噪声采样：反向 flow
# =========================
@torch.no_grad()
def sample_from_noise(
    model,
    init_noise,
    num_inference_steps=100,
    return_all_steps=False,
    save_every=1,
):
    """
    从 t=1 的噪声开始，反向走到 t=0 的数据分布。

    训练时模型学的是:
        真实数据 -> 噪声 的速度

    推理时我们要:
        噪声 -> 真实数据

    所以更新公式是:
        x = x - dt * velocity
    """
    model.eval()

    x = init_noise.clone().to(device)
    dt = 1.0 / num_inference_steps

    states = []

    if return_all_steps:
        states.append(x.detach().cpu())

    for i in range(num_inference_steps):
        t_value = 1.0 - i / num_inference_steps
        t = torch.full((x.shape[0], 1), t_value, device=device)

        velocity = model(x, t)

        # 反向积分：噪声 -> 数据
        x = x - dt * velocity

        if return_all_steps and ((i + 1) % save_every == 0 or i == num_inference_steps - 1):
            states.append(x.detach().cpu())

    if return_all_steps:
        return x.detach().cpu(), states

    return x.detach().cpu()


# =========================
# 4. 把 matplotlib figure 转成 PIL Image
# =========================
def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    plt.close(fig)
    return img


# =========================
# 5. 绘制一帧散点图
# =========================
def make_scatter_frame(points, title, xlim=(-4, 4), ylim=(-4, 4)):
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(points[:, 0], points[:, 1], s=3)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.axis("equal")
    plt.title(title)
    plt.grid(alpha=0.25)
    return fig_to_pil(fig)


# =========================
# 6. 保存 GIF
# =========================
def save_gif(frames, path, duration=300):
    """
    duration 单位是毫秒。
    """
    if len(frames) == 0:
        raise ValueError("No frames to save.")

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )


# =========================
# 7. Flow Matching 训练
# =========================
model = VelocityNet(hidden_dim=512, num_blocks=6).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
# 带 warmup 的余弦退火：前 1000 步线性增长，之后余弦衰减
scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=1000)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=19000, eta_min=1e-5)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[1000])

batch_size = 2048
num_steps = 20000

# 用来观察训练过程的 checkpoint
save_steps = [
    0,
    100,
    500,
    1000,
    2000,
    5000,
    10000,
    15000,
    20000,
]

checkpoints = {}
loss_history = []

# 固定一份噪声，方便观察不同训练阶段的生成变化
fixed_init_noise = torch.randn(2000, 2, device=device)

# 保存初始模型
checkpoints[0] = copy.deepcopy(model.state_dict())

for step in range(1, num_steps + 1):
    # 真实数据 x_0
    x_0 = sample_data(batch_size, device)

    # 随机噪声 epsilon
    noise = torch.randn_like(x_0)

    # 随机时间步 t
    t = torch.rand(batch_size, 1, device=device)

    # 构造中间状态:
    # t=0 时接近真实数据
    # t=1 时接近噪声
    x_t = (1 - t) * x_0 + t * noise

    # 这条线性路径的真实速度:
    # d x_t / d t = noise - x_0
    target_velocity = noise - x_0

    # 模型预测速度
    pred_velocity = model(x_t, t)

    # Flow Matching loss
    loss = F.mse_loss(pred_velocity, target_velocity)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    loss_history.append(loss.item())

    if step % 1000 == 0:
        print(f"step {step}, loss = {loss.item():.4f}, lr = {scheduler.get_last_lr()[0]:.2e}")

    if step in save_steps:
        checkpoints[step] = copy.deepcopy(model.state_dict())


# =========================
# 8. 保存 loss 曲线
# =========================
fig = plt.figure(figsize=(6, 4))
plt.plot(loss_history)
plt.xlabel("Training Step")
plt.ylabel("MSE Loss")
plt.title("Flow Matching Training Loss")
plt.grid(alpha=0.25)

loss_img = fig_to_pil(fig)
loss_img.save(os.path.join(out_dir, "training_loss.png"))

print("Saved loss curve:", os.path.join(out_dir, "training_loss.png"))


# =========================
# 9. GIF 1：训练过程中，生成结果如何逐步变好
# =========================
training_frames = []

for step in save_steps:
    temp_model = VelocityNet().to(device)
    temp_model.load_state_dict(checkpoints[step])

    samples = sample_from_noise(
        temp_model,
        fixed_init_noise,
        num_inference_steps=100,
        return_all_steps=False,
    )

    frame = make_scatter_frame(
        samples,
        title=f"Training Progress | Step {step}",
    )

    training_frames.append(frame)

training_gif_path = os.path.join(out_dir, "training_process.gif")
save_gif(training_frames, training_gif_path, duration=450)

print("Saved training GIF:", training_gif_path)


# =========================
# 10. GIF 2：一次推理过程中，噪声如何流到圆环
# =========================
final_model = VelocityNet().to(device)
final_model.load_state_dict(checkpoints[num_steps])

# 固定一次推理噪声
inference_init_noise = torch.randn(2000, 2, device=device)

_, inference_states = sample_from_noise(
    final_model,
    inference_init_noise,
    num_inference_steps=120,
    return_all_steps=True,
    save_every=3,
)

inference_frames = []

num_frames = len(inference_states)

for idx, state in enumerate(inference_states):
    progress = idx / max(num_frames - 1, 1)

    # 对应时间从 1 -> 0
    t_value = 1.0 - progress

    frame = make_scatter_frame(
        state,
        title=f"Inference Flow | t ≈ {t_value:.2f}",
    )

    inference_frames.append(frame)

inference_gif_path = os.path.join(out_dir, "inference_flow.gif")
save_gif(inference_frames, inference_gif_path, duration=100)

print("Saved inference GIF:", inference_gif_path)


# =========================
# 11. 额外保存最终生成结果
# =========================
final_samples = sample_from_noise(
    final_model,
    torch.randn(3000, 2, device=device),
    num_inference_steps=120,
    return_all_steps=False,
)

final_frame = make_scatter_frame(
    final_samples,
    title="Final Generated Samples",
)

final_png_path = os.path.join(out_dir, "final_generated_samples.png")
final_frame.save(final_png_path)

print("Saved final sample image:", final_png_path)


# =========================
# 12. 简单提示
# =========================
print("\nDone.")
print("Output files:")
print("1.", training_gif_path)
print("2.", inference_gif_path)
print("3.", os.path.join(out_dir, "training_loss.png"))
print("4.", final_png_path)