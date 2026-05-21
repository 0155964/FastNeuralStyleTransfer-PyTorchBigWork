import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io
import time

# 导入刚才在 train.py 中定义的网络结构
# 这里为了能独立运行 app.py，我们把 TransformerNet 的定义也包含进来
from train import TransformerNet 

# ==========================================
# 缓存模型以加快网页加载速度
# ==========================================
@st.cache_resource
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerNet()
    # 加载状态字典
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

# ==========================================
# 推理函数：将输入图片转换为风格图
# ==========================================
def stylize(model, device, content_image):
    # 预处理：限制最大分辨率以防止显存/内存溢出
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    
    # 获取原图尺寸以便还原
    original_size = content_image.size
    
    content_tensor = transform(content_image).unsqueeze(0).to(device)

    # 推理前向传播
    with torch.no_grad():
        output_tensor = model(content_tensor)
    
    # 后处理
    output_tensor = output_tensor.cpu().clone().squeeze(0)
    output_tensor = output_tensor.clamp(0, 255).numpy()
    output_tensor = output_tensor.transpose(1, 2, 0).astype("uint8")
    
    return Image.fromarray(output_tensor)

# ==========================================
# Streamlit UI 构建
# ==========================================
st.set_page_config(page_title="AIGC 实时图像风格迁移", layout="wide")

st.title("🎨 AIGC 实时图像风格迁移系统")
st.markdown("""
本项目基于 PyTorch 实现了**快速神经风格迁移 (Fast Neural Style Transfer)**。
您可以在下方上传图片并选择特定的风格模型，系统将在**毫秒级**输出带有浓烈艺术风格的画作！
""")

st.sidebar.header("⚙️ 控制面板")

# 1. 动态允许用户上传训练好的 .pth 模型
uploaded_model = st.sidebar.file_uploader("📥 上传您的风格模型 (.pth)", type=["pth", "pt"])

# 提供一个默认的示例模型路径防崩溃（如果你已经训练好放在 models 文件夹里）
default_model_path = "./models/starry_night.pth"

if uploaded_model is not None:
    # 将上传的模型保存为临时文件供 torch 加载
    temp_model_path = "temp_model.pth"
    with open(temp_model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
    model_path_to_load = temp_model_path
    st.sidebar.success("✅ 自定义模型加载成功！")
else:
    model_path_to_load = default_model_path
    st.sidebar.info("💡 目前使用的是系统默认风格模型。")


# 2. 主界面：图片上传
uploaded_img = st.file_uploader("📸 请上传一张待转换的内容图片...", type=["jpg", "png", "jpeg"])

if uploaded_img is not None:
    content_img = Image.open(uploaded_img).convert("RGB")
    
    # 增加一个滑块，让用户自己调节风格强度
    style_strength = st.slider(
        "🎚️ 风格强度调节 (0.0: 完全保留原图 --> 1.0: 极致艺术风格)", 
        min_value=0.0, 
        max_value=1.0, 
        value=1.0, 
        step=0.05
    )
    
    # 展示两列对比
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ 原始图片")
        st.image(content_img, use_container_width=True)

    with col2:
        st.subheader("✨ AIGC 风格化结果")
        
        if st.button("🚀 一键生成", use_container_width=True):
            try:
                with st.spinner("神经网络疯狂计算中..."):
                    start_time = time.time()
                    
                    # 1. 神经网络生成 100% 浓度的纯风格图
                    model, device = load_model(model_path_to_load)
                    pure_stylized_img = stylize(model, device, content_img)
                    
                    # --- [新增代码] 确保两张图片尺寸一致 ---
                    # 强制将生成图 Resize 成与原始用户上传的 content_img 一模一样的大小
                    if pure_stylized_img.size != content_img.size:
                        pure_stylized_img = pure_stylized_img.resize(content_img.size, Image.Resampling.LANCZOS)
                    # ------------------------------------

                    # 2. 根据滑块的比例，将原图与风格图进行像素级融合
                    final_img = Image.blend(content_img, pure_stylized_img, style_strength)
                    
                    end_time = time.time()
                    
                st.image(final_img, use_container_width=True)
                st.success(f"🎉 转换完成！耗时: {end_time - start_time:.3f} 秒")
                
                # 提供下载按钮
                buf = io.BytesIO()
                final_img.save(buf, format="PNG")
                st.download_button(
                    label="💾 下载最终画作",
                    data=buf.getvalue(),
                    file_name="stylized_art.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"推理发生错误，详细报错：{e}")