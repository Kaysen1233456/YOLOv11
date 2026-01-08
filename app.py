import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# --- 1. 配置区域：这是你的“大脑”规则 ---
# 这里的路径是你刚才确认过的最新模型路径
MODEL_PATH = "/mnt/workspace/project/power_safety_detection/runs/detect/my_grad_project_A10_rescue2/weights/best.pt"

# --- 2. 定义场景规则 (关键修改) ---
# 格式： "场景名称": ["必须具备的标签1", "必须具备的标签2", ...]
# 请根据你实际训练的标签名字(英文)修改下面列表里的内容
SCENARIO_RULES = {
    "场景一：普通巡检 (General Inspection)": ["helmet_person", "work_clothes"],
    "场景二：带电作业 (Live Working)": ["helmet_person", "work_clothes", "insulated_gloves"],
    "场景三：登高作业 (Climbing Work)": ["helmet_person", "safety_belt"] # 如果你没训练safety_belt，把这个删掉
}

# 页面基础设置
st.set_page_config(page_title="电力安全智能判官", layout="wide")
st.title("🛡️ 电力施工安全合规检测系统")

# 侧边栏：选择场景
st.sidebar.header("🕹️ 场景模拟设置")
selected_scenario = st.sidebar.radio(
    "请选择当前施工场景：",
    list(SCENARIO_RULES.keys())
)

# 显示当前场景的要求
required_items = SCENARIO_RULES[selected_scenario]
st.sidebar.info(f"**{selected_scenario}**\n\n必须检测到以下装备才算合格：\n" + "\n".join([f"- {item}" for item in required_items]))

# 加载模型函数
@st.cache_resource
def load_model():
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        return None

model = load_model()

if model is None:
    st.error(f"❌ 错误：找不到模型文件！路径：{MODEL_PATH}")
else:
    # 上传区域
    uploaded_file = st.file_uploader("📸 上传现场照片进行合规性审查...", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        # 1. 打开图片
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption='现场原图', use_container_width=True)

        # 2. 运行检测按钮
        if st.button('⚖️ 开始合规性判决', type="primary"):
            with st.spinner('AI 正在识别装备并根据场景规则进行审核...'):
                try:
                    # YOLO 推理
                    results = model.predict(image, conf=0.25)
                    
                    # --- 核心逻辑：获取检测到的所有物体 ---
                    detected_classes = set()
                    for c in results[0].boxes.cls:
                        class_name = model.names[int(c)]
                        detected_classes.add(class_name)
                    
                    # --- 核心逻辑：合规判断 ---
                    # 检查是否所有“必须物品”都在“检测结果”里
                    missing_items = []
                    for item in required_items:
                        if item not in detected_classes:
                            missing_items.append(item)
                    
                    is_safe = (len(missing_items) == 0)

                    # --- 结果可视化 ---
                    res_plotted = results[0].plot()
                    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                    
                    with col2:
                        st.image(res_rgb, caption='AI 识别结果', use_container_width=True)

                    # --- 最终判决书 ---
                    st.divider()
                    st.subheader("📋 判决报告")
                    
                    # 显示检测到的所有物品
                    st.write(f"🔍 **AI 实际检测到的物品：** {', '.join(detected_classes) if detected_classes else '无'}")
                    
                    if is_safe:
                        st.success(f"✅ **审核通过：安全**\n\n作业人员符合【{selected_scenario}】的着装规范。")
                        st.balloons()
                    else:
                        st.error(f"⛔ **审核不通过：违规！**\n\n作业人员违反【{selected_scenario}】规范。")
                        st.warning(f"⚠️ **缺失装备：** {', '.join(missing_items)}")

                except Exception as e:
                    st.error(f"系统运行出错: {e}")
