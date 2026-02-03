import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
# ===== 强制使用 7 类权重（毕设最终约束）=====
from config_runtime import WEIGHTS_7CLS, DATA_7CLS, EXPECTED_NC
from ultralytics import YOLO

_model_check = YOLO(WEIGHTS_7CLS)
print("=== RUNTIME CHECK ===")
print("weights =", WEIGHTS_7CLS)
print("nc =", _model_check.model.nc)
print("names =", _model_check.names)

assert _model_check.model.nc == EXPECTED_NC, (
    f"FATAL ERROR: expected {EXPECTED_NC} classes, "
    f"but got {_model_check.model.nc}. "
    f"Wrong weights loaded!"
)

# 后续代码统一用这个 model
model = _model_check


# ================= 1. 核心电力知识库（规则与问答用） =================
KNOWLEDGE_BASE = {
    "安全带": "高空作业必须使用安全带，遵循“高挂低用”原则，挂点应为牢固构件。",
    "验电笔": "检修前必须验电。验电笔使用前需在已知带电体上验证有效。",
    "工作服": "作业应穿着符合要求的工作服，避免易燃、易熔材质。",
    "手套": "接触带电或可能带电部件时，应按规程佩戴绝缘手套并检查破损。",
    "接地": "装设接地线应先接接地端后接导体端，拆除顺序相反。"
}

# ================= 2. 模型加载（不再硬编码类别；以 model.names 为准） =================
DEFAULT_WEIGHTS = "runs/detect/train/weights/best.pt"

@st.cache_resource
def load_model(weights_path: str):
    model = YOLO(weights_path)
    names = dict(model.names or {})
    return model, names, list(names.values())

def match(label_list, targets):
    for t in targets:
        if t in label_list:
            return t
    return None

# ================= 3. 界面配置 =================
st.set_page_config(page_title="电力安全AI专家系统", layout="wide")

with st.sidebar:
    st.title("⚙️ 系统设置")
    weights_path = st.text_input("权重路径", value=DEFAULT_WEIGHTS)
    conf = st.slider("识别阈值", 0.05, 0.95, 0.25, 0.01)
    st.caption("提示：类别名与数量以模型内置 names 为准；不在此处硬写。")

# 加载模型
model, model_names, label_list = (None, {}, [])
try:
    model, model_names, label_list = load_model(weights_path)
except Exception as e:
    st.error(f"模型加载失败：{e}")

# 自动匹配：同时兼容“5类版本/7类版本”的命名习惯
HELMET = match(label_list, ["helmet", "helmet_person", "hardhat", "safety_helmet"])
CLOTHES = match(label_list, ["uniform", "work_clothes", "workwear", "vest"])
GLOVES = match(label_list, ["gloves", "insulated_gloves", "glove"])
BELT = match(label_list, ["safety_belt", "harness", "belt"])
PEN = match(label_list, ["test_pen", "voltage_tester", "electric_pen"])
PERSON = match(label_list, ["person"])

SCENARIO_RULES = {
    "室内作业场景": {"req": [CLOTHES, GLOVES, PEN], "text": "工作服、手套、验电笔"},
    "高空作业场景": {"req": [HELMET, CLOTHES, BELT], "text": "安全帽、工作服、安全带"},
    "常规作业场景": {"req": [HELMET, CLOTHES], "text": "安全帽、工作服"},
}

# ================= 4. 登录（保持不变，但不写死演示数据） =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.title("⚡ 电力安全 AI 监测专家系统")
        st.markdown("---")
        with st.form("login"):
            u = st.text_input("管理员账号")
            p = st.text_input("访问密码", type="password")
            if st.form_submit_button("进入系统", use_container_width=True):
                if (u == "admin" and p == "admin888") or (u == "leader" and p == "123456"):
                    st.session_state.logged_in = True
                    st.session_state.current_user = u
                    st.rerun()
                else:
                    st.error("账号或密码错误")

def main_app():
    with st.sidebar:
        st.markdown("---")
        st.write(f"当前在线: **{st.session_state.current_user}**")
        menu = st.radio("系统功能项", ["运行状态", "智能合规检测", "AI 专家问答"])
        if st.button("退出系统"):
            st.session_state.logged_in = False
            st.rerun()

    if menu == "运行状态":
        st.title("📊 系统运行状态")
        st.info("本页面不展示任何未接入的数据（如天气/时延），避免不实信息。")
        if model is not None:
            st.success(f"模型加载成功：{weights_path}")
            st.write("类别列表：", model_names)
        else:
            st.warning("模型未加载。请检查权重路径。")

    elif menu == "智能合规检测":
        st.title("📷 自动化作业合规检查")
        if model is None:
            st.warning("模型未加载，无法检测。")
            return

        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("1. 场景配置")
            scenario = st.selectbox("选择作业环境", list(SCENARIO_RULES.keys()))
            st.info(f"判定标准：{SCENARIO_RULES[scenario]['text']}")
            uploaded_file = st.file_uploader("上传现场照片", type=["jpg", "png", "jpeg"])
            detect_trigger = st.button("🔍 开始检测", type="primary", use_container_width=True, disabled=uploaded_file is None)

        with c2:
            st.subheader("2. 分析报告")
            if uploaded_file:
                img = Image.open(uploaded_file)
                if detect_trigger:
                    res = model(np.array(img), conf=conf)[0]
                    detected = [model_names[int(c)] for c in res.boxes.cls]
                    current_req = [r for r in SCENARIO_RULES[scenario]["req"] if r is not None]
                    missing = [m for m in current_req if m not in detected]

                    st.image(res.plot(), caption="检测可视化结果", channels="BGR", use_container_width=True)
                    st.markdown("---")
                    if not missing:
                        st.success(f"✅ 判定结果：合规（{scenario}）")
                    else:
                        st.error("❌ 判定结果：不合规")
                        st.write("缺失项：")
                        for m in missing:
                            st.markdown(f"- ⚠️ {m}")
                else:
                    st.image(img, caption="照片预览", use_container_width=True)

    elif menu == "AI 专家问答":
        st.title("🤖 电力安全 AI 助手")
        q = st.text_input("请输入问题：", placeholder="例如：高空作业安全带怎么使用？")
        if q:
            found = False
            for key, val in KNOWLEDGE_BASE.items():
                if key in q:
                    st.success(val)
                    found = True
            if not found:
                st.info("该问题未收录到规则库中。建议查阅最新《安规》或咨询安全监督员。")

if __name__ == "__main__":
    if st.session_state.logged_in:
        main_app()
    else:
        login_page()
