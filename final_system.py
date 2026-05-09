import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
from ui_theme import apply_light_print_theme
# ===== 使用毕设最终权重 =====
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
    "接地": "装设接地线应先接接地端后接导体端，拆除顺序相反。",
    "安全帽": "进入施工现场必须正确佩戴安全帽，帽带应系紧，禁止使用破损或超期安全帽。",
    "高空作业": "高处作业应设置可靠防坠措施，人员佩戴安全带，工具材料应防止坠落。",
    "带电作业": "带电作业应执行审批和监护制度，穿戴绝缘防护用品，保持足够安全距离。",
    "停电检修": "停电检修应按停电、验电、挂接地线、悬挂标示牌和装设遮栏的流程执行。",
    "监护人": "现场监护人应全过程监督作业行为，发现违章应立即制止并要求整改。",
    "警示牌": "危险区域应设置清晰警示牌和围栏，防止无关人员误入作业范围。",
    "绝缘鞋": "涉电作业建议穿绝缘鞋，并保持鞋底干燥完整，潮湿环境下应加强绝缘措施。",
    "绝缘杆": "使用绝缘杆前应检查外观、试验周期和清洁干燥状态，操作时保持规定握持位置。",
    "配电柜": "打开配电柜前应确认设备编号和回路状态，操作后及时关闭柜门并恢复标识。",
    "临时用电": "临时用电应做到一机一闸一漏一箱，线路架设整齐，禁止私拉乱接。",
    "触电急救": "发生触电应先切断电源或用绝缘物使伤员脱离电源，再进行呼救和急救处理。",
    "火灾": "电气火灾应优先切断电源，使用干粉或二氧化碳灭火器，禁止直接用水扑救带电设备。",
    "巡检": "巡检应关注杆塔、导线、绝缘子、接地装置和安全通道状态，并记录异常位置。",
    "安全距离": "作业人员、工具和机械与带电体之间应保持规定安全距离，不满足时应停电或采取隔离措施。",
    "违章": "常见违章包括未戴安全帽、未穿工作服、未戴手套、未验电、未系安全带和无监护作业。",
    "模型": "本系统使用 YOLO 模型识别现场安全装备，并结合场景规则给出合规或不合规判断。",
    "阈值": "识别阈值越高，结果越保守；阈值越低，召回更多目标但误检可能增加。建议演示时使用 0.25 左右。"
}

DASHBOARD_STATS = {
    "今日检测": 128,
    "合规率": "92.4%",
    "风险告警": 7,
    "在线设备": 12,
}

DASHBOARD_TREND = pd.DataFrame(
    {
        "时段": ["08:00", "10:00", "12:00", "14:00", "16:00", "18:00"],
        "检测次数": [18, 26, 19, 31, 22, 12],
        "告警次数": [1, 2, 1, 3, 0, 0],
    }
).set_index("时段")

DASHBOARD_RISKS = pd.DataFrame(
    {
        "风险类型": ["未戴安全帽", "未穿工作服", "未戴手套", "未系安全带", "未验电"],
        "次数": [3, 2, 4, 1, 2],
    }
).set_index("风险类型")

# ================= 2. 模型加载（不再硬编码类别；以 model.names 为准） =================
DEFAULT_WEIGHTS = WEIGHTS_7CLS

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
apply_light_print_theme()

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

def render_bar_rows(items, max_value=None):
    values = [value for _, value in items]
    max_value = max_value or max(values) or 1
    for label, value in items:
        width = max(6, int(value / max_value * 100))
        st.markdown(
            f"""
            <div style="margin: 10px 0;">
                <div style="display:flex; justify-content:space-between; font-size:14px;">
                    <span>{label}</span><strong>{value}</strong>
                </div>
                <div style="height:12px; border:1px solid #c7d0d9; background:#f2f4f7; border-radius:6px; overflow:hidden;">
                    <div style="height:100%; width:{width}%; background:#1f4e79;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def render_trend_rows():
    trend_items = list(zip(DASHBOARD_TREND.index.tolist(), DASHBOARD_TREND["检测次数"].tolist()))
    alarm_map = DASHBOARD_TREND["告警次数"].to_dict()
    max_value = max(value for _, value in trend_items)
    for label, value in trend_items:
        alarm = alarm_map[label]
        width = max(8, int(value / max_value * 100))
        st.markdown(
            f"""
            <div style="margin: 12px 0;">
                <div style="display:flex; justify-content:space-between; font-size:14px;">
                    <span>🕒 {label}</span><strong>检测 {value} 次 / 告警 {alarm} 次</strong>
                </div>
                <div style="height:18px; border:1px solid #c7d0d9; background:#f7f8fa; border-radius:6px; overflow:hidden;">
                    <div style="height:100%; width:{width}%; background:#1f4e79;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def render_dashboard():
    st.title("📊 电力安全监测主页看板")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔎 今日检测", DASHBOARD_STATS["今日检测"], "+18")
    c2.metric("✅ 合规率", DASHBOARD_STATS["合规率"], "+2.1%")
    c3.metric("⚠️ 风险告警", DASHBOARD_STATS["风险告警"], "-3")
    c4.metric("🟢 在线设备", DASHBOARD_STATS["在线设备"], "稳定")

    st.markdown("---")
    left, right = st.columns([2, 1])
    with left:
        st.subheader("📈 今日检测趋势")
        render_trend_rows()

    with right:
        st.subheader("🧰 系统状态")
        status_rows = [
            ("🤖 AI 模型", "已加载" if model is not None else "未加载"),
            ("🛡️ 安全帽识别", "运行中"),
            ("🧤 手套识别", "运行中"),
            ("🦺 工作服识别", "运行中"),
            ("📡 数据通道", "正常"),
        ]
        for name, state in status_rows:
            st.write(f"{name}：**{state}**")

    bottom_left, bottom_right = st.columns([1, 1])
    with bottom_left:
        st.subheader("🚨 风险类型统计")
        risk_items = list(zip(DASHBOARD_RISKS.index.tolist(), DASHBOARD_RISKS["次数"].tolist()))
        render_bar_rows(risk_items)

    with bottom_right:
        st.subheader("📌 今日巡检摘要")
        st.success("重点区域巡检完成率：96%")
        st.info("设备状态：主控端、识别端、告警端均在线")
        st.warning("待复核事项：高空作业场景 2 条，带电作业场景 1 条")
        if model is not None:
            st.caption(f"当前权重：{weights_path}")
            st.caption(f"模型类别：{', '.join(label_list)}")

def answer_question(question: str):
    hits = [(key, val) for key, val in KNOWLEDGE_BASE.items() if key in question]
    return hits[:3]

def main_app():
    with st.sidebar:
        st.markdown("---")
        st.write(f"当前在线: **{st.session_state.current_user}**")
        menu = st.radio("系统功能项", ["运行状态", "智能合规检测", "AI 专家问答"])
        if st.button("退出系统"):
            st.session_state.logged_in = False
            st.rerun()

    if menu == "运行状态":
        render_dashboard()

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
        st.caption("可提问：安全帽、安全带、带电作业、停电检修、接地、验电笔、临时用电、触电急救、火灾、模型、阈值等。")
        if q:
            hits = answer_question(q)
            if hits:
                for key, val in hits:
                    st.success(f"【{key}】{val}")
            else:
                st.info("该问题未收录到规则库中。建议查阅最新《安规》或咨询安全监督员。")

if __name__ == "__main__":
    if st.session_state.logged_in:
        main_app()
    else:
        login_page()
