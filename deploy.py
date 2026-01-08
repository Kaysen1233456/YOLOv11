"""
YOLOv11 电力安全检测部署脚本
功能：
1. 模型推理和检测
2. Web界面部署
3. 批量图片检测
4. 视频流检测
5. 实时摄像头检测

使用方法：
- 批量检测: python deploy.py --mode batch --source images/ --weights runs/detect/train/weights/best.pt
- Web界面: python deploy.py --mode web --weights runs/detect/train/weights/best.pt
- 视频检测: python deploy.py --mode video --source video.mp4 --weights runs/detect/train/weights/best.pt
"""

import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import os
import glob
from pathlib import Path
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json


class SafetyDetectionDeploy:
    def __init__(self, weights_path, conf_threshold=0.25, iou_threshold=0.45):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.class_names = {
            0: 'person',
            1: 'helmet_person',
            2: 'insulated_gloves',
            3: 'safety_belt',
            4: 'power_pole',
            5: 'voltage_tester',
            6: 'work_clothes'
        }
        self.class_colors = {
            0: (0, 255, 0),      # person - 绿色
            1: (0, 255, 255),    # helmet_person - 黄色
            2: (255, 0, 255),    # insulated_gloves - 品红
            3: (255, 255, 0),    # safety_belt - 青色
            4: (128, 0, 128),    # power_pole - 紫色
            5: (255, 165, 0),    # voltage_tester - 橙色
            6: (128, 128, 0)     # work_clothes - 棕色
        }
        
        self.load_model()
    
    def load_model(self):
        """加载模型"""
        print(f"加载模型: {self.weights_path}")
        try:
            self.model = YOLO(self.weights_path)
            print("模型加载成功！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def detect_single_image(self, image_path, save_path=None):
        """检测单张图片"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None
        
        # 进行推理
        results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold)
        
        # 绘制检测结果
        annotated_image = results[0].plot()
        
        # 保存结果
        if save_path:
            cv2.imwrite(save_path, annotated_image)
        
        # 提取检测信息
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                
                detections.append({
                    'class_id': class_id,
                    'class_name': self.class_names.get(class_id, f'unknown_{class_id}'),
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        return annotated_image, detections
    
    def batch_detect(self, source_dir, output_dir, image_extensions=['*.jpg', '*.jpeg', '*.png']):
        """批量检测图片"""
        print(f"开始批量检测: {source_dir} -> {output_dir}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有图片文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(source_dir, ext)))
            image_files.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))
        
        print(f"找到 {len(image_files)} 张图片")
        
        # 统计信息
        detection_stats = {class_id: 0 for class_id in self.class_names.keys()}
        results_log = []
        
        # 处理每张图片
        for i, image_path in enumerate(image_files):
            print(f"处理图片 {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            try:
                # 检测
                annotated_image, detections = self.detect_single_image(image_path)
                
                # 更新统计
                for detection in detections:
                    detection_stats[detection['class_id']] += 1
                
                # 保存结果
                output_path = os.path.join(output_dir, f"detect_{os.path.basename(image_path)}")
                cv2.imwrite(output_path, annotated_image)
                
                # 记录日志
                results_log.append({
                    'image_name': os.path.basename(image_path),
                    'detections': detections,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"处理 {image_path} 时出错: {e}")
        
        # 保存统计结果
        self.save_detection_stats(detection_stats, results_log, output_dir)
        
        print("批量检测完成！")
        return detection_stats, results_log
    
    def video_detect(self, video_source, output_path=None, show_fps=True):
        """视频检测"""
        print(f"开始视频检测: {video_source}")
        
        # 打开视频
        if video_source.isdigit():
            video_source = int(video_source)
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"无法打开视频源: {video_source}")
            return
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"视频信息: {width}x{height}, {fps}fps")
        
        # 设置输出视频
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 检测统计
        detection_stats = {class_id: 0 for class_id in self.class_names.keys()}
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 检测
                start_time = time.time()
                results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
                detection_time = time.time() - start_time
                
                # 绘制结果
                annotated_frame = results[0].plot()
                
                # 更新统计
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        detection_stats[class_id] += 1
                
                # 显示FPS
                if show_fps:
                    fps_text = f"FPS: {1/detection_time:.1f}"
                    cv2.putText(annotated_frame, fps_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 显示帧
                cv2.imshow('Safety Detection', annotated_frame)
                
                # 保存到输出视频
                if output_path:
                    out.write(annotated_frame)
                
                # 按'q'退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("用户中断检测")
        
        finally:
            cap.release()
            if output_path:
                out.release()
            cv2.destroyAllWindows()
        
        print(f"视频检测完成，共处理 {frame_count} 帧")
        print("检测统计:")
        for class_id, count in detection_stats.items():
            class_name = self.class_names.get(class_id, f'unknown_{class_id}')
            print(f"  {class_name}: {count} 次检测")
        
        return detection_stats
    
    def save_detection_stats(self, stats, results_log, output_dir):
        """保存检测统计结果"""
        # 保存统计信息
        stats_file = os.path.join(output_dir, 'detection_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # 保存详细日志
        log_file = os.path.join(output_dir, 'detection_log.json')
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(results_log, f, indent=2, ensure_ascii=False)
        
        # 生成统计图表
        self.generate_stats_charts(stats, output_dir)
        
        print(f"统计结果已保存至: {output_dir}")
    
    def generate_stats_charts(self, stats, output_dir):
        """生成统计图表"""
        # 准备数据
        class_names_list = [self.class_names.get(k, f'unknown_{k}') for k in stats.keys()]
        counts = list(stats.values())
        
        # 创建柱状图
        fig = px.bar(
            x=class_names_list,
            y=counts,
            labels={'x': '类别', 'y': '检测次数'},
            title='检测结果统计',
            color=counts,
            color_continuous_scale='viridis'
        )
        
        chart_path = os.path.join(output_dir, 'detection_stats.html')
        fig.write_html(chart_path)
        
        print(f"统计图表已保存至: {chart_path}")


def web_interface():
    """Streamlit Web界面"""
    st.set_page_config(
        page_title="YOLOv11 电力安全检测",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚡ YOLOv11 电力安全检测系统")
    st.markdown("基于YOLOv11的电力作业安全装备检测")
    
    # 侧边栏配置
    with st.sidebar:
        st.header("检测配置")
        weights_path = st.text_input("模型路径", value="runs/detect/train/weights/best.pt")
        conf_threshold = st.slider("置信度阈值", 0.0, 1.0, 0.25)
        iou_threshold = st.slider("IoU阈值", 0.0, 1.0, 0.45)
        
        # 加载模型
        if st.button("加载模型"):
            try:
                detector = SafetyDetectionDeploy(weights_path, conf_threshold, iou_threshold)
                st.session_state.detector = detector
                st.success("模型加载成功！")
            except Exception as e:
                st.error(f"模型加载失败: {e}")
        
        st.divider()
        st.header("功能选择")
        mode = st.selectbox("选择模式", ["图片检测", "批量检测", "视频检测"])
    
    # 主界面
    if mode == "图片检测":
        image_detection_tab()
    elif mode == "批量检测":
        batch_detection_tab()
    elif mode == "视频检测":
        video_detection_tab()


def image_detection_tab():
    """图片检测标签页"""
    st.header("🖼️ 单张图片检测")
    
    if 'detector' not in st.session_state:
        st.warning("请先在侧边栏加载模型")
        return
    
    detector = st.session_state.detector
    
    # 上传图片
    uploaded_file = st.file_uploader("选择图片", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # 显示原始图片
        image = Image.open(uploaded_file)
        st.image(image, caption="原始图片", use_column_width=True)
        
        # 检测按钮
        if st.button("开始检测"):
            with st.spinner("正在检测..."):
                # 转换为OpenCV格式
                image_np = np.array(image)
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                # 检测
                results = detector.model(image_cv, conf=detector.conf_threshold, iou=detector.iou_threshold)
                
                # 绘制结果
                annotated_image = results[0].plot()
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                
                # 显示结果
                st.image(annotated_image_rgb, caption="检测结果", use_column_width=True)
                
                # 显示检测详情
                with st.expander("检测详情"):
                    for i, result in enumerate(results):
                        boxes = result.boxes
                        st.write(f"**检测到 {len(boxes)} 个目标:**")
                        
                        for j, box in enumerate(boxes):
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            class_name = detector.class_names.get(class_id, f'unknown_{class_id}')
                            
                            st.write(f"- 目标 {j+1}: {class_name} (置信度: {confidence:.2%})")


def batch_detection_tab():
    """批量检测标签页"""
    st.header("📁 批量图片检测")
    
    if 'detector' not in st.session_state:
        st.warning("请先在侧边栏加载模型")
        return
    
    detector = st.session_state.detector
    
    # 输入输出路径
    source_dir = st.text_input("输入图片目录")
    output_dir = st.text_input("输出目录", value="detection_results")
    
    if st.button("开始批量检测"):
        if not source_dir or not os.path.exists(source_dir):
            st.error("请输入有效的输入目录")
        else:
            with st.spinner("正在批量检测..."):
                stats, results_log = detector.batch_detect(source_dir, output_dir)
            
            st.success(f"批量检测完成！共处理 {len(results_log)} 张图片")
            
            # 显示统计结果
            st.subheader("检测统计")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**各类别检测数量:**")
                for class_id, count in stats.items():
                    class_name = detector.class_names.get(class_id, f'unknown_{class_id}')
                    st.write(f"- {class_name}: {count}")
            
            with col2:
                # 绘制饼图
                fig = px.pie(
                    values=list(stats.values()),
                    names=[detector.class_names.get(k, f'unknown_{k}') for k in stats.keys()],
                    title="检测分布"
                )
                st.plotly_chart(fig, use_container_width=True)


def video_detection_tab():
    """视频检测标签页"""
    st.header("📹 视频检测")
    
    if 'detector' not in st.session_state:
        st.warning("请先在侧边栏加载模型")
        return
    
    detector = st.session_state.detector
    
    st.info("视频检测需要在本地运行，请使用命令行模式")
    st.code("python deploy.py --mode video --source video.mp4 --weights best.pt")
    
    # 参数说明
    st.subheader("参数说明")
    st.write("- `--source`: 视频文件路径或摄像头ID")
    st.write("- `--weights`: 模型权重路径")
    st.write("- `--output`: 输出视频路径（可选）")
    st.write("- `--conf`: 置信度阈值（默认0.25）")


def main():
    parser = argparse.ArgumentParser(description='YOLOv11电力安全检测部署')
    parser.add_argument('--mode', type=str, default='web', 
                       choices=['web', 'batch', 'video', 'image'],
                       help='部署模式')
    parser.add_argument('--weights', type=str, default='runs/detect/train/weights/best.pt',
                       help='模型权重路径')
    parser.add_argument('--source', type=str, help='输入源（图片/视频路径或目录）')
    parser.add_argument('--output', type=str, default='detection_results',
                       help='输出目录')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU阈值')
    
    args = parser.parse_args()
    
    if args.mode == 'web':
        web_interface()
    elif args.mode == 'batch':
        if not args.source:
            print("请提供输入目录: --source")
            return
        detector = SafetyDetectionDeploy(args.weights, args.conf, args.iou)
        detector.batch_detect(args.source, args.output)
    elif args.mode == 'video':
        if not args.source:
            print("请提供视频源: --source")
            return
        detector = SafetyDetectionDeploy(args.weights, args.conf, args.iou)
        detector.video_detect(args.source, args.output)
    elif args.mode == 'image':
        if not args.source:
            print("请提供图片路径: --source")
            return
        detector = SafetyDetectionDeploy(args.weights, args.conf, args.iou)
        annotated_image, detections = detector.detect_single_image(args.source, args.output)
        
        print("检测结果:")
        for detection in detections:
            print(f"  {detection['class_name']}: {detection['confidence']:.2%}")


if __name__ == '__main__':
    main()
