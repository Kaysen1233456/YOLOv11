"""
PyTorch环境修复脚本
功能：
1. 检测当前环境问题
2. 提供修复建议
3. 自动修复常见问题
4. 验证修复结果

使用方法：python fix_pytorch_env.py
"""

import sys
import subprocess
import platform
import os
import json
from pathlib import Path


class PyTorchEnvFixer:
    def __init__(self):
        self.system = platform.system()
        self.python_version = sys.version_info
        self.problems = []
        self.solutions = []
    
    def detect_problems(self):
        """检测环境问题"""
        print("🔍 检测PyTorch环境问题...")
        
        # 1. 检测PyTorch是否能正常导入
        try:
            import torch
            print("✓ PyTorch导入成功")
            
            # 检测CUDA可用性
            if torch.cuda.is_available():
                print(f"✓ CUDA可用: {torch.cuda.get_device_name(0)}")
                print(f"  CUDA版本: {torch.version.cuda}")
                print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            else:
                print("⚠️ CUDA不可用，将使用CPU")
                
        except Exception as e:
            self.problems.append({
                'type': 'torch_import',
                'error': str(e),
                'description': 'PyTorch导入失败'
            })
            print(f"✗ PyTorch导入失败: {e}")
        
        # 2. 检测Ultralytics是否能正常导入
        try:
            import ultralytics
            print("✓ Ultralytics导入成功")
        except Exception as e:
            self.problems.append({
                'type': 'ultralytics_import',
                'error': str(e),
                'description': 'Ultralytics导入失败'
            })
            print(f"✗ Ultralytics导入失败: {e}")
        
        # 3. 检测OpenCV是否能正常导入
        try:
            import cv2
            print(f"✓ OpenCV导入成功 (版本: {cv2.__version__})")
        except Exception as e:
            self.problems.append({
                'type': 'opencv_import',
                'error': str(e),
                'description': 'OpenCV导入失败'
            })
            print(f"✗ OpenCV导入失败: {e}")
        
        # 4. 检测Streamlit是否能正常导入
        try:
            import streamlit
            print(f"✓ Streamlit导入成功 (版本: {streamlit.__version__})")
        except Exception as e:
            self.problems.append({
                'type': 'streamlit_import',
                'error': str(e),
                'description': 'Streamlit导入失败'
            })
            print(f"✗ Streamlit导入失败: {e}")
        
        # 5. 检测虚拟环境
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("✓ 虚拟环境检测正常")
        else:
            self.problems.append({
                'type': 'virtual_env',
                'error': 'Not in virtual environment',
                'description': '建议使用虚拟环境'
            })
            print("⚠️ 未检测到虚拟环境")
        
        # 6. 检测依赖包版本兼容性
        self.check_dependencies_compatibility()
        
        return len(self.problems) == 0
    
    def check_dependencies_compatibility(self):
        """检查依赖包版本兼容性"""
        try:
            import torch
            import ultralytics
            
            # 检查PyTorch版本兼容性
            torch_version = torch.__version__
            print(f"PyTorch版本: {torch_version}")
            
            # 检查Ultralytics版本
            uv_version = ultralytics.__version__
            print(f"Ultralytics版本: {uv_version}")
            
            # 版本兼容性检查
            if torch_version.startswith('2.'):
                print("✓ PyTorch 2.x版本兼容")
            else:
                self.problems.append({
                    'type': 'torch_version',
                    'error': f'PyTorch {torch_version} may have compatibility issues',
                    'description': 'PyTorch版本可能不兼容'
                })
                
        except:
            pass
    
    def generate_solutions(self):
        """生成解决方案"""
        print("\n🛠️ 生成修复方案...")
        
        for problem in self.problems:
            problem_type = problem['type']
            
            if problem_type == 'torch_import':
                solution = self.fix_torch_import()
            elif problem_type == 'ultralytics_import':
                solution = self.fix_ultralytics_import()
            elif problem_type == 'opencv_import':
                solution = self.fix_opencv_import()
            elif problem_type == 'streamlit_import':
                solution = self.fix_streamlit_import()
            elif problem_type == 'virtual_env':
                solution = self.fix_virtual_env()
            elif problem_type == 'torch_version':
                solution = self.fix_torch_version()
            else:
                solution = self.fix_generic()
            
            self.solutions.append(solution)
    
    def fix_torch_import(self):
        """修复PyTorch导入问题"""
        return {
            'description': '重新安装PyTorch',
            'commands': [
                'pip uninstall -y torch torchvision torchaudio',
                self.get_torch_install_command()
            ],
            'manual_steps': [
                '如果自动安装失败，请访问 https://pytorch.org/get-started/locally/ 手动安装'
            ]
        }
    
    def fix_ultralytics_import(self):
        """修复Ultralytics导入问题"""
        return {
            'description': '安装Ultralytics',
            'commands': [
                'pip install ultralytics --upgrade'
            ]
        }
    
    def fix_opencv_import(self):
        """修复OpenCV导入问题"""
        return {
            'description': '安装OpenCV',
            'commands': [
                'pip install opencv-python --upgrade'
            ]
        }
    
    def fix_streamlit_import(self):
        """修复Streamlit导入问题"""
        return {
            'description': '安装Streamlit',
            'commands': [
                'pip install streamlit --upgrade'
            ]
        }
    
    def fix_virtual_env(self):
        """修复虚拟环境问题"""
        return {
            'description': '创建虚拟环境',
            'commands': [
                'python -m venv yolov11_env',
                'yolov11_env\\Scripts\\activate' if self.system == 'Windows' else 'source yolov11_env/bin/activate',
                'pip install -r requirements.txt'
            ],
            'manual_steps': [
                '建议在项目根目录创建虚拟环境'
            ]
        }
    
    def fix_torch_version(self):
        """修复PyTorch版本问题"""
        return {
            'description': '升级PyTorch到兼容版本',
            'commands': [
                'pip install torch torchvision torchaudio --upgrade'
            ]
        }
    
    def fix_generic(self):
        """通用修复方案"""
        return {
            'description': '通用环境修复',
            'commands': [
                'pip install --upgrade pip',
                'pip install -r requirements.txt --force-reinstall'
            ]
        }
    
    def get_torch_install_command(self):
        """获取适合的PyTorch安装命令"""
        if self.system == 'Windows':
            # Windows系统
            if self.python_version >= (3, 8):
                return 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'
            else:
                return 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117'
        else:
            # Linux/Mac系统
            return 'pip install torch torchvision torchaudio'
    
    def apply_solutions(self):
        """应用解决方案"""
        if not self.solutions:
            print("没有需要修复的问题")
            return True
        
        print(f"\n🔧 开始应用 {len(self.solutions)} 个修复方案...")
        
        for i, solution in enumerate(self.solutions):
            print(f"\n[{i+1}/{len(self.solutions)}] {solution['description']}")
            
            # 执行命令
            for command in solution.get('commands', []):
                print(f"  执行: {command}")
                try:
                    result = subprocess.run(
                        command, 
                        shell=True, 
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    print(f"  ✓ 成功: {result.stdout.strip()[:100]}...")
                except subprocess.CalledProcessError as e:
                    print(f"  ✗ 失败: {e}")
                    if not self.ask_continue():
                        return False
            
            # 显示手动步骤
            for step in solution.get('manual_steps', []):
                print(f"  📝 手动步骤: {step}")
        
        return True
    
    def ask_continue(self):
        """询问是否继续"""
        response = input("\n  是否继续修复其他问题？(y/n): ").lower().strip()
        return response in ['y', 'yes', '是']
    
    def verify_fixes(self):
        """验证修复结果"""
        print("\n✅ 验证修复结果...")
        
        success_count = 0
        total_tests = 0
        
        # 重新检测问题
        try:
            import torch
            total_tests += 1
            print("✓ PyTorch导入测试通过")
            
            if torch.cuda.is_available():
                print(f"✓ CUDA测试通过: {torch.cuda.get_device_name(0)}")
            else:
                print("✓ CPU模式测试通过")
            
            success_count += 1
        except Exception as e:
            print(f"✗ PyTorch测试失败: {e}")
        
        try:
            import ultralytics
            total_tests += 1
            print("✓ Ultralytics导入测试通过")
            success_count += 1
        except Exception as e:
            print(f"✗ Ultralytics测试失败: {e}")
        
        try:
            import cv2
            total_tests += 1
            print(f"✓ OpenCV导入测试通过 (版本: {cv2.__version__})")
            success_count += 1
        except Exception as e:
            print(f"✗ OpenCV测试失败: {e}")
        
        try:
            import streamlit
            total_tests += 1
            print(f"✓ Streamlit导入测试通过 (版本: {streamlit.__version__})")
            success_count += 1
        except Exception as e:
            print(f"✗ Streamlit测试失败: {e}")
        
        return success_count, total_tests
    
    def generate_report(self, success_count, total_tests):
        """生成修复报告"""
        report = {
            'timestamp': str(datetime.now()),
            'system': self.system,
            'python_version': f"{self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}",
            'problems_found': len(self.problems),
            'solutions_applied': len(self.solutions),
            'tests_passed': success_count,
            'tests_total': total_tests,
            'success_rate': f"{success_count/total_tests*100:.1f}%" if total_tests > 0 else "0%"
        }
        
        # 保存报告
        report_path = Path(__file__).parent / 'env_fix_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 显示报告
        print("\n" + "="*60)
        print("🔧 环境修复报告")
        print("="*60)
        print(f"系统: {report['system']}")
        print(f"Python版本: {report['python_version']}")
        print(f"发现问题: {report['problems_found']} 个")
        print(f"应用方案: {report['solutions_applied']} 个")
        print(f"测试通过: {report['tests_passed']}/{report['tests_total']}")
        print(f"成功率: {report['success_rate']}")
        
        if success_count == total_tests:
            print("\n🎉 环境修复成功！可以开始训练了！")
        else:
            print("\n⚠️ 部分问题仍未解决，请查看详细报告")
        
        print(f"详细报告保存在: {report_path}")
        print("="*60)
        
        return success_count == total_tests


def main():
    print("🔧 YOLOv11 电力安全检测 - 环境修复工具")
    print("="*60)
    
    fixer = PyTorchEnvFixer()
    
    # 检测问题
    is_clean = fixer.detect_problems()
    
    if is_clean:
        print("\n✅ 环境检测正常，无需修复！")
        return
    
    # 生成解决方案
    fixer.generate_solutions()
    
    # 显示修复方案
    print(f"\n📋 发现 {len(fixer.problems)} 个问题，准备应用 {len(fixer.solutions)} 个修复方案：")
    for i, solution in enumerate(fixer.solutions, 1):
        print(f"  {i}. {solution['description']}")
    
    # 询问是否修复
    response = input("\n是否开始自动修复？(y/n): ").lower().strip()
    if response not in ['y', 'yes', '是']:
        print("取消修复")
        return
    
    # 应用修复
    if fixer.apply_solutions():
        # 验证修复
        success_count, total_tests = fixer.verify_fixes()
        
        # 生成报告
        fixer.generate_report(success_count, total_tests)
    else:
        print("修复过程中断")


if __name__ == '__main__':
    from datetime import datetime
    main()
