"""
基础类和枚举定义
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class ExperimentPhase(Enum):
    """实验阶段枚举"""

    DATASET_CHECK = "dataset_check"
    MODEL_CHECK = "model_check"
    EVALUATION = "evaluate"


@dataclass
class PhaseResult:
    """阶段执行结果"""

    phase: ExperimentPhase
    success: bool
    attempts: int
    error_message: str = ""
    output_files: list = None

    def __post_init__(self):
        if self.output_files is None:
            self.output_files = []


class ExperimentPhaseHandler:
    """实验阶段处理器基类"""

    def __init__(self, coder, container, run_num: int, max_attempts: int):
        self.coder = coder
        self.container = container
        self.run_num = run_num
        self.max_attempts = max_attempts
        self.phase_name = ""
        self.script_name = ""
        self.expected_outputs = []

    def execute_phase(self) -> PhaseResult:
        """执行阶段 - 模板方法"""
        print(f"\n{'='*60}")
        print(f" {self.phase_name}")
        print(f"{'='*60}")

        for attempt in range(self.max_attempts + 1):
            print(f"\nAttempt {attempt + 1}/{self.max_attempts + 1}")
            print("-" * 40)

            # 执行脚本
            success, error_output = self._run_script()

            if success and self._verify_outputs():
                result = PhaseResult(
                    phase=self._get_phase_enum(),
                    success=True,
                    attempts=attempt + 1,
                    output_files=self._get_output_files(),
                )
                print(f"✅ {self.phase_name} completed successfully")
                print(f"Generated files: {result.output_files}")
                return result
            else:
                print(f"❌ {self.phase_name} failed")

                # 尝试用 Aider 修复
                if attempt < self.max_attempts and os.getenv("OPENAI_API_KEY"):
                    print(f"🔧 Fixing {self.script_name} with Aider...")

                    if self._fix_with_aider(error_output):
                        print("✅ Aider fix completed")
                    else:
                        print("❌ Aider fix failed")
                elif attempt < self.max_attempts:
                    print("⚠️  No OPENAI_API_KEY - cannot use Aider")
                    break
                else:
                    print("💀 Maximum attempts reached")

        return PhaseResult(
            phase=self._get_phase_enum(),
            success=False,
            attempts=self.max_attempts + 1,
            error_message=error_output,
        )

    def _run_script(self) -> Tuple[bool, str]:
        """运行脚本"""
        try:
            result = self.container.exec_run(
                f"python /workspace/{self.script_name}", workdir="/workspace"
            )

            output = result.output.decode("utf-8", errors="ignore")
            print(f"Exit code: {result.exit_code}")
            print(f"Output:\n{output}")

            # 检查输出中是否有错误指标
            has_execution_errors = self._has_errors(output)
            success = (result.exit_code == 0) and not has_execution_errors

            if not success:
                error_output = output
            else:
                error_output = ""

            return success, error_output

        except Exception as e:
            error_msg = f"Container execution failed: {e}"
            print(error_msg)
            return False, error_msg

    def _has_errors(self, output: str) -> bool:
        """检查输出中是否有错误"""
        error_indicators = [
            "Error:",
            "ERROR:",
            "Exception:",
            "FAILED:",
            "Failed to",
            "FileNotFoundError",
            "ModuleNotFoundError",
            "ImportError",
            "ValueError",
            "TypeError",
            "KeyError",
            "IndexError",
            "RuntimeError",
            "ConnectionError",
            "TimeoutError",
        ]

        for indicator in error_indicators:
            if indicator in output:
                print(f"🚨 Found error indicator: {indicator}")
                return True
        return False

    def _verify_outputs(self) -> bool:
        """验证输出文件是否存在"""
        if not self.expected_outputs:
            return True

        try:
            for expected_file in self.expected_outputs:
                result = self.container.exec_run(f"ls -la {expected_file}")
                if result.exit_code != 0:
                    print(f"❌ Expected output file missing: {expected_file}")
                    return False
                else:
                    print(f"✅ Found expected file: {expected_file}")
            return True
        except Exception as e:
            print(f"❌ Error checking outputs: {e}")
            return False

    def _get_output_files(self) -> List[str]:
        """获取输出文件列表"""
        output_files = []
        for expected_file in self.expected_outputs:
            try:
                result = self.container.exec_run(f"ls {expected_file}")
                if result.exit_code == 0:
                    output_files.append(expected_file)
            except Exception as e:
                print(f"❌ Error getting output files: {e}")
                continue
        return output_files

    def _fix_with_aider(self, error_output: str) -> bool:
        """使用 Aider 修复错误"""
        try:
            # 生成修复提示
            fix_prompt = self._generate_fix_prompt(error_output)

            # 运行 Aider 命令
            aider_cmd = f"""cd /workspace && echo "{fix_prompt}" | aider --no-git --yes {self.script_name}"""

            result = self.container.exec_run(["bash", "-c", aider_cmd], workdir="/workspace")

            aider_output = result.output.decode("utf-8", errors="ignore")
            print(f"Aider output:\n{aider_output}")

            return result.exit_code == 0

        except Exception as e:
            print(f"Aider execution failed: {e}")
            return False

    def _generate_fix_prompt(self, error_output: str) -> str:
        """生成修复提示 - 子类需要实现"""
        return f"Fix the error in {self.script_name}. Error: {error_output}"

    def _get_phase_enum(self) -> ExperimentPhase:
        """获取阶段枚举 - 子类需要实现"""
        raise NotImplementedError
