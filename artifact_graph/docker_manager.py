"""
Docker container manager - responsible for Docker container creation, configuration and management
"""

import os
from typing import Any, Dict, List, Optional

import docker
import docker.types


class DockerManager:
    """Docker container manager"""

    def __init__(
        self,
        image_name: str = "simple-coder:latest",
        memory_limit: str = "8g",
        enable_gpu: bool = True,
    ):
        self.image_name = image_name
        self.memory_limit = memory_limit
        self.enable_gpu = enable_gpu
        self.docker_client = None
        self.container = None
        self.installed_packages = set()  # 缓存已安装的包

    def create_container(
        self, output_dir: str, environment_vars: Optional[Dict[str, str]] = None
    ) -> Optional[Any]:
        """Create and start Docker container"""
        try:
            # 先创建输出目录，确保是当前用户权限
            os.makedirs(output_dir, exist_ok=True)

            # 初始化Docker客户端
            self.docker_client = docker.from_env()

            # 准备环境变量
            env_vars = self._prepare_environment_vars(environment_vars)

            # 准备容器配置
            container_config = {
                "image": self.image_name,
                "command": "sleep infinity",
                "detach": True,
                "remove": False,
                "working_dir": "/workspace",
                "volumes": {os.path.abspath(output_dir): {"bind": "/workspace", "mode": "rw"}},
                "mem_limit": self.memory_limit,
                "environment": env_vars,
            }

            # 添加GPU支持（如果启用）
            if self.enable_gpu:
                container_config["device_requests"] = [
                    docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                ]

            # 创建并启动容器
            print(f"🐳 Creating Docker container ({self.image_name})...")
            self.container = self.docker_client.containers.run(**container_config)

            print("✅ Docker container created successfully")
            return self.container

        except Exception as e:
            print(f"❌ Failed to create Docker container: {e}")
            return None

    def _prepare_environment_vars(
        self, custom_vars: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """准备环境变量"""
        default_vars = {
            "PYTHONPATH": "/workspace",
            "HF_TOKEN": os.getenv("HF_TOKEN", ""),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
        }

        if custom_vars:
            default_vars.update(custom_vars)

        return default_vars

    def execute_command(
        self, command: str, workdir: str = "/workspace", timeout: Optional[int] = None
    ) -> tuple[int, str]:
        """Execute command in container"""
        if not self.container:
            raise RuntimeError("Container not created. Call create_container() first.")

        try:
            result = self.container.exec_run(command, workdir=workdir, stdout=True, stderr=True)

            output = result.output.decode("utf-8", errors="ignore")
            return result.exit_code, output

        except Exception as e:
            return -1, f"Command execution failed: {e}"

    def execute_script(self, script_name: str, timeout: int = 300) -> tuple[int, str]:
        """Execute Python script with timeout control"""
        # Set specific timeouts for different script types
        if script_name == "metric_check.py":
            timeout = 600  # 10 minutes for evaluation
        elif script_name == "dataset_check.py":
            timeout = 900  # 15 minutes for dataset download/analysis
        elif script_name == "model_check.py":
            timeout = 600  # 10 minutes for model checking

        print(f"🔄 Executing {script_name} with {timeout}s timeout...")

        try:
            import threading

            # Create a result container
            result_container = {"result": None, "error": None}

            def run_script():
                try:
                    result_container["result"] = self.container.exec_run(
                        f"python /workspace/{script_name}",
                        workdir="/workspace",
                        stdout=True,
                        stderr=True,
                    )
                except Exception as e:
                    result_container["error"] = e

            # Start script execution in a separate thread
            thread = threading.Thread(target=run_script)
            thread.daemon = True
            thread.start()

            # Wait for completion or timeout
            thread.join(timeout)

            if thread.is_alive():
                # Timeout occurred
                print(f"⏰ Script {script_name} timed out after {timeout}s")
                try:
                    # Try to stop the container execution (best effort)
                    self.container.exec_run("pkill -f python", workdir="/workspace")
                except Exception as e:
                    print(f"❌ Error stopping container: {e}")
                    pass
                return -1, f"Script execution timed out after {timeout} seconds"

            # Check if execution completed successfully
            if result_container["error"]:
                raise result_container["error"]

            if result_container["result"] is None:
                return -1, "Script execution failed - no result returned"

            result = result_container["result"]
            output = result.output.decode("utf-8", errors="ignore")

            # Check for results.json after metric_check.py
            if script_name == "metric_check.py" and result.exit_code == 0:
                if not self.check_file_exists("/workspace/results.json"):
                    output += "\n⚠️ Script completed but results.json missing"
                    return 1, output

            return result.exit_code, output

        except Exception as e:
            return -1, f"Script execution failed: {e}"

    def read_file(self, file_path: str) -> tuple[int, str]:
        """读取容器内文件内容"""
        return self.execute_command(f"cat {file_path}")

    def check_file_exists(self, file_path: str) -> bool:
        """检查文件是否存在"""
        exit_code, _ = self.execute_command(f"ls {file_path}")
        return exit_code == 0

    def install_packages(self, packages: List[str]) -> bool:
        """安装Python包（带缓存，避免重复安装）"""
        # 过滤出未安装的包
        new_packages = [pkg for pkg in packages if pkg not in self.installed_packages]

        if not new_packages:
            print(f"📦 All packages already installed: {', '.join(packages)}")
            return True

        print(f"📦 Installing new packages: {', '.join(new_packages)}")

        for package in new_packages:
            try:
                exit_code, output = self.execute_command(f"pip install {package}")
                if exit_code == 0:
                    print(f"✅ {package} installed")
                    self.installed_packages.add(package)
                else:
                    print(f"⚠️ {package} installation failed: {output}")
            except Exception as e:
                print(f"⚠️ Failed to install {package}: {e}")

        return True

    def run_aider_fix(self, script_name: str, error_output: str) -> bool:
        """Use Aider to fix script"""
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️ No OPENAI_API_KEY - cannot use Aider")
            return False

        try:
            print(f"🔧 Using Aider to fix {script_name}...")

            fix_prompt = (
                f"Fix the error in {script_name}. Error after running this script: {error_output}"
            )
            aider_cmd = (
                f"""cd /workspace && echo "{fix_prompt}" | aider --no-git --yes {script_name}"""
            )

            exit_code, output = self.execute_command(["bash", "-c", aider_cmd])

            if exit_code == 0:
                print("✅ Aider fix completed")
                return True
            else:
                print(f"❌ Aider fix failed: {output}")
                return False

        except Exception as e:
            print(f"❌ Aider execution failed: {e}")
            return False

    def cleanup(self):
        """Clean up container resources"""
        if self.container:
            try:
                print("🧹 Cleaning up container...")
                self.container.stop()
                self.container.remove()
                print("✅ Container cleaned up")
            except Exception as e:
                print(f"⚠️ Error during cleanup: {e}")
            finally:
                self.container = None

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()

    @property
    def is_ready(self) -> bool:
        """检查容器是否就绪"""
        return self.container is not None

    def get_container_info(self) -> Dict[str, Any]:
        """获取容器信息"""
        if not self.container:
            return {}

        try:
            container_info = self.container.attrs
            return {
                "id": container_info.get("Id", "")[:12],
                "name": container_info.get("Name", "").lstrip("/"),
                "status": container_info.get("State", {}).get("Status", "unknown"),
                "image": self.image_name,
                "memory_limit": self.memory_limit,
                "gpu_enabled": self.enable_gpu,
            }
        except Exception:
            return {}


class DockerConfig:
    """Docker配置类"""

    # 默认配置
    DEFAULT_IMAGE = "simple-coder:latest"
    DEFAULT_MEMORY_LIMIT = "32g"

    # GPU配置
    GPU_ENABLED = True
    CUDA_VERSION = "11.8"

    # 环境变量配置
    REQUIRED_ENV_VARS = ["OPENAI_API_KEY", "HF_TOKEN", "ANTHROPIC_API_KEY"]

    @classmethod
    def create_manager(
        cls,
        output_dir: str,
        image_name: Optional[str] = None,
        memory_limit: Optional[str] = None,
        enable_gpu: Optional[bool] = None,
        **kwargs,
    ) -> DockerManager:
        """创建配置好的Docker管理器"""

        manager = DockerManager(
            image_name=image_name or cls.DEFAULT_IMAGE,
            memory_limit=memory_limit or cls.DEFAULT_MEMORY_LIMIT,
            enable_gpu=enable_gpu if enable_gpu is not None else cls.GPU_ENABLED,
        )

        return manager

    @classmethod
    def check_environment(cls) -> Dict[str, bool]:
        """检查环境变量和依赖"""
        status = {}

        # 检查环境变量
        for var in cls.REQUIRED_ENV_VARS:
            status[f"env_{var}"] = bool(os.getenv(var))

        # 检查Docker
        try:
            docker_client = docker.from_env()
            docker_client.ping()
            status["docker_available"] = True
        except Exception:
            status["docker_available"] = False

        # 检查GPU（如果启用）
        if cls.GPU_ENABLED:
            try:
                docker_client = docker.from_env()
                info = docker_client.info()
                status["gpu_available"] = "nvidia" in str(info).lower()
            except Exception:
                status["gpu_available"] = False

        return status
