import os
from shlex import join

# 获取项目根目录
def get_project_root() -> str:
    # 获得当前文件的绝对路径
    current_abs_path = os.path.abspath(__file__)

    # 获取当前文件所在文件夹绝对路径
    current_dir_path = os.path.dirname(current_abs_path)

    # 获取项目根目录
    project_root = os.path.dirname(current_dir_path)

    return project_root

# 获取在项目中文件的绝对路径
def get_abs_path(filepath: str) -> str:
    return os.path.join(get_project_root(), filepath)