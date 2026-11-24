"""
文件操作工具集：沙箱安全加固版
安全定位：Python 应用层基线防护，防御已知路径遍历、符号/硬链接逃逸、异常崩溃
架构边界：无法做到绝对安全，多用户生产环境需叠加 OS 级隔离（容器/低权限用户）

文件操作工具集：沙箱安全加固版 v2
安全定位：Python 应用层基线防护，防御已知路径遍历、符号/硬链接逃逸、异常崩溃
架构边界：无法做到绝对安全，多用户生产环境需叠加 OS 级隔离（容器/低权限用户）

目录约定：
  agent_workspace/
  ├── readonly/      # 只读区：存放工具/Agent 依赖的参考材料，禁止写入
  ├── output/        # 建议的可写区：生成结果放这里（本模块不强制，需业务层约定）
  └── (其他)         # 默认可写，向后兼容现有调用
"""
import os
from typing import Tuple
from utils.logger_tool import logger
from utils.path_tool import get_abs_path


# ========== 安全配置（集中管理，可通过配置文件覆盖） ==========
# 沙箱根目录：所有文件操作强制限制在此目录下
SANDBOX_ROOT = get_abs_path("agent_workspace")

# 文件后缀白名单：仅允许纯文本类文件
ALLOWED_EXTENSIONS = {".txt", ".md", ".markdown", ".json", ".csv", ".log"}

# 单次写入最大字节数（默认 1MB）
MAX_WRITE_BYTES = 1024 * 1024

# 路径最大长度（含目录），防御超长路径 DoS
MAX_PATH_LEN = 255

# 最大目录深度，防御极深目录耗尽 inode
MAX_DIR_DEPTH = 10

# 只读子目录黑名单：这些子目录下的文件禁止写入（相对沙箱根目录）
# 用于隔离"工具/Agent 依赖的参考材料"与"运行时生成结果"
READONLY_SUBDIRS = {"readonly"}

# Windows 保留设备名：这些名字会被 OS 解释为设备而非文件，写入会走控制台
# 规则：basename 去扩展名后大写若命中，则拒绝；Linux 下不误伤
_WIN_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


# ========== 启动时创建沙箱目录 + 固化根信息 ==========
# 修复：先确保目录存在，再 stat，避免首次运行时 stat 抛 FileNotFoundError
# 导致 _SANDBOX_DEV/INO 被置为 None，进而 _sandbox_root_intact() 永远返回 False，
# 所有写入被"路径越界"误拒。
os.makedirs(SANDBOX_ROOT, exist_ok=True)

_SANDBOX_REAL_PATH = os.path.realpath(SANDBOX_ROOT)

try:
    _sandbox_stat = os.stat(_SANDBOX_REAL_PATH)
    _SANDBOX_DEV = _sandbox_stat.st_dev
    _SANDBOX_INO = _sandbox_stat.st_ino
except OSError as e:
    logger.error(f"沙箱根目录初始化 stat 失败：{e}")
    _SANDBOX_DEV = None
    _SANDBOX_INO = None


def _sandbox_root_intact() -> bool:
    """
    校验沙箱根目录是否未被篡改
    原理：对比启动时记录的设备号+inode号，运行时不一致则根目录被替换过
    只能防御根目录整体被替换，子目录篡改由 realpath 覆盖
    """
    if _SANDBOX_DEV is None or _SANDBOX_INO is None:
        return False
    try:
        st = os.stat(_SANDBOX_REAL_PATH)
        return (st.st_dev == _SANDBOX_DEV) and (st.st_ino == _SANDBOX_INO)
    except OSError:
        return False


def _is_hardlink(real_path: str) -> bool:
    """
    检查目标文件是否为硬链接（inode 链接数 > 1）
    修复高危：硬链接逃逸 —— realpath 不解析硬链接，可指向沙箱外文件
    返回 True 表示是多链接文件，禁止写入
    """
    if not os.path.exists(real_path):
        # 文件不存在，无法是硬链接，放行
        return False
    try:
        st = os.lstat(real_path)
    except OSError:
        # 拿不到状态，保守拒绝
        return True
    return st.st_nlink > 1


def _is_windows_reserved(file_path: str) -> bool:
    """
    检查是否使用了 Windows 保留设备名
    规则：basename（去扩展名后）大写若为 CON/PRN/NUL/COM1 等，则拒绝
    - "CON.txt" / "a/b/CON.md" / "con.txt" 均命中
    - "console.txt" / "aux_backup.md" 不命中
    Linux 下会一并拦截（严格策略：跨平台行为一致，避免部署到 Windows 时出现行为漂移）
    """
    base = os.path.basename(file_path.replace("\\", "/"))
    stem = os.path.splitext(base)[0].upper()
    return stem in _WIN_RESERVED_NAMES


def _is_under_readonly(real_path: str) -> bool:
    """
    检查目标是否位于只读子目录下
    基于 real_path 与沙箱根计算相对路径，取第一段判断是否在 READONLY_SUBDIRS 中
    """
    try:
        rel = os.path.relpath(real_path, _SANDBOX_REAL_PATH)
    except ValueError:
        # 跨盘符等异常，保守拒绝
        return True
    if rel == ".." or rel.startswith(".." + os.sep):
        # 理论上 _is_path_safe 已拦截越界，这里双保险
        return True
    if rel == ".":
        # 目标是沙箱根目录本身，由后续 isdir 检查拦截
        return False
    first_segment = rel.split(os.sep, 1)[0]
    return first_segment in READONLY_SUBDIRS


def _validate_path_basic(file_path: str) -> Tuple[bool, str]:
    """
    第一层：基础字符串校验，在做系统调用前先挡掉明显非法输入
    防御：null byte 注入、换行符日志注入、超长路径、过深目录、绝对路径
    """
    if not isinstance(file_path, str):
        return False, "错误：file_path 必须是字符串"
    if "\x00" in file_path:
        return False, "错误：路径包含非法字符"
    if "\n" in file_path or "\r" in file_path:
        return False, "错误：路径包含非法字符"
    if os.path.isabs(file_path):
        return False, "错误：不允许使用绝对路径，请传入相对沙箱目录的文件路径"
    if len(file_path) > MAX_PATH_LEN:
        return False, f"错误：路径过长（{len(file_path)}字符），最大允许{MAX_PATH_LEN}字符"
    normalized = file_path.replace("\\", "/")
    depth = normalized.count("/")
    if depth > MAX_DIR_DEPTH:
        return False, f"错误：目录层级过深（{depth}层），最大允许{MAX_DIR_DEPTH}层"
    return True, ""


def _is_path_safe(target_path: str) -> Tuple[bool, str]:
    """
    第二层：核心路径安全校验
    流程：校验沙箱根完整性 → 拼接路径 → realpath 规范化 → 边界校验
    """
    if not _sandbox_root_intact():
        logger.warning("沙箱根目录校验失败，可能已被替换")
        return False, ""

    full_path = os.path.join(_SANDBOX_REAL_PATH, target_path)

    try:
        real_path = os.path.realpath(full_path)
    except (ValueError, OSError):
        return False, ""

    sandbox_prefix = _SANDBOX_REAL_PATH.rstrip(os.sep) + os.sep
    if not (real_path == _SANDBOX_REAL_PATH or real_path.startswith(sandbox_prefix)):
        logger.warning(f"路径越界拦截：输入={target_path}，真实路径={real_path}")
        return False, ""

    return True, real_path


def _is_extension_allowed(real_path: str) -> bool:
    """
    第三层：文件类型白名单校验
    基于 real_path 校验，而非原始输入路径，保证语义一致
    """
    ext = os.path.splitext(real_path)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def write_file(file_path: str, content: str, append: bool = False) -> str:
    """
    安全写入文件
    七层防护：
      1. 基础字符串校验（类型/null byte/换行/绝对路径/长度/深度）
      2. Windows 保留设备名拦截
      3. 路径边界校验（沙箱根 inode + realpath + startswith）
      4. 只读区隔离
      5. 文件类型白名单
      6. 内容大小限制
      7. 目录/硬链接检查
    """
    try:
        # 第一层：基础字符串校验
        is_valid, err_msg = _validate_path_basic(file_path)
        if not is_valid:
            return err_msg

        # 第二层：Windows 保留设备名
        if _is_windows_reserved(file_path):
            return (
                f"错误：'{os.path.basename(file_path)}' 使用了操作系统保留的设备名，禁止写入"
            )

        # 第三层：路径边界安全校验
        is_safe, real_path = _is_path_safe(file_path)
        if not is_safe:
            return "错误：文件路径超出允许的工作目录范围，禁止越权写入"

        # 第四层：只读区隔离
        if _is_under_readonly(real_path):
            return "错误：目标路径位于只读区域（readonly/），禁止写入"

        # 第五层：文件类型白名单
        if not _is_extension_allowed(real_path):
            ext = os.path.splitext(real_path)[1]
            return (
                f"错误：不支持的文件类型 {ext}，"
                f"允许的类型：{', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        # 第六层：内容大小限制
        try:
            content_bytes = content.encode("utf-8")
        except UnicodeEncodeError:
            return "错误：写入内容包含无法编码的字符"

        if len(content_bytes) > MAX_WRITE_BYTES:
            return (
                f"错误：写入内容过大（{len(content_bytes)}字节），"
                f"最大允许 {MAX_WRITE_BYTES} 字节"
            )

        # 第七层：目录检查（先于硬链接，避免目录的 st_nlink>1 被误判为硬链接）
        if os.path.isdir(real_path):
            return "错误：目标路径是已存在的目录，不能写入"

        if _is_hardlink(real_path):
            return "错误：目标文件存在多个硬链接，禁止写入以防止沙箱逃逸"

        # ========== 执行写入 ==========
        parent_dir = os.path.dirname(real_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        mode = "a" if append else "w"
        with open(real_path, mode, encoding="utf-8") as f:
            f.write(content)

        logger.info(
            f"文件写入成功 | 路径={file_path} | 模式={'追加' if append else '覆盖'} | "
            f"大小={len(content_bytes)}字节"
        )
        return f"文件写入成功，保存路径：{file_path}，文件大小：{len(content_bytes)} 字节"

    except Exception as e:
        logger.error(f"写入文件失败 {file_path}：{e}", exc_info=True)
        return f"写入文件失败：{str(e)}"


# ========== 初始化：确保只读子目录存在 ==========
# 沙箱根已在模块顶部 makedirs，这里只需补 readonly 子目录
for _sub in READONLY_SUBDIRS:
    try:
        os.makedirs(os.path.join(_SANDBOX_REAL_PATH, _sub), exist_ok=True)
    except OSError as e:
        logger.warning(f"创建只读子目录 {_sub} 失败：{e}")

logger.info(
    f"文件沙箱初始化完成 | 工作目录：{_SANDBOX_REAL_PATH} | "
    f"只读区：{sorted(READONLY_SUBDIRS)}"
)