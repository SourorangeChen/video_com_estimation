# pytest 配置文件：让测试能直接 import 上一级目录(code/)中的脚本模块。
import sys
from pathlib import Path

# 把 code/ 目录(本文件父目录的父目录)加入模块搜索路径，
# 这样测试里可以直接 `from validate_com_normalized import ...`。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
