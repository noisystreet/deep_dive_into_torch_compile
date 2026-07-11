#!/usr/bin/env python3
"""
从 examples/ 中的 .py 文件提取代码，同步到 RST 文档。

用法:
    python scripts/sync_examples.py              # 同步所有示例
    python scripts/sync_examples.py --check       # 只检查，不修改
    python scripts/sync_examples.py --verbose     # 显示详细信息

工作方式:
    1. 读取 sync_config.json 配置文件
    2. 对每个配置项:
       a. 读取 .py 文件，提取 # --- docs: start --- 和 # --- docs: end --- 之间的代码
       b. 找到 RST 文件中 .. synced-code-start:: 和 .. synced-code-end:: 之间的内容
       c. 替换为提取的代码 (保持 RST 缩进格式)
    3. 输出修改摘要
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "scripts" / "sync_config.json"


def load_config():
    """加载同步配置。"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_code_from_py(py_path: Path, section: str = "default") -> str | None:
    """从 .py 文件提取指定 section 的代码。

    section 对应 # --- docs: <section> --- 标记。
    如果 section 为 "default"，则使用 # --- docs: start --- 标记。
    """
    if not py_path.exists():
        print(f"  [错误] 文件不存在: {py_path}")
        return None

    content = py_path.read_text(encoding="utf-8")

    if section == "default":
        start_marker = "# --- docs: start ---"
        end_marker = "# --- docs: end ---"
    else:
        start_marker = f"# --- docs: {section} ---"
        end_marker = f"# --- docs: end ---"

    start_idx = content.find(start_marker)
    if start_idx == -1:
        print(f"  [警告] 未找到起始标记 '{start_marker}' 在 {py_path}")
        return None

    end_idx = content.find(end_marker, start_idx + len(start_marker))
    if end_idx == -1:
        print(f"  [警告] 未找到结束标记 '{end_marker}' 在 {py_path}")
        return None

    code = content[start_idx + len(start_marker) : end_idx]
    # 去除首尾空行
    code = code.strip("\n")
    # 统一去除代码的公共缩进
    lines = code.split("\n")
    if lines and lines[0].startswith("\n"):
        lines = lines[1:]
    # 去除尾部空行
    while lines and not lines[-1].strip():
        lines.pop()
    code = "\n".join(lines)

    return code


def get_rst_marker_line(
    rst_lines: list[str], marker: str, section: str = "", start_from: int = 0
) -> int:
    """在 RST 行列表中查找标记行，返回行号（0-based）。

    支持带 section 名的标记，如 `.. synced-code-start:: add_kernel`。
    """
    full_marker = f"{marker} {section}".strip() if section else marker
    for i in range(start_from, len(rst_lines)):
        line = rst_lines[i].rstrip()
        if line == full_marker:
            return i
    return -1


def sync_to_rst(
    rst_path: Path, code: str, section: str = "", indent: int = 3, dry_run: bool = False
) -> bool:
    """将代码同步到 RST 文件的 synced-code-start/end 标记之间。

    返回 True 表示有修改，False 表示无变化。
    """
    if not rst_path.exists():
        print(f"  [错误] RST 文件不存在: {rst_path}")
        return False

    base_marker = ".. synced-code-start::"
    end_marker = ".. synced-code-end::"

    rst_lines = rst_path.read_text(encoding="utf-8").split("\n")

    start_line = get_rst_marker_line(rst_lines, base_marker, section)
    if start_line == -1:
        marker_display = f"{base_marker} {section}".strip() if section else base_marker
        print(f"  [警告] 未找到起始标记 '{marker_display}' 在 {rst_path}")
        return False

    end_line = get_rst_marker_line(rst_lines, end_marker, "", start_line + 1)
    if end_line == -1:
        print(f"  [警告] 未找到结束标记 '{end_marker}' 在 {rst_path}")
        return False

    # 生成缩进的代码块
    indent_str = " " * indent
    code_indent_str = " " * (indent + 3)  # 代码比 code-block 多缩进 3 格
    code_lines = code.split("\n")
    indented_code = "\n".join(
        f"{code_indent_str}{line}" if line.strip() else "" for line in code_lines
    )

    # 构建新内容
    new_lines = (
        rst_lines[: start_line + 1]
        + [""]
        + [f"{indent_str}.. code-block:: python"]
        + [f"{indent_str}   :linenos:"]
        + [""]
        + indented_code.split("\n")
        + [""]
        + rst_lines[end_line:]
    )

    new_content = "\n".join(new_lines)

    if new_content == rst_path.read_text(encoding="utf-8"):
        return False

    if not dry_run:
        rst_path.write_text(new_content, encoding="utf-8")

    return True


def sync_all(dry_run: bool = False, verbose: bool = False):
    """同步所有配置的示例。"""
    config = load_config()
    total_modified = 0
    total_errors = 0
    total_skipped = 0

    print(f"{'检查' if dry_run else '同步'}模式: 共 {len(config)} 个配置项\n")

    for entry in config:
        py_rel = entry["py"]
        rst_rel = entry["rst"]
        section = entry.get("section", "default")
        label = entry.get("label", "")

        py_path = REPO_ROOT / py_rel
        rst_path = REPO_ROOT / rst_rel

        print(f"[{py_rel}] → [{rst_rel}]", end="")

        if label:
            print(f"  ({label})", end="")
        print()

        # 提取代码
        code = extract_code_from_py(py_path, section)
        if code is None:
            print(f"  ⚠ 跳过: 代码提取失败")
            total_skipped += 1
            continue

        if verbose:
            print(f"  --- 提取的代码 ({len(code.split(chr(10)))} 行) ---")
            for line in code.split("\n"):
                print(f"  | {line}")
            print("  ---")

        # 同步到 RST（"default" 表示无 section 名的纯标记）
        rst_section = section if section != "default" else ""
        modified = sync_to_rst(rst_path, code, section=rst_section, dry_run=dry_run)
        if modified:
            print(f"  ✓ 已{'需要' if dry_run else ''}更新")
            total_modified += 1
        else:
            print(f"  - 已是最新")
            total_skipped += 1

    print(f"\n{'=' * 40}")
    print(f"总计: {len(config)} 配置项")
    print(f"已修改: {total_modified}")
    print(f"已跳过: {total_skipped}")
    if total_errors:
        print(f"错误: {total_errors}")

    return total_modified > 0


def main():
    parser = argparse.ArgumentParser(description="从 examples/ 同步代码到 RST 文档")
    parser.add_argument("--check", action="store_true", help="只检查差异，不修改文件")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    args = parser.parse_args()

    sync_all(dry_run=args.check, verbose=args.verbose)


if __name__ == "__main__":
    main()
