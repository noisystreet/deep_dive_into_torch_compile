#!/usr/bin/env python3
"""
CJK 间距检查器
===============
检查 RST 文件中 inline markup（**bold**、``literal``）与 CJK 字符之间的
空格缺失问题。

用法:
  ./scripts/check-cjk-spacing.py              # 检查所有 RST 文件
  ./scripts/check-cjk-spacing.py --fix         # 自动修复（谨慎使用）
  ./scripts/check-cjk-spacing.py source/foo.rst  # 检查指定文件

返回码: 0=通过, 1=发现问题
"""

import os, re, sys, argparse

# CJK 字符范围
CJK = set("\u4e00-\u9fff\u3000-\u303f\uff00-\uffef")
# CJK 标点（这些跟在 markup 后面不需要空格也正常渲染）
CJK_PUNCT = set("）？。，：；！、】』』』』』』")


def check_file(path, fix=False):
    """检查单个文件，返回 (issues_found, fixed_lines)"""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    in_code_block = False
    in_bold = False  # 跨行保持状态
    in_literal = False  # 跨行保持状态
    issues = []
    fixed_lines = []
    changed = False

    for lineno, line in enumerate(lines, 1):
        stripped = line.rstrip("\n")

        # 跳过代码块
        if stripped.strip().startswith(".. code-block::"):
            in_code_block = True
            fixed_lines.append(line)
            continue
        if in_code_block:
            if not stripped.strip() or not stripped.startswith("   "):
                in_code_block = False
            else:
                fixed_lines.append(line)
                continue

        # 跳过纯注释行
        if stripped.strip().startswith(".. "):
            fixed_lines.append(line)
            continue

        # 逐字符扫描
        i = 0
        result = []

        while i < len(stripped):
            ch = stripped[i]

            # 检测 **
            if i + 1 < len(stripped) and stripped[i : i + 2] == "**":
                prev = stripped[i - 1] if i > 0 else " "
                next_ = stripped[i + 2] if i + 2 < len(stripped) else " "

                if not in_bold and not in_literal:
                    # 可能是 opening **
                    # CJK 紧挨 before ** → 缺少空格
                    if is_cjk(prev):
                        issues.append(
                            (path, lineno, f"'{prev}**' 前缺少空格: ...{prev}**...")
                        )
                        if fix:
                            result.append(" ")
                    in_bold = True
                    result.append("**")
                    i += 2
                    continue

                elif in_bold and not in_literal:
                    # 可能是 closing **
                    # 先去除 result 末尾空格（RST 不允许 ** 前有空格当关闭标记）
                    while result and result[-1] == " ":
                        result.pop()
                    # CJK 紧挨 after ** → 缺少空格
                    if is_cjk(next_):
                        issues.append(
                            (path, lineno, f"'**{next_}' 后缺少空格: ...**{next_}...")
                        )
                        if fix:
                            result.append("** ")
                            i += 2
                            in_bold = False
                            continue
                    in_bold = False
                    result.append("**")
                    i += 2
                    continue

            # 检测 ``
            if i + 1 < len(stripped) and stripped[i : i + 2] == "``":
                prev = stripped[i - 1] if i > 0 else " "
                next_ = stripped[i + 2] if i + 2 < len(stripped) else " "

                if not in_literal and not in_bold:
                    if is_cjk(prev):
                        issues.append(
                            (path, lineno, f"'前 ``' 前缺少空格: ...{prev}``...")
                        )
                        if fix:
                            result.append(" ")
                    in_literal = True
                    result.append("``")
                    i += 2
                    continue

                elif in_literal and not in_bold:
                    # 去除 result 末尾空格
                    while result and result[-1] == " ":
                        result.pop()
                    if is_cjk(next_):
                        issues.append(
                            (path, lineno, f"'``{next_}' 后缺少空格: ...``{next_}...")
                        )
                        if fix:
                            result.append("`` ")
                            i += 2
                            in_literal = False
                            continue
                    in_literal = False
                    result.append("``")
                    i += 2
                    continue

            result.append(ch)
            i += 1

        fixed_lines.append("".join(result) + "\n")

    # 检测是否有内容发生变化
    changed = fixed_lines != lines
    return issues, fixed_lines, changed


def is_cjk(ch):
    """判断是否为 CJK 字符"""
    cp = ord(ch)
    return 0x4E00 <= cp <= 0x9FFF or 0x3000 <= cp <= 0x303F or 0xFF00 <= cp <= 0xFFEF


def main():
    parser = argparse.ArgumentParser(description="CJK 间距检查器")
    parser.add_argument("files", nargs="*", help="要检查的文件（默认全部）")
    parser.add_argument("--fix", action="store_true", help="自动修复（谨慎）")
    args = parser.parse_args()

    # 确定检查范围
    if args.files:
        files = [f for f in args.files if f.endswith(".rst")]
    else:
        # 检查所有 RST 文件
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        files = []
        for root, dirs, fnames in os.walk(os.path.join(base, "source")):
            for fn in fnames:
                if fn.endswith(".rst"):
                    files.append(os.path.join(root, fn))

    total_issues = 0
    for f in sorted(files):
        if not os.path.exists(f):
            continue
        issues, fixed, changed = check_file(f, fix=args.fix)

        if issues:
            if args.fix:
                with open(f, "w", encoding="utf-8") as fh:
                    fh.writelines(fixed)
                print(f"  🔧 {len(issues)} 处修复: {f}")
            else:
                print(f"\n  ⚠ {f}:")
                for path, lineno, msg in issues:
                    print(f"    L{lineno}: {msg}")
            total_issues += len(issues)
        elif args.fix and changed:
            with open(f, "w", encoding="utf-8") as fh:
                fh.writelines(fixed)
            print(f"  🔧 格式修正: {f}")
        elif args.fix and changed:
            # 无 CJK 问题但有格式修正（如去掉关闭 ** 前的空格）
            with open(f, "w", encoding="utf-8") as fh:
                fh.writelines(fixed)
            print(f"  🔧 格式修正: {f}")

    if total_issues == 0:
        print("✅ 所有文件 CJK 间距正确")
        return 0
    else:
        print(f"\n共 {total_issues} 个问题" + ("（已自动修复）" if args.fix else ""))
        return 1


if __name__ == "__main__":
    sys.exit(main())
