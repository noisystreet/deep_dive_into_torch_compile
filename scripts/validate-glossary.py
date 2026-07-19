#!/usr/bin/env python3
"""
术语表覆盖率检查器
===================
扫描所有 RST 文件，提取正文中使用的术语，与 ``glossary.rst`` 中的定义做双向对比。

功能：
  1. 从 glossary.rst 提取所有术语名称（中文、英文及别名）
  2. 扫描所有 RST 文件，查找术语在正文中的出现情况
  3. 报告：正文中以 **加粗** 形式出现但 glossary 未收录的疑似术语
  4. 报告：glossary 中定义了但正文从未引用的冗余术语

用法：
  ./scripts/validate-glossary.py                       # 检查所有文件
  ./scripts/validate-glossary.py --ci                   # CI 模式（exit 1 表示失败）

返回码：0=通过, 1=发现问题
"""

import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "source"
GLOSSARY_FILE = SOURCE_DIR / "appendix" / "03_glossary.rst"

# 排除的 RST 文件（不扫描的路径片段）
EXCLUDE_PATTERNS = [
    "_build",
    ".DS_Store",
    "appendix/03_glossary.rst",  # 不扫描自身
]

# 基础术语黑名单：这些词太基础，不需要进入本项目的 glossary
# （编程语言、框架、工具名等业内公认的基础概念）
BASIC_TERMS = {
    "python",
    "pytorch",
    "cuda",
    "linux",
    "gpu",
    "cpu",
    "numpy",
    "tensor",
    "c++",
    "windows",
    "macos",
    "docker",
    "wsl2",
    "http",
    "ssh",
    "json",
    "yaml",
    "xml",
    "html",
    "css",
    "git",
    "github",
    "cli",
    "gui",
    "api",
    "sdk",
    "ide",
    "url",
    "uri",
    "diy",
    "jit",
    "fp32",
    "fp16",
    "bf16",
    "int8",
    "int32",
    "int64",
    "pip",
    "conda",
    "ruff",
    "makefile",
    "bash",
    "resnet",
    "resnet-50",
    "transformer",
    "softmax",
    "linux x86_64",
    "amd",
    "cpu",
    "nvcc",
}

# 中文黑名单：这些中文加粗通常是界面文字或普通强调，不是术语
BASIC_CN_TERMS = {
    "上一步",
    "下一步",
    "保存",
    "取消",
    "确认",
    "取消",
    "示例",
    "概述",
    "小结",
    "前言",
    "附录",
    "目录",
    "说明",
    "提示",
    "注意",
    "警告",
    "参考",
    "更多",
    "方法一",
    "方法二",
    "方法三",
    "方案一",
    "方案二",
    "常见陷阱",
    "最佳实践",
    "延伸阅读",
    "本章小结",
    "通用约束",
    "项目概述",
}


def is_likely_term(text):
    """判断加粗文本是否可能是需要收录 glossary 的术语。

    采用保守策略：只匹配符合本书写作约定的模式——
    "术语首次出现时附英文原文，并用 **加粗** 标记"

    匹配的模式：
      1. 含空格的多词英文（如 "Graph Break"、"Symbolic Shapes"）
      2. 中文+英文+括号（如 "即时编译（JIT）"、"Fusion（融合）"）
      3. 纯英文多词技术名（含连字符，如 "Min-Cut"、"Define-by-run"）

    排除：
      - 纯英文单书（True, Print 等代码上下文产物）
      - 代码样式（含 =、() 的纯英文）
      - 基础工具名（Python, PyTorch 等）
    """
    text = text.strip()
    if len(text) < 3 or len(text) > 40:
        return False

    # 纯数字/符号
    if re.match(r"^[\d\s\-–—•·\[\]()（）.,:;，。：；xX]+$", text):
        return False

    # 基础英文黑名单
    if text.lower() in BASIC_TERMS:
        return False

    # 代码样式
    if re.search(r"[=]", text) and not re.search(r"[\u4e00-\u9fff]", text):
        return False
    if text.endswith(".py") or text.startswith("import"):
        return False

    # 模式 1：含空格的多词英文（至少 2 个英文词）
    if re.match(r"^[a-zA-Z][a-zA-Z0-9_.\- ]+[a-zA-Z0-9_.]$", text):
        # 至少两个单词
        words = [w for w in re.split(r"[\s\-]", text) if w]
        if len(words) >= 2:
            return True

    # 模式 2：中文+英文混合（如 "Guard 机制"、"编译 API"）
    # 必须是短名词短语（<= 20 字符），不含句子成分
    if re.search(r"[\u4e00-\u9fff]", text) and re.search(r"[a-zA-Z]", text):
        # 排除明显是句子的：含中文动词/介词/助词
        if re.search(
            r"[\u4e86\u4e0d\u4f1a\u53ef\u4ee5\u5c06\u628a\u88ab\u7528\u662f\u5728\u6709\u7740\u8fc7\u7684\u800c]",
            text,
        ):
            return False
        # 排除长文本（> 20 字符的混合文本通常是句子）
        if len(text) > 20:
            return False
        # 含括号的（如 "即时编译（JIT）"、"Fusion（融合）"）
        if re.search(r"[（(]", text):
            return True
        # 短名词性混合（如 "Guard 机制"、"编译 API"、"PEP 523"）
        # 不含中文标点、不含动词性结构
        if not re.search(r"[\uff1a\uff0c\u3002\uff01\uff1f]", text):
            # 排除包含 = 或 .py 等代码特征
            if "=" not in text and not text.endswith(".py"):
                # 必须包含至少一个中文词和一个英文词
                cn_words = re.findall(r"[\u4e00-\u9fff]+", text)
                en_words = re.findall(r"[a-zA-Z][a-zA-Z0-9._]+", text)
                if cn_words and en_words:
                    return True

    return False


def is_excluded(path):
    """判断文件是否应该被排除"""
    rel = str(path.relative_to(SOURCE_DIR))
    for pat in EXCLUDE_PATTERNS:
        if pat in rel:
            return True
    return False


def parse_glossary(path):
    """从 glossary.rst 提取术语名称列表。

    Sphinx glossary 格式：
    .. glossary::

       Term Name
          定义文本...

       Term Name（别名）
          定义文本...

    返回：[(line_number, term_name), ...]
    """
    if not path.exists():
        print(f"❌ 未找到 glossary 文件: {path}")
        sys.exit(1)

    terms = []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    in_glossary = False
    for lineno, line in enumerate(lines, 1):
        stripped = line.rstrip("\n")

        if stripped.strip() == ".. glossary::":
            in_glossary = True
            continue

        if not in_glossary:
            continue

        # 离开 glossary 块：遇到非缩进、非空行的 directive 或标题
        if stripped and not stripped.startswith(" ") and not stripped.startswith("\t"):
            if (
                stripped.startswith(".. ")
                or stripped.startswith("===")
                or stripped.startswith("---")
            ):
                break
            # 空行后遇到非缩进文本（如另一个 section 标题）
            if stripped.strip() and not stripped[0].isspace():
                break

        # 术语行：缩进 3 个空格，后面跟术语名（非空，不包含 directive）
        # 示例："   TorchDynamo" 或 "   Dynamic Shapes（动态形状）"
        if re.match(r"^   \S", stripped):
            term_name = stripped.strip()
            # 跳过非术语行（如空行、注释）
            if term_name and not term_name.startswith(".."):
                terms.append((lineno, term_name))

    return terms


def collect_rst_files():
    """收集所有需要扫描的 RST 文件"""
    files = []
    for root, dirs, fnames in os.walk(SOURCE_DIR):
        for fn in fnames:
            if not fn.endswith(".rst"):
                continue
            fpath = Path(root) / fn
            if is_excluded(fpath):
                continue
            files.append(fpath)
    return sorted(files)


def extract_bold_terms(filepath):
    """从 RST 文件中提取 **加粗** 标记的文本。

    处理以下模式：
      - **Term Name**
      - **术语（English）**
      - **English（中文）**

    注意：不会捕获代码块中的 ``**...**``（code-block 和 inline literal 中的不计入）。

    返回：[(line_number, bold_text), ...]
    """
    bold_terms = []
    in_code_block = False
    code_block_indent = None

    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    for lineno, line in enumerate(lines, 1):
        stripped = line.rstrip("\n")

        # 检测代码块边界
        if re.match(r"^\.\. code-block::", stripped.strip()):
            in_code_block = True
            code_block_indent = None
            continue
        if in_code_block:
            if code_block_indent is None and stripped.strip():
                code_block_indent = len(stripped) - len(stripped.lstrip())
            if code_block_indent is not None:
                if stripped.strip() and not stripped.startswith(
                    " " * code_block_indent
                ):
                    in_code_block = False
                elif not stripped.strip():
                    continue
                else:
                    continue
            else:
                continue

        # 跳过 directive 行
        if re.match(r"^\.\. ", stripped.strip()) and "::" in stripped.strip():
            continue

        # 跳过 mermaid、code、literal 的 directive 块
        if re.match(
            r"^\.\. (mermaid|note|tip|warning|seealso|important)::", stripped.strip()
        ):
            in_code_block = True
            code_block_indent = None
            continue

        # 提取 **text** 模式（不在代码块中）
        for match in re.finditer(r"\*\*([^*]+)\*\*", stripped):
            text = match.group(1).strip()
            if text:
                bold_terms.append((lineno, text))

    return bold_terms


def normalize_term(term):
    """规范化术语名用于匹配。

    例如：
      "Dynamic Shapes（动态形状）" → ["Dynamic Shapes", "动态形状"]
      "TorchDynamo" → ["TorchDynamo"]
      "AOTAutograd" → ["AOTAutograd"]
      "Lowering（降级）" → ["Lowering", "降级"]
      "Pointwise（逐元素操作）" → ["Pointwise", "逐元素操作"]
    """
    # 尝试提取 "English（中文）" 模式
    m = re.match(r"^(.+?)（(.+?)）$", term)
    if m:
        return [m.group(1).strip(), m.group(2).strip()]
    # 尝试提取 "English (中文)" 模式（英文括号）
    m = re.match(r"^(.+?)\((.+?)\)$", term)
    if m:
        return [m.group(1).strip(), m.group(2).strip()]
    # 单一名称
    return [term.strip()]


def scan_glossary_usage(glossary_terms, rst_files):
    """扫描所有 RST 文件，检查 glossary 术语的引用情况。

    返回：
      used_terms: set — 在正文中被引用的 glossary 术语原始名称
      bold_not_in_glossary: [(file, line, bold_text)] — 加粗但 glossary 未收录的术语
    """
    # 构建查找表：所有术语的规范化变体 → 原始术语名
    variant_to_term = {}
    for _, term_name in glossary_terms:
        variants = normalize_term(term_name)
        for v in variants:
            variant_to_term[v.lower()] = term_name

    used_terms = set()
    bold_not_in_glossary = []

    for fpath in rst_files:
        bold_terms = extract_bold_terms(fpath)

        for lineno, bold_text in bold_terms:
            # 检查是否匹配 glossary 中的某个术语
            matched = False
            for _, term_name in glossary_terms:
                variants = normalize_term(term_name)
                for v in variants:
                    if v.lower() in bold_text.lower() or bold_text.lower() in v.lower():
                        used_terms.add(term_name)
                        matched = True
                        break
                if matched:
                    break

            if not matched:
                bold_not_in_glossary.append((fpath, lineno, bold_text))

    return used_terms, bold_not_in_glossary


def check_plain_text_usage(glossary_terms, used_terms, rst_files):
    """对于未以加粗形式出现的 glossary 术语，检查是否以纯文本形式出现。"""
    all_text = ""
    for fpath in rst_files:
        with open(fpath, encoding="utf-8") as f:
            content = f.read()
        # 移除代码块内容减少误报
        content = re.sub(
            r"\.\. code-block::.*?(?=^\.\. |\Z)",
            "",
            content,
            flags=re.DOTALL | re.MULTILINE,
        )
        all_text += content.lower()

    found_in_plain = set()
    for lineno, term_name in glossary_terms:
        if term_name in used_terms:
            continue
        variants = normalize_term(term_name)
        for v in variants:
            if v.lower() in all_text:
                found_in_plain.add(term_name)
                break

    return found_in_plain


def main():
    import argparse

    parser = argparse.ArgumentParser(description="术语表覆盖率检查器")
    parser.add_argument("--ci", action="store_true", help="CI 模式（exit 1 表示失败）")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出：同时显示已自动跳过的非术语文本",
    )
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="启用疑似术语检测：扫描正文中 **加粗** 但 glossary 未收录的候选术语",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("术语表覆盖率检查")
    print("=" * 60)

    # 1. 解析 glossary
    print(f"\n📖 解析术语表: {GLOSSARY_FILE.relative_to(PROJECT_ROOT)}")
    glossary_terms = parse_glossary(GLOSSARY_FILE)
    print(f"   共 {len(glossary_terms)} 个术语定义")

    # 2. 收集 RST 文件
    rst_files = collect_rst_files()
    print(f"\n📂 扫描文件: {len(rst_files)} 个 RST 文件")

    # 3. 扫描正文
    used_terms, bold_not_in_glossary = scan_glossary_usage(glossary_terms, rst_files)

    # 4. 检查纯文本引用
    plain_used = check_plain_text_usage(glossary_terms, used_terms, rst_files)

    # 5. 报告
    has_issues = False

    # 5a. 未引用的 glossary 术语
    unused_terms = []
    for lineno, term_name in glossary_terms:
        if term_name not in used_terms and term_name not in plain_used:
            unused_terms.append((lineno, term_name))

    if unused_terms:
        has_issues = True
        print(f"\n❌ Glossary 已定义但正文未引用的术语 ({len(unused_terms)} 个)：")
        for lineno, term_name in unused_terms:
            print(f"   - L{lineno}: {term_name}")
        print("   建议：检查这些术语是否需要保留，或是否在正文中使用了不同的表述。")
    else:
        green = "\033[0;32m" if sys.stdout.isatty() else ""
        reset = "\033[0m" if sys.stdout.isatty() else ""
        print(f"\n✅ 所有 glossary 术语在正文中至少出现一次。")

    # 5b. 加粗但 glossary 未收录的术语（仅 --suggest 模式下显示）
    if args.suggest and bold_not_in_glossary:
        # 使用 is_likely_term 过滤
        suspect_terms = []
        skipped_bolds = []
        for item in bold_not_in_glossary:
            if is_likely_term(item[2]):
                suspect_terms.append(item)
            else:
                skipped_bolds.append(item)

        if suspect_terms:
            has_issues = True
            print(
                f"\n⚠️  正文中 **加粗** 但 glossary 未收录的疑似术语 ({len(suspect_terms)} 个)："
            )
            for fpath, lineno, bold_text in suspect_terms[:30]:  # 最多显示30个
                rel_path = fpath.relative_to(PROJECT_ROOT)
                print(f"   - L{lineno}: {bold_text}  ({rel_path})")
            if len(suspect_terms) > 30:
                print(f"   ... 还有 {len(suspect_terms) - 30} 个未显示")
            print("   建议：确认这些是否是正式术语，如果是则补充到 glossary 中。")

        if args.verbose and skipped_bolds:
            print(f"\n📋 已自动跳过的非术语文本 ({len(skipped_bolds)} 个，前 10 个)：")
            for fpath, lineno, bold_text in skipped_bolds[:10]:
                rel_path = fpath.relative_to(PROJECT_ROOT)
                print(f"   - L{lineno}: {bold_text}  ({rel_path})")

    if not has_issues:
        print(f"\n✅ 全部通过！术语表覆盖率完整。")
        return 0
    else:
        if args.ci:
            print(f"\n❌ CI 检查未通过。")
            return 1
        else:
            print(f"\n⚠️  发现问题，建议修复后再提交。")
            return 1


if __name__ == "__main__":
    sys.exit(main())
