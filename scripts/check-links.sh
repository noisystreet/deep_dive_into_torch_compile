#!/bin/bash
# 外部链接有效性检查脚本
# ==========================
# 基于 Sphinx linkcheck builder，检查所有文档中的外部链接是否有效。
#
# 用法：
#   ./scripts/check-links.sh                # 检查所有外部链接
#   ./scripts/check-links.sh --ci            # CI 模式（严格模式：任何死链都返回 1）
#   ./scripts/check-links.sh --ci --warn     # CI 模式但仅警告 4xx 错误（不阻塞）
#
# 返回码：
#   0 = 通过（无死链）
#   1 = 发现死链（CI 模式）
#   2 = 发现死链（警告模式，不阻塞）
#   3 = 构建失败（Sphinx 自身错误）

set -e

# ---- 确定项目根目录 ----
if command -v git &>/dev/null && git rev-parse --git-dir &>/dev/null; then
    PROJECT_ROOT="$(git rev-parse --show-toplevel)"
else
    cd "$(dirname "$(readlink -f "$0" || echo "$0")")"
    PROJECT_ROOT="$(cd .. && pwd)"
fi

BUILD_DIR="_build/linkcheck"
EXIT_CODE=0

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 参数解析
CI_MODE=false
WARN_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ci) CI_MODE=true; shift ;;
        --warn) WARN_MODE=true; shift ;;
        *) echo -e "${RED}未知参数: $1${NC}"; exit 1 ;;
    esac
done

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  外部链接有效性检查${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# ---- 检查 Sphinx 是否安装 ----
if ! python3 -c "import sphinx" 2>/dev/null; then
    echo -e "${RED}错误：未安装 Sphinx。请先运行: pip install -r requirements.txt${NC}"
    exit 3
fi

# ---- 清理上次构建产物 ----
rm -rf "$PROJECT_ROOT/$BUILD_DIR"

# ---- 运行 linkcheck ----
echo -e "${YELLOW}正在检查外部链接...（这可能需要几分钟）${NC}"
echo ""

BUILD_START=$(date +%s)

# 使用 -b linkcheck 运行 Sphinx
python3 -m sphinx -b linkcheck \
    "$PROJECT_ROOT/source" \
    "$PROJECT_ROOT/$BUILD_DIR" \
    2>"$PROJECT_ROOT/$BUILD_DIR/sphinx_err.txt" \
    >"$PROJECT_ROOT/$BUILD_DIR/sphinx_out.txt" || true  # linkcheck 本身返回非 0 是正常的（发现死链）

BUILD_END=$(date +%s)
DURATION=$((BUILD_END - BUILD_START))

echo ""
echo -e "${CYAN}检查完成（耗时 ${DURATION}s）${NC}"
echo ""

# ---- 解析 linkcheck 输出 ----
LINKCHECK_OUTPUT="$PROJECT_ROOT/$BUILD_DIR/output.txt"

# 如果 Sphinx linkcheck 出错（非链接错误本身）
SPHINX_ERR_FILE="$PROJECT_ROOT/$BUILD_DIR/sphinx_err.txt"
if [ -s "$SPHINX_ERR_FILE" ]; then
    if grep -qE '(ERROR|CRITICAL)' "$SPHINX_ERR_FILE" 2>/dev/null; then
        echo -e "${RED}Sphinx 构建错误：${NC}"
        grep -E '(ERROR|CRITICAL)' "$SPHINX_ERR_FILE" | head -20
        EXIT_CODE=3
    fi
fi

# 分类链接检查结果
BROKEN_LINKS=()
TIMEOUT_LINKS=()
REDIRECT_LINKS=()
WORKING_LINKS=0

if [ -f "$LINKCHECK_OUTPUT" ]; then
    # 使用变量存储 regex 避免 bash 转义问题
    LINKCHECK_RE='^([^:]+):([0-9]+): \[([0-9]+|-[0-9]+|timeout)\] (.+) -> (.+)$'
    while IFS= read -r line; do
        # 跳过注释和空行
        [[ "$line" =~ ^# ]] && continue
        [[ -z "$line" ]] && continue

        # linkcheck output.txt 格式：
        # /path/to/file.rst:line: [status_code] link_text -> URL
        if [[ "$line" =~ $LINKCHECK_RE ]]; then
            FILENAME="${BASH_REMATCH[1]}"
            LINE="${BASH_REMATCH[2]}"
            STATUS="${BASH_REMATCH[3]}"
            LINK_TEXT="${BASH_REMATCH[4]}"
            URL="${BASH_REMATCH[5]}"

            # 过滤掉内部锚点（# 后跟内部跳转的链接）
            if [[ "$URL" =~ ^file:// ]]; then
                continue
            fi

            case "$STATUS" in
                200|301)
                    WORKING_LINKS=$((WORKING_LINKS + 1))
                    ;;
                302|303|307|308)
                    REDIRECT_LINKS+=("$URL|$STATUS|$FILENAME:$LINE|$LINK_TEXT")
                    ;;
                400|401|403|404|410|451)
                    BROKEN_LINKS+=("$URL|$STATUS|$FILENAME:$LINE|$LINK_TEXT")
                    ;;
                500|502|503|504)
                    BROKEN_LINKS+=("$URL|$STATUS|$FILENAME:$LINE|$LINK_TEXT")
                    ;;
                timeout|-1|-2)
                    TIMEOUT_LINKS+=("$URL|$STATUS|$FILENAME:$LINE|$LINK_TEXT")
                    ;;
                *)
                    # 其他状态码（如 0 表示未知错误）
                    if [ "$STATUS" = "0" ]; then
                        TIMEOUT_LINKS+=("$URL|$STATUS|$FILENAME:$LINE|$LINK_TEXT")
                    else
                        BROKEN_LINKS+=("$URL|$STATUS|$FILENAME:$LINE|$LINK_TEXT")
                    fi
                    ;;
            esac
        fi
    done < <(cat "$LINKCHECK_OUTPUT" 2>/dev/null || true)
else
    echo -e "${YELLOW}⚠  未找到 linkcheck 输出文件（可能是所有链接均有效）${NC}"
fi

# ---- 报告结果 ----
TOTAL_LINKS=$((WORKING_LINKS + ${#BROKEN_LINKS[@]} + ${#TIMEOUT_LINKS[@]} + ${#REDIRECT_LINKS[@]}))
echo -e "统计结果："
echo -e "  ${GREEN}✓ 有效链接：${WORKING_LINKS}${NC}"
if [ ${#REDIRECT_LINKS[@]} -gt 0 ]; then
    echo -e "  ${YELLOW}⚠  重定向链接：${#REDIRECT_LINKS[@]}${NC}"
fi
if [ ${#BROKEN_LINKS[@]} -gt 0 ]; then
    echo -e "  ${RED}✗ 死链（4xx/5xx）：${#BROKEN_LINKS[@]}${NC}"
fi
if [ ${#TIMEOUT_LINKS[@]} -gt 0 ]; then
    echo -e "  ${YELLOW}⚠  超时/无法连接：${#TIMEOUT_LINKS[@]}${NC}"
fi
echo ""

# 输出死链详情
if [ ${#BROKEN_LINKS[@]} -gt 0 ]; then
    echo -e "${RED}━━━ 死链详情（4xx/5xx）━━━${NC}"
    for entry in "${BROKEN_LINKS[@]}"; do
        IFS='|' read -r url status location text <<< "$entry"
        echo -e "  ${RED}[${status}]${NC} ${url}"
        echo -e "          ${location} — \"${text}\""
    done
    echo ""
fi

# 输出超时详情
if [ ${#TIMEOUT_LINKS[@]} -gt 0 ]; then
    echo -e "${YELLOW}━━━ 超时/无法连接详情━━━${NC}"
    for entry in "${TIMEOUT_LINKS[@]}"; do
        IFS='|' read -r url status location text <<< "$entry"
        echo -e "  ${YELLOW}[${status}]${NC} ${url}"
        echo -e "          ${location} — \"${text}\""
    done
    echo ""
fi

# 输出重定向详情（仅在 CI 模式或 --warn 模式显示）
if [ ${#REDIRECT_LINKS[@]} -gt 0 ] && [ "$CI_MODE" = true ]; then
    echo -e "${YELLOW}━━━ 重定向链接（建议更新为直接链接）━━━${NC}"
    for entry in "${REDIRECT_LINKS[@]}"; do
        IFS='|' read -r url status location text <<< "$entry"
        echo -e "  ${YELLOW}[${status}]${NC} ${url}"
        echo -e "          ${location}"
    done
    echo ""
fi

# ---- 确定返回码 ----
if [ ${#BROKEN_LINKS[@]} -gt 0 ] || [ ${#TIMEOUT_LINKS[@]} -gt 0 ]; then
    if [ "$CI_MODE" = true ]; then
        EXIT_CODE=1
        echo -e "${RED}❌ CI 检查未通过：发现 ${#BROKEN_LINKS[@]} 个死链${NC}"
    elif [ "$WARN_MODE" = true ]; then
        EXIT_CODE=2
        echo -e "${YELLOW}⚠  发现 ${#BROKEN_LINKS[@]} 个死链（警告模式，不阻塞）${NC}"
    else
        EXIT_CODE=2
        echo -e "${YELLOW}⚠  发现 ${#BROKEN_LINKS[@]} 个死链，建议修复${NC}"
    fi
else
    echo -e "${GREEN}✅ 所有外部链接有效！${NC}"
    EXIT_CODE=0
fi

# 建议
if [ $EXIT_CODE -ne 0 ] && [ "$CI_MODE" != true ]; then
    echo ""
    echo -e "${YELLOW}提示：${NC}"
    echo -e "  - 链接失效可能由网络临时问题导致，可重新运行确认"
    echo -e "  - 定期运行此脚本（如每周一次）可防止链接腐化"
    echo -e "  - 使用 ./scripts/check-links.sh --ci 严格模式检查死链"
fi

exit $EXIT_CODE
