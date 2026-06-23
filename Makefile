# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= python3 -m sphinx
SOURCEDIR     = source
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make html".
.DEFAULT_GOAL := html

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean precommit check serve

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# 预提交检查：验证 RST 语法
precommit:
	@bash scripts/precommit-check.sh

# html 构建前自动运行 precommit 检查（警告不阻塞构建）
html: Makefile
	@$(MAKE) precommit 2>/dev/null; true
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# 本地预览服务器（默认 8000 端口）
serve: html
	@echo "Open http://localhost:8000/ in your browser"
	@cd $(BUILDDIR)/html && python3 -m http.server $(PORT)

# 增强 clean
clean:
	rm -rf $(BUILDDIR)
