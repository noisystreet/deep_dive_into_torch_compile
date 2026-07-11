"""
synced-code-start / synced-code-end 指令支持
=============================================
使 ``sync_examples.py`` 使用的代码同步标记在 Sphinx 构建中透明通过，
不会因"未知指令"报错而阻止渲染内部的 ``code-block``。
"""

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import ViewList
from sphinx.application import Sphinx
from sphinx.util.docutils import switch_source_input


class SyncedCodeStart(Directive):
    """``.. synced-code-start::`` — 透明容器，直接传递内部内容。"""

    has_content = True

    def run(self):
        # 将内容解析为 docutils 节点树，透明传递
        node = nodes.container()
        self.state.nested_parse(self.content, self.content_offset, node)
        return node.children


class SyncedCodeEnd(Directive):
    """``.. synced-code-end::`` — 无操作，仅作标记。"""

    has_content = False

    def run(self):
        return []


def setup(app: Sphinx):
    app.add_directive("synced-code-start", SyncedCodeStart)
    app.add_directive("synced-code-end", SyncedCodeEnd)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
