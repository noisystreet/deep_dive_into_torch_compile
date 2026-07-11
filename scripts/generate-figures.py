#!/usr/bin/env python3
"""
批量生成书籍的 Matplotlib 图表。
用法:
  ./scripts/generate-figures.py              # 生成全部图表
  ./scripts/generate-figures.py --watch      # 文件变动时自动重生成
  ./scripts/generate-figures.py compilation_pipeline  # 只生成指定图
"""

import os, sys, subprocess, glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_SRC = os.path.join(PROJECT_ROOT, "source", "figures")
FIGURE_OUT = os.path.join(PROJECT_ROOT, "source", "_static", "figures")


def generate_one(name):
    """运行单个图的 Python 脚本，输出 SVG。"""
    script = os.path.join(FIGURE_SRC, f"{name}.py")
    if not os.path.exists(script):
        print(f"  ⚠ 找不到脚本: {script}")
        return False
    out = os.path.join(FIGURE_OUT, f"{name}.svg")
    os.makedirs(FIGURE_OUT, exist_ok=True)
    result = subprocess.run(
        [sys.executable, script, out], capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    if result.returncode == 0:
        print(f"  ✅ {name}.svg")
        return True
    else:
        print(f"  ❌ {name}: {result.stderr.strip()}")
        return False


def generate_all():
    """生成所有图脚本。"""
    scripts = sorted(glob.glob(os.path.join(FIGURE_SRC, "*.py")))
    total = 0
    for script in scripts:
        name = os.path.splitext(os.path.basename(script))[0]
        if name == "style":
            continue
        if generate_one(name):
            total += 1
    print(f"\n共生成 {total} 个图表")


def watch():
    """监听文件变动（需安装 watchdog）"""
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print("pip install watchdog 可实现文件监听")
        return

    class Handler(FileSystemEventHandler):
        def on_modified(self, event):
            if event.src_path.endswith(".py") and "figures" in event.src_path:
                name = os.path.splitext(os.path.basename(event.src_path))[0]
                if name != "style":
                    generate_one(name)

    observer = Observer()
    observer.schedule(Handler(), FIGURE_SRC, recursive=False)
    observer.start()
    print(f"监听 {FIGURE_SRC}，按 Ctrl+C 停止...")
    try:
        while observer.is_alive():
            observer.join(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    if "--watch" in sys.argv:
        watch()
    elif len(sys.argv) > 1 and sys.argv[1] not in ("--watch", "--help", "-h"):
        for name in sys.argv[1:]:
            if not name.startswith("-"):
                generate_one(name)
    else:
        generate_all()
