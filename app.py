from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fake_news_detector.app_logic import build_demo


demo = build_demo()


if __name__ == "__main__":
    host = os.getenv("FAKE_NEWS_HOST", "127.0.0.1")
    port = int(os.getenv("FAKE_NEWS_PORT", "7860"))
    print(f"Starting Fake News Detector at http://{host}:{port}")
    demo.launch(
        server_name=host,
        server_port=port,
        inbrowser=True,
        share=False,
        quiet=False,
    )
