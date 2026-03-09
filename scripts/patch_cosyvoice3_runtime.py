#!/usr/bin/env python3
"""Patch the pinned CosyVoice runtime for Dictator's offline image contract."""

from __future__ import annotations

import sys
from pathlib import Path


IMPORT_OLD = "import os\nimport re\nimport inflect\n"
IMPORT_NEW = "import os\nimport re\nfrom pathlib import Path\nimport inflect\n"

WETEXT_OLD = """        except:\n            try:\n                from wetext import Normalizer as ZhNormalizer\n                from wetext import Normalizer as EnNormalizer\n                self.zh_tn_model = ZhNormalizer(remove_erhua=False)\n                self.en_tn_model = EnNormalizer()\n                self.text_frontend = 'wetext'\n                logging.info('use wetext frontend')\n            except:\n                self.text_frontend = ''\n                logging.info('no frontend is avaliable')\n"""

WETEXT_NEW = """        except:\n            try:\n                from wetext import Normalizer as WetextNormalizer\n                modelscope_cache = Path(os.environ.get('MODELSCOPE_CACHE', str(Path.home() / '.cache' / 'modelscope')))\n                wetext_root = Path(os.environ.get('WETEXT_MODEL_DIR', str(modelscope_cache / 'hub' / 'pengzhendong' / 'wetext')))\n                zh_tagger = wetext_root / 'zh' / 'tn' / 'tagger.fst'\n                zh_verbalizer = wetext_root / 'zh' / 'tn' / 'verbalizer_remove_erhua.fst'\n                en_tagger = wetext_root / 'en' / 'tn' / 'tagger.fst'\n                en_verbalizer = wetext_root / 'en' / 'tn' / 'verbalizer.fst'\n                required_paths = (zh_tagger, zh_verbalizer, en_tagger, en_verbalizer)\n                missing_paths = [str(path) for path in required_paths if not path.is_file()]\n                if missing_paths:\n                    raise FileNotFoundError(f'missing wetext files: {missing_paths}')\n                self.zh_tn_model = WetextNormalizer(\n                    tagger_path=str(zh_tagger),\n                    verbalizer_path=str(zh_verbalizer),\n                    lang='zh',\n                )\n                self.en_tn_model = WetextNormalizer(\n                    tagger_path=str(en_tagger),\n                    verbalizer_path=str(en_verbalizer),\n                    lang='en',\n                )\n                self.text_frontend = 'wetext'\n                logging.info('use wetext frontend from %s', wetext_root)\n            except Exception:\n                self.text_frontend = ''\n                logging.info('no frontend is avaliable')\n"""


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_cosyvoice3_runtime.py <cosyvoice-root>", file=sys.stderr)
        return 2
    frontend_path = Path(sys.argv[1]) / "cosyvoice" / "cli" / "frontend.py"
    source = frontend_path.read_text()
    if IMPORT_NEW in source and WETEXT_NEW in source:
        return 0
    if IMPORT_OLD not in source:
        raise SystemExit(f"expected import block not found in {frontend_path}")
    source = source.replace(IMPORT_OLD, IMPORT_NEW, 1)
    if WETEXT_OLD not in source:
        raise SystemExit(f"expected wetext block not found in {frontend_path}")
    source = source.replace(WETEXT_OLD, WETEXT_NEW, 1)
    frontend_path.write_text(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
