"""techniques/・tools/配下のMarkdownから、タイピング練習(typing-practice/)用のお題を抽出してJSON化する。

抽出対象:
  - フェンス付き```python```ブロック全体(複数行の「お題」)
  - インラインの`...`コードスパン(短い「お題」。PyMCの関数呼び出し・パラメータ名など)

インラインスパンはヒューリスティックでコードらしいものだけを残す(日本語や
ファイルパス・素のプレーンワードは除外)。3件未満しか残らない章はサイトに出しても
意味が薄いため出力から除く。
"""
import json
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "typing-practice" / "data" / "snippets.js"

CHAPTERS = [
    ("techniques/eda.md", "モデリング前のEDA(探索的データ分析)"),
    ("techniques/prior-predictive-check.md", "事前分布設計・prior predictive check"),
    ("techniques/observation-model.md", "尤度・観測モデル選択"),
    ("techniques/reparameterization.md", "パラメータ化・非識別性対策"),
    ("techniques/diagnostics.md", "診断・収束判定"),
    ("techniques/model-evaluation.md", "モデル評価・比較"),
    ("techniques/data-pitfalls.md", "データ・単位・前処理の落とし穴"),
    ("techniques/implementation-hacks.md", "実装上のハック"),
    ("tools/evaluation-metrics.md", "評価指標・推定量"),
    ("tools/observation-models.md", "観測モデル・尤度分布"),
    ("tools/greek-letters.md", "ギリシャ文字の用途一覧"),
    ("tools/mcmc-diagnostics.md", "MCMC診断指標"),
    ("tools/inference-methods.md", "推論エンジン・サンプリング手法"),
    ("tools/posterior-pathologies.md", "事後分布の幾何学的病理"),
    ("tools/statistical-biases.md", "統計的バイアス・概念"),
    ("tools/state-space-models.md", "状態空間モデルの型"),
    ("tools/prior-distributions.md", "事前分布の選び方"),
    ("tools/spatial-models.md", "空間モデルの型"),
    ("tools/missing-data.md", "欠測データ処理の型"),
    ("tools/acquisition-functions.md", "獲得関数の型"),
    ("tools/pymc-code-patterns.md", "PyMC/ArviZコーディングパターン"),
]

FENCE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
INLINE_RE = re.compile(r"`([^`\n]+)`")
CODE_HINT_RE = re.compile(r"[_.()\[\]=:]|\d")
FILENAME_RE = re.compile(r"\.(md|ipynb|py)$")

MIN_LEN = 4
MAX_LEN = 80
MIN_PROBLEMS = 3


def strip_fences(text: str) -> str:
    return FENCE_RE.sub("", text)


def looks_like_code(span: str) -> bool:
    if not (MIN_LEN <= len(span) <= MAX_LEN):
        return False
    if not span.isascii():
        return False
    if FILENAME_RE.search(span):
        return False
    if not re.search(r"[A-Za-z]", span):
        return False
    if not CODE_HINT_RE.search(span):
        return False
    return True


def extract_chapter(path: Path):
    text = path.read_text(encoding="utf-8")

    blocks = []
    for m in FENCE_RE.finditer(text):
        body = textwrap.dedent(m.group(1)).strip("\n")
        if body.strip():
            blocks.append(body)

    inline_source = strip_fences(text)
    seen = set()
    spans = []
    for m in INLINE_RE.finditer(inline_source):
        span = m.group(1).strip()
        if span in seen:
            continue
        if not looks_like_code(span):
            continue
        seen.add(span)
        spans.append(span)

    return blocks, spans


def main():
    chapters_out = []
    total_blocks = 0
    total_spans = 0
    for rel_path, title in CHAPTERS:
        path = ROOT / rel_path
        blocks, spans = extract_chapter(path)
        problems = [{"type": "block", "text": b} for b in blocks]
        problems += [{"type": "line", "text": s} for s in spans]
        if len(problems) < MIN_PROBLEMS:
            print(f"skip  {rel_path}: {len(problems)}件 (閾値{MIN_PROBLEMS}未満)")
            continue
        chapters_out.append({
            "id": rel_path.split("/")[-1].removesuffix(".md"),
            "title": title,
            "source": rel_path,
            "problems": problems,
        })
        total_blocks += len(blocks)
        total_spans += len(spans)
        print(f"ok    {rel_path}: block={len(blocks)} line={len(spans)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"chapters": chapters_out}, ensure_ascii=False, indent=2)
    # file://での直接閲覧時にfetch()のCORS制限を踏まないよう、JSONではなく
    # <script>で読み込めるJSファイルとして書き出す。
    OUT_PATH.write_text(
        f"// このファイルは scripts/generate_typing_snippets.py が自動生成する。直接編集しない。\n"
        f"window.SNIPPETS_DATA = {payload};\n",
        encoding="utf-8",
    )
    print(f"\n書き出し: {OUT_PATH}")
    print(f"章数: {len(chapters_out)} / block合計: {total_blocks} / line合計: {total_spans}")


if __name__ == "__main__":
    main()
