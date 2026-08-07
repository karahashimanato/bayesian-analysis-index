"""複数の生成スクリプトで共有する配色・matplotlib設定。"""

import matplotlib.pyplot as plt

# ドキュメント全体で色の意味を統一する
COLOR_OK = "#4C72B0"         # 通常のドロー(非発散)
COLOR_DIVERGENT = "#D55E00"  # divergence を起こしたドロー
COLOR_CHAIN = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]  # chainごとの色
COLOR_ALT = "#8172B2"        # 比較対象(ADVI・別モデルなど)の強調色


def apply_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 11,
            "axes.titlesize": 12,
            "font.family": "IPAGothic",  # 日本語ラベルの文字化け(tofu)を防ぐ(bold書体は無いため通常太さを使用)
            "axes.unicode_minus": False,
        }
    )
