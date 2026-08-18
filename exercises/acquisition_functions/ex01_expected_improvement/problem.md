# EI (Expected Improvement) の実装

参照: [tools/acquisition-functions.md — EI (Expected Improvement)](../../../tools/acquisition-functions.md#ei-expected-improvement)

> **数式・仕組み**: `EI(x) = (μ(x)-f_best)・Φ(z) + σ(x)・φ(z)`、`z = (μ(x)-f_best)/σ(x)`(`φ`は標準正規分布の密度関数)。

## 課題

`solution.py` の `expected_improvement` を実装する。

```python
def expected_improvement(mu: float, sigma: float, f_best: float) -> float:
    ...
```

- `mu`, `sigma`: GP事後分布の平均・標準偏差(候補点 `x` における値)
- `f_best`: 現在のベスト観測値
- `Φ`(標準正規分布のCDF)・`φ`(標準正規分布のPDF)は `scipy.stats.norm` を使ってよい

**実装上の注意**(ドキュメントの数式には明記されていないが、数値的に必要な処理):
- `sigma == 0` のとき `z` が定義できない(ゼロ除算)。この場合は `EI = 0.0` を返す(不確実性がゼロなら改善の余地を評価できないため、という一般的な規約。本リポジトリのドキュメント自体には記載がない実装上の取り決め)。

## 進め方

1. `solution.py` の `NotImplementedError` を実装で置き換える。
2. ローカルで確認: `pytest exercises/acquisition_functions/ex01_expected_improvement -v`
3. push すると GitHub Actions が自動採点する。
