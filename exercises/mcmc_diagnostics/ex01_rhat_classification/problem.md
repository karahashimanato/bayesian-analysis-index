# r_hat による収束判定

参照: [tools/mcmc-diagnostics.md — r_hat(Gelman-Rubin統計量)](../../../tools/mcmc-diagnostics.md#r_hatgelman-rubin統計量)

> **定義**: 複数のMCMCチェーンが同じ目標分布に収束しているかを測る指標。1.00に近いほど良く、慣習的に**1.01未満**が目安とされる。

## 課題

`solution.py` の `classify_rhat` を実装する。

```python
def classify_rhat(rhat_values: dict[str, float]) -> dict[str, str]:
    ...
```

- 入力: パラメータ名 → r_hat値 の辞書(`az.rhat()` の出力を想定)
- 出力: パラメータ名 → 判定文字列 の辞書
  - r_hat が **1.01 未満** → `"converged"`
  - それ以外(1.01 以上)→ `"not_converged"`

境界値 `1.01` ちょうどは「1.01未満」を満たさないため `"not_converged"` とする。

## 進め方

1. `solution.py` の `NotImplementedError` を実装で置き換える。
2. ローカルで確認: リポジトリルートから `pytest exercises/mcmc_diagnostics/ex01_rhat_classification -v`
3. すべて green になったら commit して push すると、GitHub Actions が同じテストを自動実行する。
