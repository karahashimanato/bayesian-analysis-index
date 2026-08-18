# 事前分布を見分けるフローチャートの実装

参照: [tools/prior-distributions.md — 事前分布を見分けるフローチャート](../../../tools/prior-distributions.md#事前分布を見分けるフローチャート)

ドキュメント冒頭の mermaid フローチャートをそのまま関数として実装する。

## 課題

`solution.py` の `pick_prior_family` を実装する。

```python
def pick_prior_family(kind: str, detail: str | None = None) -> str:
    ...
```

`kind` と `detail` の組み合わせから分布族を返す(フローチャートの分岐そのまま):

| kind | detail | 戻り値 |
|---|---|---|
| `"probability"` | (不要) | `"Beta"` |
| `"positive_scale"` | `"avoid_zero_variance_explosion"` | `"Gamma"` |
| `"positive_scale"` | `"no_constraint"` | `"HalfNormal_or_HalfCauchy"` |
| `"changepoint"` | `"avoid_ess_drop"` | `"Uniform_plus_sigmoid"` |
| `"changepoint"` | `"discrete_ok"` | `"DiscreteUniform"` |
| `"real_unconstrained"` | (不要) | `"Normal"` |
| `"wide_magnitude_positive"` | (不要) | `"LogNormal"` |

- `kind` が上記のいずれでもない場合、または `detail` が必要なのに未指定・不正な値の場合は `ValueError` を送出する。

## 進め方

1. `solution.py` の `NotImplementedError` を実装で置き換える。
2. ローカルで確認: `pytest exercises/prior_distributions/ex01_pick_prior_family -v`
3. push すると GitHub Actions が自動採点する。
