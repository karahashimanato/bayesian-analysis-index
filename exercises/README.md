# 演習問題(コーディングテスト)

`techniques/`・`tools/`の各エントリの理解度を、実際にコードを書いて確認するための自己学習用演習。

## 構成

演習は `exercises/<トピック>/ex<番号>_<課題名>/` ごとに以下の3ファイルを持つ。

| ファイル | 役割 |
|---|---|
| `problem.md` | 課題の説明。根拠となる`tools/`・`techniques/`のエントリへのリンク付き |
| `solution.py` | 実装対象の関数スタブ(`NotImplementedError`を実装で置き換える) |
| `test_solution.py` | 採点用のpytestテスト(このファイルは編集しない) |

## 解き方(オンデマンド採点)

1. 好きな演習の `problem.md` を読む
2. `solution.py` を実装する
3. `exercises/grade.py` でその場で採点する(git操作は不要)

   ```bash
   pip install -r exercises/requirements.txt

   # 演習一覧を表示(名前の一部を指定して採点する)
   python exercises/grade.py

   # 名前の一部を指定して1問だけ採点
   python exercises/grade.py rhat_classification

   # 全演習を採点
   python exercises/grade.py all
   ```

   pass/failがその場で表示される(passなら終了コード0、failなら1)。git pull/pushは不要で、何度でも即座に採点し直せる。

`pytest`を直接使っても同じ結果になる(`pytest exercises/mcmc_diagnostics/ex01_rhat_classification -v`)。

## GitHub Actionsでの採点(任意)

commitしてpush(またはPR作成)すると、[GitHub Actions](../.github/workflows/exercises.yml) が同じテストを自動実行し、結果がチェックとジョブサマリーに表示される。ローカルで解き終えた演習を記録として残したい場合に使う。オンデマンド採点だけで完結させたい場合はpush自体不要。

## 演習一覧

| トピック | 演習 | 参照ドキュメント |
|---|---|---|
| MCMC診断指標 | [r_hatによる収束判定](mcmc_diagnostics/ex01_rhat_classification/problem.md) | [tools/mcmc-diagnostics.md](../tools/mcmc-diagnostics.md#r_hatgelman-rubin統計量) |
| 事前分布の選び方 | [フローチャートの実装](prior_distributions/ex01_pick_prior_family/problem.md) | [tools/prior-distributions.md](../tools/prior-distributions.md#事前分布を見分けるフローチャート) |
| 獲得関数の型 | [EIの実装](acquisition_functions/ex01_expected_improvement/problem.md) | [tools/acquisition-functions.md](../tools/acquisition-functions.md#ei-expected-improvement) |

演習は既存の`techniques/`・`tools/`エントリと同じ粒度で今後追加していく想定(1エントリ = 1演習)。
