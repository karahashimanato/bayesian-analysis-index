# 演習問題(コーディングテスト)

`techniques/`・`tools/`の各エントリの理解度を、実際にコードを書いて確認するための自己学習用演習。

## 構成

演習は `exercises/<トピック>/ex<番号>_<課題名>/` ごとに以下の3ファイルを持つ。

| ファイル | 役割 |
|---|---|
| `problem.md` | 課題の説明。根拠となる`tools/`・`techniques/`のエントリへのリンク付き |
| `solution.py` | 実装対象の関数スタブ(`NotImplementedError`を実装で置き換える) |
| `test_solution.py` | 採点用のpytestテスト(このファイルは編集しない) |

## 解き方

1. 好きな演習の `problem.md` を読む
2. `solution.py` を実装する
3. リポジトリルートで該当ディレクトリを指定してpytestを実行し、ローカルで確認する

   ```bash
   pip install -r exercises/requirements.txt
   pytest exercises/mcmc_diagnostics/ex01_rhat_classification -v
   ```

   全演習をまとめて実行する場合:

   ```bash
   pytest exercises -v
   ```

4. すべて green になったら commit して push(またはPRを作成)する。[GitHub Actions](../.github/workflows/exercises.yml) が同じテストを自動実行し、結果がチェックとジョブサマリーに表示される。

## 演習一覧

| トピック | 演習 | 参照ドキュメント |
|---|---|---|
| MCMC診断指標 | [r_hatによる収束判定](mcmc_diagnostics/ex01_rhat_classification/problem.md) | [tools/mcmc-diagnostics.md](../tools/mcmc-diagnostics.md#r_hatgelman-rubin統計量) |
| 事前分布の選び方 | [フローチャートの実装](prior_distributions/ex01_pick_prior_family/problem.md) | [tools/prior-distributions.md](../tools/prior-distributions.md#事前分布を見分けるフローチャート) |
| 獲得関数の型 | [EIの実装](acquisition_functions/ex01_expected_improvement/problem.md) | [tools/acquisition-functions.md](../tools/acquisition-functions.md#ei-expected-improvement) |

演習は既存の`techniques/`・`tools/`エントリと同じ粒度で今後追加していく想定(1エントリ = 1演習)。
