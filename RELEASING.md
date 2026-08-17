# Release process

リリースは、`main` に含まれるコミットへ `vMAJOR.MINOR.PATCH` 形式のタグを付けることで開始します。
GitHub Actions がタグ、プロジェクトバージョン、ブランチを検証し、テストとビルドに成功した場合だけ GitHub Release を作成します。

## 通常リリース

1. `feature/*` または `fix/*` を `develop` から作成し、Pull Request で `develop` へマージします。
2. リリース時に `pyproject.toml` の `project.version` を次のバージョンへ更新します。
3. `develop` から `main` への Pull Request を作成し、必須CIの成功後にマージします。
4. ローカルの `main` を更新し、マージされたコミットへ annotated tag を付けて push します。

```sh
git switch main
git pull --ff-only origin main
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
```

タグ `v0.2.0` と `pyproject.toml` のバージョン `0.2.0` は一致させてください。

## Hotfix リリース

1. `main` から `hotfix/*` を作成します。
2. 修正とバージョン更新を行い、Pull Request で `main` へマージします。
3. 通常リリースと同様に、更新後の `main` へタグを付けて push します。
4. リリース後、`main` の hotfix を `develop` にもマージして修正を取り込みます。

## GitHub の推奨設定

- `main` と `develop` を保護し、Pull Request 経由の変更と `Test and build` を必須チェックに設定する
- `main` への force push と削除を禁止する
- `v*` タグを tag protection rule / ruleset で保護し、作成・削除できる人を制限する
- GitHub Actions の既定権限は read-only にし、このリリースworkflowだけ `contents: write` を許可する

誤ったタグでは Release は作成されません。まだ push していないタグはローカルで作り直せます。push 済みタグは履歴の参照になるため、原則として削除・再利用せず、バージョンを上げて新しいタグを発行してください。
