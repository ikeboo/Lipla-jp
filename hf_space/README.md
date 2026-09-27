---
title: Lipla-jp
emoji: 🚘
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.20.0
python_version: 3.12
app_file: app.py
pinned: false
license: mit
models:
  - bukuroo/Lipla-jp
preload_from_hub:
  - bukuroo/Lipla-jp pose_m_260927.onnx,ppocrv6_det.onnx,ppocrv6_rec.onnx,inference.yml 947afab7678ff50d7101e5cf802de25e9d4d0fe1
---

# Lipla-jp Gradio demo

画像をドロップすると、日本の自動車ナンバープレートを検出・認識します。

- `Result.det_image` と `Result.result_image` をギャラリー表示します。
- `Result` の画像以外のフィールドをJSONテキストで表示します。
- 複数のナンバープレートを検出した場合は、結果を検出順に表示します。

このディレクトリの内容をHugging Face Spaceリポジトリのルートへ配置して
ZeroGPUで公開してください。Spaceのビルド時にモデルと日本語フォントが準備され、
推論時にZeroGPUが割り当てられます。
