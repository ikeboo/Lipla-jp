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
  - bukuroo/Lipla-jp ecpose_m_260809.onnx,ppocrv6_det.onnx,ppocrv6_rec.onnx,inference.yml c66f50ce0cc08e20318b00ad832c9b848b4d580b
---

# Lipla-jp Gradio demo

画像をドロップすると、日本の自動車ナンバープレートを検出・認識します。

- `LPDetResult.det_image` と `LPDetResult.result_image` をギャラリー表示します。
- `LPDetResult` の画像以外のフィールドをJSONテキストで表示します。
- 複数のナンバープレートを検出した場合は、結果を検出順に表示します。

このディレクトリの内容をHugging Face Spaceリポジトリのルートへ配置して
ZeroGPUで公開してください。Spaceのビルド時にモデルと日本語フォントが準備され、
推論時にZeroGPUが割り当てられます。
