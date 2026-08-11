"""Lipla-jpのHugging Face Spaces向けGradioアプリ。"""

from __future__ import annotations

import gradio as gr

try:
    from .space_inference import recognize_image
except ImportError:  # Spaceでapp.pyを直接実行する場合
    from space_inference import recognize_image


def build_demo() -> gr.Blocks:
    """画像ドロップで推論を開始するGradio UIを構築する。"""
    with gr.Blocks(title="Lipla-jp") as demo:
        gr.Markdown(
            "# Lipla-jp\n"
            "日本の自動車ナンバープレートを検出・認識します。"
            "画像をドロップすると自動的に処理を開始します。"
        )

        with gr.Row():
            input_image = gr.Image(
                label="入力画像",
                type="numpy",
                image_mode="RGB",
                sources=["upload", "clipboard"],
            )
            det_gallery = gr.Gallery(
                label="検出位置",
                columns=1,
                object_fit="contain",
                height="auto",
            )

        with gr.Row():
            result_gallery = gr.Gallery(
                label="認識結果",
                columns=1,
                object_fit="contain",
                height="auto",
            )
            result_json = gr.Textbox(
                label="認識結果（JSON）",
                value="[]",
                lines=20,
                max_lines=30,
            )

        input_image.change(
            fn=recognize_image,
            inputs=input_image,
            outputs=[det_gallery, result_gallery, result_json],
            api_name="recognize",
        )

    return demo


demo = build_demo()
demo.queue(max_size=8)


if __name__ == "__main__":
    demo.launch()
