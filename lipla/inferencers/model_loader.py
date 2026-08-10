"""公開Hugging FaceリポジトリからLiplaのモデルを取得する。"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from huggingface_hub import hf_hub_download

MODEL_REPO_ID: Final = "bukuroo/Lipla-jp"
# Pin the default assets to an immutable commit so the same Lipla release uses
# the same model files in every environment.
MODEL_REVISION: Final = "c66f50ce0cc08e20318b00ad832c9b848b4d580b"

ECPOSE_MODEL_FILENAME: Final = "ecpose_m_260809.onnx"
PPOCR_DET_MODEL_FILENAME: Final = "ppocrv6_det.onnx"
PPOCR_REC_MODEL_FILENAME: Final = "ppocrv6_rec.onnx"
PPOCR_DICT_FILENAME: Final = "inference.yml"


def download_model_file(
    filename: str,
    *,
    cache_dir: str | Path | None = None,
    revision: str = MODEL_REVISION,
    local_files_only: bool = False,
) -> Path:
    """Liplaモデルリポジトリ内のファイルを取得する。

    インストール済みパッケージ内ではなく、Hugging Faceのバージョン対応
    ユーザーキャッシュを利用する。公開リポジトリへ匿名で接続し、ローカルに
    設定された認証情報は使用しない。

    Args:
        filename: モデルリポジトリ内のファイル名。
        cache_dir: Hugging Faceキャッシュの保存先。
        revision: 取得するコミットまたはリビジョン。
        local_files_only: ネットワークへ接続せずキャッシュだけを利用するか。

    Returns:
        取得済みファイルのローカルパス。
    """

    path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
        token=False,
        local_files_only=local_files_only,
    )
    return Path(path)
