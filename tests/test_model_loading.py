from pathlib import Path

from lipla.inferencers import model_loader


def test_download_model_file_is_anonymous_and_revision_pinned(
    monkeypatch, tmp_path
):
    calls = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / kwargs["filename"])

    monkeypatch.setattr(
        model_loader, "hf_hub_download", fake_hf_hub_download
    )

    result = model_loader.download_model_file(
        model_loader.ECPOSE_MODEL_FILENAME
    )

    assert result == tmp_path / model_loader.ECPOSE_MODEL_FILENAME
    assert calls == [
        {
            "repo_id": "bukuroo/Lipla-jp",
            "filename": "ecpose_m_260809.onnx",
            "revision": "c66f50ce0cc08e20318b00ad832c9b848b4d580b",
            "cache_dir": None,
            "token": False,
            "local_files_only": False,
        }
    ]


class _ModelInput:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _ModelOutput:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _ModelMetadata:
    custom_metadata_map = {}


class _PoseSession:
    def __init__(self, path, *_args, **_kwargs):
        self.path = path

    def get_inputs(self):
        return [_ModelInput("images", [1, 3, 320, 320])]

    def get_outputs(self):
        return [
            _ModelOutput("scores", [1, 100]),
            _ModelOutput("labels", [1, 100]),
            _ModelOutput("keypoints", [1, 100, 4, 2]),
        ]

    def get_modelmeta(self):
        return _ModelMetadata()


def test_ecpose_downloads_default_model(monkeypatch, tmp_path):
    from lipla.inferencers import ec_pose

    model_path = tmp_path / model_loader.ECPOSE_MODEL_FILENAME
    calls = []
    monkeypatch.setattr(
        ec_pose,
        "download_model_file",
        lambda filename, **kwargs: calls.append((filename, kwargs))
        or model_path,
    )
    monkeypatch.setattr(ec_pose, "create_inference_session", _PoseSession)

    inferencer = ec_pose.ECPose()

    assert inferencer.session.path == str(model_path)
    assert calls == [
        (
            model_loader.ECPOSE_MODEL_FILENAME,
            {
                "cache_dir": None,
                "revision": model_loader.MODEL_REVISION,
                "local_files_only": False,
            },
        )
    ]


class _OCRSession:
    def __init__(self, path, **_kwargs):
        self.path = path

    def get_inputs(self):
        return [_ModelInput("input", None)]


class _Decoder:
    def __init__(self, dict_path, characters_path, *, new_area_names=None):
        self.dict_path = Path(dict_path)
        self.characters_path = Path(characters_path)
        self.new_area_names = new_area_names


def test_ppocr_downloads_unspecified_assets(monkeypatch, tmp_path):
    from lipla.inferencers import ppocr

    calls = []

    def fake_download(filename, **kwargs):
        calls.append((filename, kwargs))
        return tmp_path / filename

    monkeypatch.setattr(ppocr, "download_model_file", fake_download)
    monkeypatch.setattr(ppocr, "create_inference_session", _OCRSession)
    monkeypatch.setattr(ppocr, "CTCDecoder", _Decoder)

    inferencer = ppocr.PPOCR()

    assert inferencer.det_session.path == str(
        tmp_path / model_loader.PPOCR_DET_MODEL_FILENAME
    )
    assert inferencer.rec_session.path == str(
        tmp_path / model_loader.PPOCR_REC_MODEL_FILENAME
    )
    assert inferencer.decoder.dict_path == (
        tmp_path / model_loader.PPOCR_DICT_FILENAME
    )
    assert inferencer.decoder.characters_path.name == "characters.yml"
    assert inferencer.decoder.new_area_names is None
    assert [filename for filename, _ in calls] == [
        model_loader.PPOCR_DET_MODEL_FILENAME,
        model_loader.PPOCR_REC_MODEL_FILENAME,
        model_loader.PPOCR_DICT_FILENAME,
    ]


def test_explicit_model_paths_do_not_download(monkeypatch, tmp_path):
    from lipla.inferencers import ppocr

    def fail_download(*_args, **_kwargs):
        raise AssertionError("download should not be called")

    monkeypatch.setattr(ppocr, "download_model_file", fail_download)
    monkeypatch.setattr(ppocr, "create_inference_session", _OCRSession)
    monkeypatch.setattr(ppocr, "CTCDecoder", _Decoder)

    ppocr.PPOCR(
        tmp_path / "det.onnx",
        tmp_path / "rec.onnx",
        tmp_path / "dict.yml",
        tmp_path / "characters.yml",
    )


def test_ppocr_passes_new_area_names_to_decoder(monkeypatch, tmp_path):
    from lipla.inferencers import ppocr

    monkeypatch.setattr(ppocr, "create_inference_session", _OCRSession)
    monkeypatch.setattr(ppocr, "CTCDecoder", _Decoder)

    inferencer = ppocr.PPOCR(
        tmp_path / "det.onnx",
        tmp_path / "rec.onnx",
        tmp_path / "dict.yml",
        tmp_path / "characters.yml",
        new_area_names=["札幌新"],
    )

    assert inferencer.decoder.new_area_names == ["札幌新"]
