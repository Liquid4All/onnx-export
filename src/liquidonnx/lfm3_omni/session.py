"""
LFM3-Omni generation on onnxruntime: lfm3_audio.session.ONNXAudioModel (speech in, the text /
speech mode machine, MaskGIT, the codec) with LFM3-VL's vision encoder in front of the decoder.

Images go through lfm3_vl.session's Qwen2VLImageProcessor (preprocessor_config.json, which
lfm3-omni-export writes from processor_config.json's image_processor) and run_vision; the chat
template writes <|vision_start|><|image|><|vision_end|> per image and
<|vision_start|><|audio|><|vision_end|> per clip, and prompt_ids expands each <|image|> to its
grid's tokens and each <|audio|> to its clip's. The decoder scatters image_features over the
<|image|> placeholders at prefill, as audio_features over <|audio|>. Video runs through the same
graphs (run_vision of the video grids, video_features), which the tests cover; lfm3-omni-infer
takes images only.
"""

import json
import pathlib

import numpy as np

from liquidonnx.lfm3_audio.session import BUNDLES as AUDIO_BUNDLES
from liquidonnx.lfm3_audio.session import ONNXAudioModel
from liquidonnx.lfm3_omni.config import parse_vision
from liquidonnx.lfm3_vl.session import BUNDLES as VL_BUNDLES
from liquidonnx.lfm3_vl.session import (
    expand_image_tokens,
    image_processor,
    preprocess_images,
    run_vision,
)

# precision -> graph role -> file: LFM3-Audio's bundle and LFM3-VL's vision encoder (int8 in q4,
# fp16 in the other quantized bundles)
BUNDLES = {
    precision: {**files, "vision_encoder": VL_BUNDLES[precision][1]}
    for precision, files in AUDIO_BUNDLES.items()
}


class ONNXOmniModel(ONNXAudioModel):
    bundles = BUNDLES

    def __init__(self, model_dir: str | pathlib.Path, *args, **kwargs):
        """ONNXAudioModel's arguments; also parses the vision config (self.vision_config,
        self.vision_tokens)."""
        super().__init__(model_dir, *args, **kwargs)
        raw = json.loads((self.model_dir / "config.json").read_text())
        self.vision_config, self.vision_tokens = parse_vision(raw)

    def load_preprocessing(self):
        """ONNXAudioModel's tokenizer and mel filters, and the image processor."""
        super().load_preprocessing()
        self.image_processor = image_processor(self.model_dir)

    def encode_images(
        self, images: list, max_image_tokens: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """(image_features [tokens, H], grids [n, 3]) of PIL images."""
        pixel_values, grids = preprocess_images(self.image_processor, images, max_image_tokens)
        merge = self.vision_config.merge_size
        return run_vision(self.vision_encoder, pixel_values, grids, merge), grids

    def prompt_ids(
        self,
        messages: list[dict],
        audio_token_counts: list[int],
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        grids: np.ndarray | None = None,
    ) -> list[int]:
        """ONNXAudioModel.prompt_ids (messages' images as {"type": "image"}), with each <|image|>
        expanded to its grid's tokens (grids of encode_images, in prompt order)."""
        ids = super().prompt_ids(
            messages, audio_token_counts, add_generation_prompt, continue_final_message
        )
        grids = np.zeros((0, 3), dtype=np.int64) if grids is None else grids
        return expand_image_tokens(
            ids, grids, self.vision_tokens.image, self.vision_config.merge_size
        )
