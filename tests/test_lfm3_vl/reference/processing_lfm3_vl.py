# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import numpy as np

from transformers.image_utils import is_pil_image, is_valid_image, make_flat_list_of_images
from transformers.processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin
from transformers.utils import auto_docstring, is_numpy_array, is_torch_tensor, logging
from transformers.video_utils import VideoMetadata, is_valid_video, make_batched_videos


logger = logging.get_logger(__name__)


class Lfm3VlProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": False,
        },
        "videos_kwargs": {"return_metadata": True},
    }


def _check_video_fps(metadata: VideoMetadata) -> None:
    if metadata.fps is not None and not (math.isfinite(metadata.fps) and metadata.fps > 0):
        raise ValueError(f"`video_metadata.fps` must be a positive number, got {metadata.fps}")


def _media_per_prompt(media, is_item) -> list[int] | None:
    # rows name the prompt that owns them while the expansion consumes media in flat order; a row is a list of whole
    # items or one stacked batch, and a list of single frames is one video, not a row
    if not isinstance(media, (list, tuple)):
        return None
    counts = []
    for row in media:
        if isinstance(row, (list, tuple)) and all(is_item(item) for item in row):
            counts.append(len(row))
        elif (is_numpy_array(row) or is_torch_tensor(row)) and row.ndim == 4 and (len(row) == 0 or is_item(row[0])):
            counts.append(len(row))
        else:
            return None
    return counts


def _check_vision_placeholder_counts(processor, prompts, images, videos) -> None:
    for name, token, media, flatten, is_item in (
        (
            "images",
            processor.image_token,
            images,
            make_flat_list_of_images,
            lambda item: is_valid_image(item) and (is_pil_image(item) or item.ndim == 3),
        ),
        (
            "videos",
            processor.video_token,
            videos,
            make_batched_videos,
            lambda item: isinstance(item, str) or (not is_pil_image(item) and is_valid_video(item)),
        ),
    ):
        num_placeholders = sum(prompt.count(token) for prompt in prompts)
        num_media = 0 if media is None else len(flatten(media))
        if num_placeholders != num_media:
            raise ValueError(f"Found {num_placeholders} {token} tokens in the text but received {num_media} {name}")
        # one prompt cannot take another's media
        per_prompt = _media_per_prompt(media, is_item) if len(prompts) > 1 else None
        in_text = [prompt.count(token) for prompt in prompts]
        if per_prompt is not None and per_prompt != in_text:
            raise ValueError(f"Found {in_text} {token} tokens per prompt but received {per_prompt} {name} per prompt")


@auto_docstring
class Lfm3VlProcessor(ProcessorMixin):
    valid_processor_kwargs = Lfm3VlProcessorKwargs

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        self.image_token = getattr(tokenizer, "image_token", "<|image|>")
        self.video_token = getattr(tokenizer, "video_token", "<|video|>")
        self.vision_start_token = getattr(tokenizer, "vision_start_token", "<|vision_start|>")
        self.vision_end_token = getattr(tokenizer, "vision_end_token", "<|vision_end|>")
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        self.vision_start_token_id = tokenizer.convert_tokens_to_ids(self.vision_start_token)
        self.vision_end_token_id = tokenizer.convert_tokens_to_ids(self.vision_end_token)
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        merge_length = self.image_processor.merge_size**2
        num_image_tokens = image_inputs["image_grid_thw"][image_idx].prod() // merge_length
        return self.image_token * num_image_tokens

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        metadata = video_inputs["video_metadata"][video_idx]
        if metadata.frames_indices is None:
            raise ValueError("Frame timestamps need the sampled frame indices in `video_metadata`")
        _check_video_fps(metadata)
        merge_length = self.video_processor.merge_size**2
        num_frames = video_inputs["video_grid_thw"][video_idx][0]
        frame_seqlen = video_inputs["video_grid_thw"][video_idx][1:].prod() // merge_length
        metadata = video_inputs["video_metadata"][video_idx]
        video_placeholder = ""

        if metadata.fps is None:
            logger.warning_once(
                "Lfm3Vl requires frame timestamps to construct prompts, but the `fps` of the input video could not be inferred. "
                "Probably `video_metadata` was missing from inputs and you passed pre-sampled frames. "
                "Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results."
            )
        # a released transformers reads torchcodec's scanned average fps, which carries float noise
        # (30.000000000000004) where the fork reads the container header
        metadata.fps = 24 if metadata.fps is None else round(metadata.fps, 6)

        # if timestamps are not provided, calculate them
        curr_timestamp = self._calculate_timestamps(
            metadata.frames_indices,
            metadata.fps,
            self.video_processor.temporal_patch_size,
        )

        for frame_idx in range(num_frames):
            curr_time = curr_timestamp[frame_idx]
            video_placeholder += f"<{curr_time:.1f} seconds>"
            video_placeholder += self.vision_start_token + self.video_token * frame_seqlen + self.vision_end_token
        return video_placeholder

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.

        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.
            video_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (num_frames, height, width) per each video.

        Returns:
            `MultiModalData`: the number of tokens per provided modality.
        """
        vision_data = {}
        if image_sizes is not None:
            # copy: `update` on the class-level defaults would leak kwargs into every later call
            images_kwargs = dict(Lfm3VlProcessorKwargs._defaults.get("images_kwargs", {}))
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get("merge_size") or self.image_processor.merge_size

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            num_image_tokens = [(num_patches // merge_size**2) for num_patches in num_image_patches]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        if video_sizes is not None:
            videos_kwargs = dict(Lfm3VlProcessorKwargs._defaults.get("videos_kwargs", {}))
            videos_kwargs.update(kwargs)
            merge_size = videos_kwargs.get("merge_size") or self.video_processor.merge_size
            num_video_patches = [
                self.video_processor.get_num_of_video_patches(*video_size, videos_kwargs) for video_size in video_sizes
            ]
            num_video_tokens = [(num_patches // merge_size**2) for num_patches in num_video_patches]
            vision_data["num_video_tokens"] = num_video_tokens

        return MultiModalData(**vision_data)

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the tokenization spaces. Argument passed to the tokenizer's `batch_decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self):
        return list(
            dict.fromkeys(
                self.tokenizer.model_input_names
                + self.image_processor.model_input_names
                + self.video_processor.model_input_names
            )
        )

    def _calculate_timestamps(self, indices: list[int] | np.ndarray, video_fps: float, merge_size: int = 2):
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
        timestamps = [idx / video_fps for idx in indices]
        # @JJJYmmm frames are merged by self.merge_size, \
        # so we need to average the timestamps between the first/last frame within the temporal patch
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps

    def prepare_inputs_layout(self, images=None, text=None, videos=None, audio=None, **kwargs):
        for name, media in (("images", images), ("videos", videos)):
            if isinstance(media, (list, tuple)) and len(media) == 0:
                raise ValueError(f"`{name}` must hold at least one item; pass `{name}=None` for prompts without one")
        return super().prepare_inputs_layout(images=images, text=text, videos=videos, audio=audio, **kwargs)

    def validate_inputs(self, images=None, text=None, videos=None, audio=None, **kwargs):
        super().validate_inputs(images=images, text=text, videos=videos, audio=audio, **kwargs)
        if text is not None:
            _check_vision_placeholder_counts(self, [text] if isinstance(text, str) else text, images, videos)

    def _process_videos(self, videos, **kwargs):
        # the frame timestamps need the metadata whatever the caller's `return_metadata`; `__call__` drops it from
        # the output unless it was asked for
        kwargs["return_metadata"] = True
        return super()._process_videos(videos, **kwargs)


__all__ = ["Lfm3VlProcessor"]
