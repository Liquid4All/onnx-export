"""Minimal OpenAI-compatible server for LFM2-VL models.

Usage:
    uv run lfm2-vl-server --model exports/LFM2-VL-450M-ONNX --port 8000
"""

import argparse
import base64
import io
import json
import logging
import time
import uuid
from pathlib import Path

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel
from transformers import AutoProcessor

from liquidonnx.lfm2_vl.preprocessing import (
    build_inputs_embeds,
    detect_vision_format,
    get_image_embeddings,
    get_image_token_id,
    pad_to_square,
)
from liquidonnx.session import initialize_cache, load_onnx_session, update_cache

logger = logging.getLogger(__name__)

# === Pydantic Models ===


class ImageUrl(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: str
    text: str | None = None
    image_url: ImageUrl | None = None


class Message(BaseModel):
    role: str
    content: str | list[ContentPart]


class ChatCompletionRequest(BaseModel):
    model: str = "lfm2-vl"
    messages: list[Message]
    max_tokens: int = 512
    temperature: float = 0.0
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


# === VL Model Wrapper ===


def load_image(url: str) -> Image.Image:
    """Load image from URL or base64 data URL."""
    if url.startswith("data:"):
        # Base64 data URL
        header, data = url.split(",", 1)
        image_data = base64.b64decode(data)
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    else:
        # HTTP(S) URL
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")


class VLModelServer:
    def __init__(
        self,
        model_path: str,
        encoder_precision: str | None = None,
        decoder_precision: str | None = None,
    ):
        self.model_path = Path(model_path)
        self.encoder_precision = encoder_precision
        self.decoder_precision = decoder_precision
        self.processor = None
        self.tokenizer = None
        self.embed_tokens_sess = None
        self.embed_images_sess = None
        self.decoder_sess = None
        self.vision_format = None
        self.image_token_id = None

    def _get_onnx_path(self, name: str, precision: str | None) -> Path:
        """Get ONNX file path for given name and precision."""
        onnx_dir = self.model_path / "onnx"
        if precision:
            return onnx_dir / f"{name}_{precision}.onnx"
        return onnx_dir / f"{name}.onnx"

    def load(self):
        enc_label = self.encoder_precision or "fp32"
        dec_label = self.decoder_precision or "fp32"
        logger.info(f"Loading model from {self.model_path} (encoder={enc_label}, decoder={dec_label})...")

        # Load processor (includes tokenizer)
        self.processor = AutoProcessor.from_pretrained(str(self.model_path), trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.image_token_id = get_image_token_id(self.tokenizer)

        # Load embed_tokens (uses encoder precision for fp16, otherwise fp32)
        embed_tokens_prec = self.encoder_precision if self.encoder_precision == "fp16" else None
        self.embed_tokens_sess = load_onnx_session(
            self._get_onnx_path("embed_tokens", embed_tokens_prec)
        )

        # Force CPU for embed_images due to CUDA Expand op bug with dynamic shapes
        self.embed_images_sess = load_onnx_session(
            self._get_onnx_path("embed_images", self.encoder_precision),
            providers=["CPUExecutionProvider"],
        )

        self.decoder_sess = load_onnx_session(
            self._get_onnx_path("decoder", self.decoder_precision)
        )

        self.vision_format = detect_vision_format(self.embed_images_sess)
        logger.info(f"Model loaded. Vision format: {self.vision_format}")

    def _embed_tokens(self, input_ids: np.ndarray) -> np.ndarray:
        return self.embed_tokens_sess.run(None, {"input_ids": input_ids})[0]

    def _get_image_embeddings(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Get embeddings for a list of images."""
        return get_image_embeddings(
            self.embed_images_sess,
            images,
            vision_format=self.vision_format,
            processor=self.processor,
            do_pad_to_square=False,  # Already padded in generate()
            do_image_splitting=True,  # Match processor default for tiling
        )

    def _extract_images(self, messages: list[Message]) -> list[Image.Image]:
        """Extract and load images from messages."""
        images = []
        for msg in messages:
            if isinstance(msg.content, list):
                for part in msg.content:
                    if part.type == "image_url" and part.image_url:
                        try:
                            img = load_image(part.image_url.url)
                            images.append(img)
                        except Exception as e:
                            logger.warning(f"Failed to load image: {e}")
        return images

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert messages to format expected by processor."""
        result = []
        for msg in messages:
            if isinstance(msg.content, str):
                result.append({"role": msg.role, "content": msg.content})
            else:
                # Combine content parts
                parts = []
                for part in msg.content:
                    if part.type == "text" and part.text:
                        parts.append({"type": "text", "text": part.text})
                    elif part.type == "image_url":
                        parts.append({"type": "image"})
                result.append({"role": msg.role, "content": parts})
        return result

    def generate(
        self,
        messages: list[Message],
        max_tokens: int = 512,
        stream: bool = False,
    ):
        """Generate response, yielding tokens if streaming."""
        # Extract images
        images = self._extract_images(messages)
        converted_messages = self._convert_messages(messages)

        # Pad images to square for consistent tiling
        images = [pad_to_square(img) for img in images]

        # Tokenize using processor
        prompt = self.processor.apply_chat_template(
            converted_messages, tokenize=False, add_generation_prompt=True
        )
        # Use "pt" tensors when images present (processor requirement), else "np"
        return_tensors = "pt" if images else "np"
        inputs = self.processor(
            text=prompt,
            images=images if images else None,
            return_tensors=return_tensors,
        )
        if images:
            input_ids = inputs["input_ids"].numpy().astype(np.int64)
        else:
            input_ids = inputs["input_ids"].astype(np.int64)

        # Get embeddings
        if images:
            image_embeds_list = self._get_image_embeddings(images)
            text_embeds = self._embed_tokens(input_ids)[0]
            inputs_embeds = build_inputs_embeds(
                text_embeds, image_embeds_list, self.image_token_id, input_ids
            )
        else:
            inputs_embeds = self._embed_tokens(input_ids)

        # Initialize cache
        cache = initialize_cache(self.decoder_sess)
        output_infos = self.decoder_sess.get_outputs()

        seq_len = inputs_embeds.shape[1]
        attention_mask = np.ones((1, seq_len), dtype=np.int64)

        generated_tokens = []
        cur_len = seq_len

        for _ in range(max_tokens):
            if len(generated_tokens) == 0:
                embeds = inputs_embeds
            else:
                token_ids = np.array([[generated_tokens[-1]]], dtype=np.int64)
                embeds = self._embed_tokens(token_ids)

            attention_mask = np.ones((1, cur_len), dtype=np.int64)

            feed = {"inputs_embeds": embeds, "attention_mask": attention_mask}
            feed.update(cache)

            outputs = self.decoder_sess.run(None, feed)
            logits = outputs[0][0, -1]

            next_token = int(np.argmax(logits))
            generated_tokens.append(next_token)

            update_cache(cache, outputs, output_infos)
            cur_len += 1

            # Decode token
            token_str = self.tokenizer.decode([next_token])

            if stream:
                yield token_str

            if next_token == self.tokenizer.eos_token_id:
                break

        if not stream:
            yield self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


# === FastAPI App ===

app = FastAPI(title="LFM2-VL Server", version="1.0.0")
model: VLModelServer | None = None


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": "lfm2-vl", "object": "model", "owned_by": "liquidai"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    if request.stream:
        return StreamingResponse(
            stream_response(request, request_id, created),
            media_type="text/event-stream",
        )

    # Non-streaming response
    response_text = ""
    for token in model.generate(request.messages, max_tokens=request.max_tokens, stream=False):
        response_text = token  # Last yield is full text

    # Clean up end tokens
    response_text = response_text.replace("<|im_end|>", "").strip()

    return ChatCompletionResponse(
        id=request_id,
        created=created,
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )


async def stream_response(request: ChatCompletionRequest, request_id: str, created: int):
    """Generate SSE stream for chat completion."""
    for token in model.generate(request.messages, max_tokens=request.max_tokens, stream=True):
        if "<|im_end|>" in token or "<|endoftext|>" in token:
            continue

        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Send final chunk
    final_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


def main():
    global model

    parser = argparse.ArgumentParser(description="LFM2-VL OpenAI-compatible server")
    parser.add_argument("--model", required=True, help="Path to ONNX model directory")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--encoder",
        choices=["fp32", "fp16", "q4", "q8"],
        default="fp32",
        help="Encoder precision (default: fp32)",
    )
    parser.add_argument(
        "--decoder",
        choices=["fp32", "fp16", "q4", "q8"],
        default="fp32",
        help="Decoder precision (default: fp32)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    # Convert "fp32" to None (use default files without suffix)
    encoder_prec = None if args.encoder == "fp32" else args.encoder
    decoder_prec = None if args.decoder == "fp32" else args.decoder

    model = VLModelServer(args.model, encoder_precision=encoder_prec, decoder_precision=decoder_prec)
    model.load()

    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
