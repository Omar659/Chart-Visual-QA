"""Qwen3-VL chat wrapper (vendored from the model team's `model/qwen_vl_chat.py`).

Kept self-contained inside `backend/` so the Docker image needs no cross-package
imports. The training/eval `__main__` demo and the `datasets` dependency were
dropped — only the inference path is needed here.
"""

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


class QwenVLChat:
    """Wrapper around Qwen3-VL for single-image visual chat."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        dtype: str = "auto",
        device_map: str = "auto",
        attn_implementation: str = "sdpa",
        adapter_path: str | None = None,
    ):
        # Load the model on the available device(s) (device_map="auto" uses the GPU).
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        # Optionally load a LoRA adapter checkpoint on top of the base model.
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            # Fold the LoRA weights into the base model for faster inference.
            self.model = self.model.merge_and_unload()
        # Prefer the checkpoint's processor (in case it added tokens) and fall
        # back to the base model's when running without an adapter.
        self.processor = AutoProcessor.from_pretrained(adapter_path or model_name)

    @staticmethod
    def build_messages(
        image,
        text: str,
        system_prompt: str | None = None,
        answer: str | None = None,
    ) -> list:
        """Build the chat `messages` structure for the Qwen processor."""
        messages = []
        if system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        )
        if answer is not None:
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            )
        return messages

    def chat(
        self,
        image,
        text: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 128,
    ) -> str:
        """Send an image + text (with optional system prompt) and return the reply.

        `image` accepts a URL, a local file path, or a PIL image.
        """
        messages = self.build_messages(image, text, system_prompt)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]
