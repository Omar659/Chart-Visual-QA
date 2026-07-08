from transformers import Blip2ForConditionalGeneration, Blip2Processor

from chartqa.models.qwen_vl_chat import build_quantization_config


class Blip2Chat:
    """Wrapper around BLIP-2 for single-image visual question answering.

    Mirrors the QwenVLChat interface (`chat(image, text, system_prompt,
    max_new_tokens)`) so the evaluation loop can swap models with only an import
    change. Key differences from Qwen3-VL:
      * BLIP-2 has no chat template and no system role -> any instruction is
        folded into a single text prompt (see build_prompt).
      * The vision encoder runs at a fixed low resolution (224x224), so dense
        charts read poorly; expect lower accuracy than Qwen3-VL on ChartQA.
      * `generate` already returns only the answer tokens, so there is no prompt
        prefix to trim before decoding.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-flan-t5-xl",
        dtype: str = "auto",
        device_map: str = "auto",
        adapter_path: str | None = None,
        quantization: str | None = None,
        cpu_offload: bool = False,
    ):
        # Opt-in 4-bit/8-bit loading; None/"none" keeps full precision. cpu_offload
        # lets a model that doesn't fit in VRAM spill some layers to CPU (slow).
        quantization_config = build_quantization_config(quantization, cpu_offload)
        quant_kwargs = (
            {"quantization_config": quantization_config} if quantization_config else {}
        )
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
            **quant_kwargs,
        )
        # Optionally load a LoRA adapter checkpoint on top of the base model.
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            if quantization_config is None:
                # Fold the LoRA weights into the base model for faster inference.
                self.model = self.model.merge_and_unload()
            # With a quantized base the adapter must stay attached: merge_and_unload()
            # cannot fold LoRA deltas into 4-/8-bit weights, so inference runs
            # through the PeftModel wrapper (slightly slower, memory-cheap).
        # Prefer the checkpoint's processor (in case it added tokens) and fall
        # back to the base model's when running without an adapter.
        self.processor = Blip2Processor.from_pretrained(adapter_path or model_name)

    @staticmethod
    def build_prompt(text: str, system_prompt: str | None = None) -> str:
        """Build BLIP-2's single text prompt.

        BLIP-2 was trained on a VQA-style "Question: ... Answer:" framing and has
        no system role, so any instruction is simply prepended to the same
        string (Qwen put it in a dedicated system turn instead).
        """
        prompt = f"Question: {text} Answer:"
        if system_prompt:
            prompt = f"{system_prompt}\n{prompt}"
        return prompt

    def chat(
        self,
        image,
        text: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 128,
    ) -> str:
        """Send an image + text and return the reply.

        `image` is a PIL image (converted to RGB for the vision encoder).
        """
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        prompt = self.build_prompt(text, system_prompt)

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        # BLIP-2 returns only the generated answer; decode it directly (no trim).
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return output_text[0].strip()
