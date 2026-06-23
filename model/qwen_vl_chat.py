from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset


class QwenVLChat:
    """Wrapper around Qwen3-VL for single-image visual chat."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        dtype: str = "auto",
        device_map: str = "auto",
        attn_implementation: str = "sdpa",
    ):
        # Load the model on the available device(s) (device_map="auto" uses the GPU).
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

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

        # Preparation for inference
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference: generation of the output
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


if __name__ == "__main__":
    model = QwenVLChat()
    counter_correct = 0
    counter_total = 0
    for i in load_dataset("HuggingFaceM4/ChartQA")["test"]:
        output = model.chat(
            image=i['image'],
            text=i['query'] + " Please answer directly.",
        )
        
        if output == i['label'][0]:
            counter_correct += 1
        counter_total += 1

        print(f"Question: {i['query']}")
        print(f"Answer: {i['label'][0]}")
        print(f"Output: {output}")
        print(f"Correct: {counter_correct}/{counter_total}")
        print("\n\n")
    print(f"Accuracy: {counter_correct / counter_total * 100:.2f}%")