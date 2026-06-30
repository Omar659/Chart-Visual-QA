import torch

from chartqa.constants import ANSWER_SUFFIX
from chartqa.models.blip2_chat import Blip2Chat


class ChartQACollator:
    """Class-based collator that turns raw samples into Qwen3-VL training batches.

    Fixes the alignment bug by computing prompt_len through the multimodal
    pipeline, so the expanded image tokens are counted correctly.
    """

    def __init__(self, processor, system_prompt=None, answer_suffix=ANSWER_SUFFIX):
        self.processor = processor
        self.system_prompt = system_prompt
        self.answer_suffix = answer_suffix

        # Set the pad_token if the model's processor does not define one.
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def __call__(self, batch):
        pad_id = self.processor.tokenizer.pad_token_id
        input_ids_list, labels_list, attn_list, mm_type_list = [], [], [], []
        pixel_values_list, grid_thw_list = [], []

        for ex in batch:
            # 1. Messages for the whole exchange (prompt + answer).
            full_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": ex["image"]},
                        {"type": "text", "text": ex["question"] + self.answer_suffix}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ex["answer"]}]
                }
            ]
            if self.system_prompt:
                full_messages.insert(0, {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}]
                })

            # 2. Messages for the prompt only (to count exactly the tokens to mask).
            prompt_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": ex["image"]},
                        {"type": "text", "text": ex["question"] + self.answer_suffix}
                    ]
                }
            ]
            if self.system_prompt:
                prompt_messages.insert(0, {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}]
                })

            # Apply the full multimodal template to both message sets.
            full_inputs = self.processor.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt"
            )

            prompt_inputs = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )

            prompt_len = prompt_inputs["input_ids"].shape[1]

            ids = full_inputs["input_ids"][0]
            labels = ids.clone()

            # Mask the entire prompt (including the expanded image tokens) with -100.
            labels[:prompt_len] = -100

            input_ids_list.append(ids)
            labels_list.append(labels)
            attn_list.append(full_inputs["attention_mask"][0])

            # Required by Qwen3-VL's M-RoPE mechanism.
            if "mm_token_type_ids" in full_inputs:
                mm_type_list.append(full_inputs["mm_token_type_ids"][0])

            pixel_values_list.append(full_inputs["pixel_values"])
            grid_thw_list.append(full_inputs["image_grid_thw"])

        # 3. Right-pad all text tensors.
        max_len = max(x.size(0) for x in input_ids_list)

        def pad(seq, value):
            out = torch.full((max_len,), value, dtype=seq.dtype)
            out[:seq.size(0)] = seq
            return out

        collated = {
            "input_ids": torch.stack([pad(x, pad_id) for x in input_ids_list]),
            "attention_mask": torch.stack([pad(x, 0) for x in attn_list]),
            "labels": torch.stack([pad(x, -100) for x in labels_list]),
            # Qwen concatenates the image pixels along the zero dimension.
            "pixel_values": torch.cat(pixel_values_list, dim=0),
            "image_grid_thw": torch.cat(grid_thw_list, dim=0),
        }

        if mm_type_list:
            collated["mm_token_type_ids"] = torch.stack([pad(x, 0) for x in mm_type_list])

        return collated


class Blip2Collator:
    """Collate ChartQA samples into a BLIP-2 (Flan-T5) seq2seq training batch.

    Unlike Qwen3-VL's chat-template collator, BLIP-2 takes a plain
    `(images, text)` pair: the visual query tokens are injected by the Q-Former
    inside `forward`, so the text `input_ids` only hold the prompt. For Flan-T5
    the answer is tokenized separately as the decoder `labels`.
    """

    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def __call__(self, batch):
        images = [ex["image"].convert("RGB") for ex in batch]
        prompts = [Blip2Chat.build_prompt(ex["question"]) for ex in batch]
        answers = [ex["answer"] for ex in batch]

        # Encoder side: prompt text (right-padded) + pixel values.
        inputs = self.processor(
            images=images, text=prompts, return_tensors="pt", padding=True
        )
        # Decoder side: the answer tokens. Padding is masked out of the loss
        # with -100 so only real answer tokens are supervised.
        labels = self.tokenizer(
            text_target=answers, return_tensors="pt", padding=True
        ).input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs
