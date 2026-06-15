import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "openbmb/MiniCPM-V-4.6"

processor = AutoProcessor.from_pretrained(model_id)
# model = AutoModelForImageTextToText.from_pretrained(
#     model_id, torch_dtype="auto", device_map="auto"
# )

# Flash Attention 2 is recommended for better acceleration and memory saving,
# especially in multi-image and video scenarios.
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# ===== Image Inference ===== #
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "url": "https://huggingface.co/datasets/openbmb/DemoCase/resolve/main/refract.png"},
#             {"type": "text", "text": "What causes this phenomenon?"},
#         ],
#     }
# ]

# downsample_mode = "16x"  # Using `downsample_mode="4x"` for Finer Detail

# inputs = processor.apply_chat_template(
#     messages, tokenize=True, add_generation_prompt=True,
#     return_dict=True, return_tensors="pt",
#     downsample_mode=downsample_mode,
#     max_slice_nums=36,
# ).to(model.device)

# generated_ids = model.generate(**inputs, downsample_mode=downsample_mode, max_new_tokens=512)
# generated_ids_trimmed = [
#     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text[0])
# ===== END OF Image Inference ===== #


# ===== Video Inference ===== #
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "url": "https://huggingface.co/datasets/openbmb/DemoCase/resolve/main/football.mp4"},
            {"type": "text", "text": "Describe this video in detail. Follow the timeline and focus on on-screen text, interface changes, main actions, and scene changes."},
        ],
    }
]

downsample_mode = "16x"  # Using `downsample_mode="4x"` for Finer Detail

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
    downsample_mode=downsample_mode,
    max_num_frames=128,
    stack_frames=1,
    max_slice_nums=1,
    use_image_id=False,
).to(model.device)

generated_ids = model.generate(**inputs, downsample_mode=downsample_mode, max_new_tokens=2048)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])

# ===== END OF Video Inference ===== #