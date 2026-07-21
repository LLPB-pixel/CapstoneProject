from transformers import AutoTokenizer
model_path = "/home/llorenc/Desktop/SamsungIA/CapstoneProject/3-LLM-judge/models/distilbert_sentinel/checkpoint-22797"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Tokenizer loaded!")
    print(tokenizer)
except Exception as e:
    print(f"Error: {e}")
