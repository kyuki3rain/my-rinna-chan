import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft", torch_dtype=torch.float16)

if torch.cuda.is_available():
    model = model.to("cuda")

log = ""

def chat(sentence):
    global log

    # プロンプトの準備
    prompt_base = "ユーザー: {}<NL>システム: "
    prompt = prompt_base.format(sentence)
    log += prompt

    # 推論
    token_ids = tokenizer.encode(log, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            do_sample=True,
            max_new_tokens=256,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])
    log += output
    log += "<NL>"
    torch.cuda.empty_cache()

    return output.replace("</s>", "").replace("<NL>", "\n")

if __name__ == "__main__":
    while True:
        string = input("あなた: ")
        if string == "exit":
            break
    
        print("りんな: " + chat(string))