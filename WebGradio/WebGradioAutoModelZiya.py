
import os
import platform
import gradio as gr
import torch
from loguru import logger
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM

if platform.system() == 'Windows':
    os.environ['PATH'] = os.environ.get("PATH", "") + os.pathsep + r'D:\devtools\PythonVenv\llama2\Lib\site-packages\torch\lib'

MODEL_PATH = '/opt/Data/ModelWeight/IDEA-CCNL/Ziya-LLaMA-13B'

device = torch.device("cuda")

model = LlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="cuda:0", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

def tripStr(text):
    assist = "<bot>  :"
    if assist in text:
        pos = text.index(assist) + len(assist)
        data = text[pos:len(text)]
        data=data.replace("</s>","")
        return data

    return text

def predict(chatbot, history):
    input = chatbot[-1][0]
    logger.debug(f"input->:{input}")    
    history += [{"role":"user", "content":input}]
    prompt = '<human>:' + input.strip() + '\n<bot>:'
        
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generate_ids = model.generate(
                input_ids,
                max_new_tokens=1024, 
                do_sample = True, 
                top_p = 0.85, 
                temperature = 1.0, 
                repetition_penalty=1., 
                eos_token_id=2, 
                bos_token_id=1, 
                pad_token_id=0)

    output = tokenizer.batch_decode(generate_ids)[0]

    response = tripStr(output)
    logger.debug(response)
    history += [{"role":"assistant", "content":response }]
    
    chatbot[-1][1] = response
    logger.debug(f"history->:{history}")

    return chatbot, history

def reset_user_input():
    return gr.update(value='')

def reset_state():
    return [], [], None

def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)

with gr.Blocks(title = "Llama2", css="footer {visibility: hidden}") as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        height=800,
        show_copy_button = False,
        layout= "bubble",
        avatar_images=("./image/Einstein.jpg", "./image/openai.png")
    )

    with gr.Row():
        with gr.Column(scale=9):
            user_input = gr.Textbox(show_label=False, placeholder="请输入您的问题,刷新页面可清除历史", lines=1, container=False)

        with gr.Column(min_width=1, scale=1):
            submitBtn = gr.Button("提交", variant="primary")

    begin = [{"role":"system", "content":"你是智能客服小蓝，仔细分析用户的输入并作详细又准确的回答。"}]
    history = gr.State(value=begin)

    subMsg = submitBtn.click(fn=add_text, inputs=[chatbot, user_input], outputs=[chatbot, user_input], queue=False).then(fn=predict, inputs=[chatbot, history], outputs=[chatbot, history], show_progress=True, concurrency_limit=1)
    inputMsg = user_input.submit(fn=add_text, inputs=[chatbot, user_input], outputs=[chatbot, user_input], queue=False).then(fn=predict, inputs=[chatbot, history], outputs=[chatbot, history], show_progress=True, concurrency_limit=1)

    subMsg.then(fn=lambda: gr.Textbox(interactive=True), inputs=None, outputs=[user_input], queue=False)
    inputMsg.then(fn=lambda: gr.Textbox(interactive=True), inputs=None, outputs=[user_input], queue=False)

auth=[("llm","llm123456"),("zhangsan","zhangsan123")]

demo.queue().launch(server_name="192.168.2.200", server_port=8000, inbrowser=False, share=False, auth=auth)