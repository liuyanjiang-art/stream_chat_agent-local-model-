import os
import uuid
import gradio as gr
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import asyncio

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2:7b")
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
DEFAULT_SESSION_ID = os.getenv("DEFAULT_SESSION_ID", "session_001")

# LangChain 独立上下文：只存消息对象，给 LLM 使用
session_store = {}

def get_langchain_history(sid: str):
    if sid not in session_store:
        session_store[sid] = [
            SystemMessage(content="你是友好的AI助手，记住本次对话上下文。")
        ]
    return session_store[sid]

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.7
)

async def chat_handler(user_input, gradio_history, session_id):
    if not user_input.strip():
        yield gradio_history
        return

    # 1. LangChain 侧追加用户消息
    lc_history = get_langchain_history(session_id)
    lc_history.append(HumanMessage(content=user_input))

    # 2. Gradio 侧追加用户消息（字典格式，匹配校验要求）
    gradio_history.append({"role": "user", "content": user_input})
    gradio_history.append({"role": "assistant", "content": ""})
    full_reply = ""

    # 3. 流式调用 LLM
    async for chunk in llm.astream(lc_history):
        if chunk.content:
            full_reply += chunk.content
            gradio_history[-1]["content"] = full_reply
            yield gradio_history
        await asyncio.sleep(0.1)

    # 4. LangChain 侧保存 AI 回复
    lc_history.append(AIMessage(content=full_reply))

def clear_session(session_id):
    if session_id in session_store:
        del session_store[session_id]
    return [], session_id

def new_session():
    new_sid = f"session_{uuid.uuid4().hex[:6]}"
    return [], new_sid
def trans(sid_box):
    result=[]
    for item in session_store[sid_box]:
        if isinstance(item,HumanMessage):
            dict1={"role":"user","content":item.content}
            result.append(dict1)
        if isinstance(item,AIMessage):
            dict2={"role":"assistant","content":item.content}
            result.append(dict2)
    return result
def load(sid):
    if sid in session_store:
        res = trans(sid)
        gr.Info(f"加载会话：{sid}")
    else:
        res = []
        gr.Info(f"会话 {sid} 不存在")
    return res, sid


    
        
    

    

# UI：Chatbot 不加 type 参数，直接传字典格式
with gr.Blocks() as demo:
    gr.Markdown(f'<h1 style="text-align:center">主人，我是你的{OLLAMA_MODEL}助手</h1>')
    with gr.Row():
        sid_box = gr.Textbox(label="会话ID", value=DEFAULT_SESSION_ID, scale=3)
        btn_new = gr.Button("新建会话", scale=1)
        btn_clear = gr.Button("清空当前会话", variant="stop", scale=1)
        btn_load=gr.Button("查询对话",variant="secondary",scale=1)
        

    chatbot = gr.Chatbot(height=520)
    text_input = gr.Textbox(label="输入消息", placeholder="请输入你的问题~", lines=1)
    send_btn = gr.Button("发送", variant="primary")

    send_btn.click(chat_handler, inputs=[text_input, chatbot, sid_box], outputs=chatbot,concurrency_limit=2)
    text_input.submit(chat_handler, inputs=[text_input, chatbot, sid_box], outputs=chatbot,concurrency_limit=2)
    btn_clear.click(clear_session, inputs=[sid_box], outputs=[chatbot, sid_box])
    btn_new.click(new_session, outputs=[chatbot, sid_box])
    btn_load.click(
        load,
        inputs=[sid_box],
        outputs=[chatbot, sid_box]
    )

    # ========== 新增：下拉框选中自动加载会话 ==========
    def load_from_dropdown(selected_sid):
        # selected_sid 是下拉选中的id
        if not selected_sid:
            return [], selected_sid
        res = trans(selected_sid)
        gr.Info(f"下拉加载会话：{selected_sid}")
        return res, selected_sid


if __name__ == "__main__":
    #启动排队请求
    demo.queue(max_size=20)
    #拉取web服务
    demo.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT,
        share=False
    )
