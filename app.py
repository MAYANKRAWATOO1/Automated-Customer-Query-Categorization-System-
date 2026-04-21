import gradio as gr
import pandas as pd
import os, sys

sys.path.insert(0, os.path.dirname(__file__))

from model import QueryClassifier
from faiss_db import FAISSDatabase
from chatbot import ChatbotEngine

# ------------------ LOAD ------------------

print("⏳ Loading dataset...")
df = pd.read_csv("dataset.csv")
df = df[df["Query"] != "Query"].reset_index(drop=True)

print("⏳ Training models...")
classifier = QueryClassifier()
classifier.train(df)

print("⏳ Building FAISS...")
vectors = classifier.get_all_vectors(df["Query"].tolist())
faiss_db = FAISSDatabase()
faiss_db.build(df, vectors)

chatbot = ChatbotEngine()
print("✅ System ready!\n")

# ------------------ RESPONSE ------------------

def respond(user_message, history):

    if history is None:
        history = []

    if not user_message.strip():
        return history, _empty_table()

    # ML prediction
    ml = classifier.predict(user_message)
    cat = ml["final_category"]
    vec = ml["feature_vector"]

    # FAISS search
    results = faiss_db.search(vec, k=1)
    context = results[0]["response"] if results else "Please contact support."

    # Chatbot response
    bot_reply, mode = chatbot.generate_response(user_message, cat, context)

    # ✅ FIXED MESSAGE FORMAT
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": bot_reply})

    # ML Panel
    table = f"""
<div style="color:white">
<b>Category:</b> {cat}<br>
<b>Mode:</b> {mode}<br>
<b>Context:</b> {context[:80]}...
</div>
"""

    return history, table


def _empty_table():
    return "Ask something to see ML results"


def clear_chat():
    return [], ""


# ------------------ UI ------------------

with gr.Blocks() as demo:

    gr.Markdown("## 🤖 Hybrid AI Customer Support")

    with gr.Row():

        # LEFT SIDE CHAT
        with gr.Column(scale=3):

            chatbot_ui = gr.Chatbot(

            )

            msg = gr.Textbox(placeholder="Type your query...")
            send = gr.Button("Send")
            clear = gr.Button("Clear")

        # RIGHT SIDE ML PANEL
        with gr.Column(scale=2):
            ml_output = gr.HTML(value=_empty_table())

    state = gr.State([])

    # SEND BUTTON
    send.click(
        respond,
        inputs=[msg, state],
        outputs=[chatbot_ui, ml_output]
    ).then(lambda: "", outputs=[msg])

    # ENTER PRESS
    msg.submit(
        respond,
        inputs=[msg, state],
        outputs=[chatbot_ui, ml_output]
    ).then(lambda: "", outputs=[msg])

    # CLEAR BUTTON
    clear.click(
        clear_chat,
        outputs=[chatbot_ui, ml_output]
    ).then(lambda: [], outputs=[state])

    # STATE SYNC
    chatbot_ui.change(lambda x: x, chatbot_ui, state)

# ------------------ RUN ------------------

if __name__ == "__main__":
    demo.launch(share=True)
