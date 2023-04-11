import os
import gradio as gr

from chatcaptioner.video_chat import set_openai_key, summarize_chat, AskQuestions
from chatcaptioner.blip2 import Blip2


openai_key = os.environ["OPENAI_API_KEY"]
set_openai_key(openai_key)


blip2 = Blip2('FlanT5 XXL', device_id=0, bit8=True)
chat = AskQuestions(None, blip2, 'gpt-3.5-turbo', n_blip2_context=1)


def gradio_reset(gr_img):
    chat.reset(gr_img)
    return None


def gradio_ask(chatbot):
    question = chat.ask_question()
    question = chat.question_trim(question)
    chat.questions.append(question)
    chatbot = chatbot + [[None, question]]
    return chatbot


def gradio_answer(chatbot):
    answer = chat.answer_question()
    answer = chat.answer_trim(answer)
    chat.answers.append(answer)
    chatbot = chatbot + [[answer, None]]
    return chatbot


def gradio_summarize(chatbot):
    summary, summary_prompt, n_token_sum = summarize_chat(chat.questions, chat.answers, model='gpt-3.5-turbo')
    chatbot = chatbot + [[None, 'Final Caption: ' + summary]]
    return chatbot


with gr.Blocks() as demo:
    gr.Markdown("## ChatCaptioner Demo")
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil")
            start = gr.Button("Start Chat")
        chatbot = gr.Chatbot()

    start.click(gradio_reset, image, chatbot) \
        .then(gradio_ask, chatbot, chatbot) \
        .then(gradio_answer, chatbot, chatbot) \
        .then(gradio_ask, chatbot, chatbot) \
        .then(gradio_answer, chatbot, chatbot) \
        .then(gradio_ask, chatbot, chatbot) \
        .then(gradio_answer, chatbot, chatbot) \
        .then(gradio_ask, chatbot, chatbot) \
        .then(gradio_answer, chatbot, chatbot) \
        .then(gradio_ask, chatbot, chatbot) \
        .then(gradio_answer, chatbot, chatbot) \
        .then(gradio_ask, chatbot, chatbot) \
        .then(gradio_answer, chatbot, chatbot) \
        .then(gradio_ask, chatbot, chatbot) \
        .then(gradio_answer, chatbot, chatbot) \
        .then(gradio_ask, chatbot, chatbot) \
        .then(gradio_answer, chatbot, chatbot) \
        .then(gradio_ask, chatbot, chatbot) \
        .then(gradio_answer, chatbot, chatbot) \
        .then(gradio_ask, chatbot, chatbot) \
        .then(gradio_answer, chatbot, chatbot) \
        .then(gradio_summarize, chatbot, chatbot)

demo.launch()