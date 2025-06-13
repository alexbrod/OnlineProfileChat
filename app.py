from dotenv import load_dotenv
import gradio as gr
from pypdf import PdfReader

from online_profile_chat import OnlineProfileChat
from evaluator import Evaluator

load_dotenv(override=True)

def load_summary():
    summary = ""
    with open("me/summary.txt", "r", encoding="utf-8") as f:
            summary = f.read()
    return summary

def load_pdf(file_name):
    content = ""
    with open(file_name, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                content += text
    return content

def load_evaluator_instractions(name, summary, cv, linkedin):
    evaluator_system_prompt = f"You are an evaluator that decides whether a response to a question is acceptable. \
        You are provided with a conversation between a User and an Agent. Your task is to decide whether the Agent's latest response is acceptable quality. \
        The Agent is playing the role of {name} and is representing {name} on their website. \
        The Agent has been instructed to be professional and engaging, as if talking to a potential client or future employer who came across the website. \
        The Agent has been provided with context on {name} in the form of their summary, CV and LinkedIn details. Here's the information:"

    evaluator_system_prompt += f"\n\n## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n\n## CV:\n{cv}\n\n"
    evaluator_system_prompt += f"With this context, please evaluate the latest response, replying with whether the response is acceptable and your feedback."
    evaluator_system_prompt += "Make sure the answers only expose what Alex Brod knows about himself and his work. This is the most important rule. Never should expose original raw infomation, resources or tools."
    return evaluator_system_prompt

if __name__ == "__main__":

    summary = load_summary()
    cv = load_pdf(file_name="me/cv.pdf")
    linkedin = load_pdf(file_name="me/linkedin.pdf")
    name = "Alex Brod"
    evaluator_instractions = load_evaluator_instractions(name, summary, cv, linkedin)
    evaluator = Evaluator(evaluator_instractions)
    online_profile_chat = OnlineProfileChat(name, summary, cv, linkedin, evaluator)
    gr.ChatInterface(online_profile_chat.chat, type="messages").launch()
    