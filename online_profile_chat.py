import json
import os
from openai import OpenAI

from tools import TOOLS, record_user_details, record_unknown_question
from evaluator import Evaluator

class OnlineProfileChat:
    MAX_RETRIES = 2
    def __init__(self, name, summary, cv, linkedin, evaluator: Evaluator):
        self.gemini_client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.name = name
        self.summary = summary
        self.cv = cv
        self.linkedin = linkedin
        self.model = "gemini-2.0-flash"
        self.evaluator = evaluator
        self.tools = {
            "record_user_details": record_user_details,
            "record_unknown_question": record_unknown_question
        }
    
    def chat(self, message, history):
        retries = 0
        is_acceptable = False
        messages = [{"role": "system", "content": self._system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            try:
                response = self.gemini_client.chat.completions.create(model=self.model, messages=messages, tools=TOOLS)
                if response.choices[0].finish_reason=="tool_calls":
                    message = response.choices[0].message
                    tool_calls = message.tool_calls
                    results = self._handle_tool_call(tool_calls)
                    messages.append(message)
                    messages.extend(results)
                else:
                    done = True
            except Exception as e:
                if "overloaded" in str(e).lower():
                    return "I had to take a quick coffee break ☕. Could you please try asking your question again in a moment?"
                return "I'm sorry, I'm having trouble processing your request right now. Could you try again?"

        while not is_acceptable and retries < OnlineProfileChat.MAX_RETRIES:
            evaluation = self.evaluator.evaluate(response.choices[0].message.content, message, history)
            if not evaluation.is_acceptable:
                response = self._rerun(response.choices[0].message.content, message, history, evaluation.feedback)
                evaluation = self.evaluator.evaluate(response.choices[0].message.content, message, history)
                retries += 1
            is_acceptable = evaluation.is_acceptable
        if retries == OnlineProfileChat.MAX_RETRIES:
            response.choices[0].message.content = "I'm sorry, I'm not able to answer that question. Try asking something else."

        return response.choices[0].message.content
    
    
    def _rerun(self, reply, message, history, feedback):
        updated_system_prompt = self._system_prompt() + f"\n\n## Previous answer rejected\nYou just tried to reply, but the quality control rejected your reply\n"
        updated_system_prompt += f"## Your attempted answer:\n{reply}\n\n"
        updated_system_prompt += f"## Reason for rejection:\n{feedback}\n\n"
        messages = [{"role": "system", "content": updated_system_prompt}] + history + [{"role": "user", "content": message}]
        response = self.gemini_client.chat.completions.create(model=self.model, messages=messages)
        return response

    def _handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = self.tools.get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def _system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
            particularly questions related to {self.name}'s career, background, skills and experience. \
            Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
            You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
            Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
            If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
            If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. \
            Make the conversation natural, make longer answers when asked or appropriate, for example, you do not need to describe the whole biography of {self.name}, \
            when the user greets you. When adked of questions that are completely irrelevant to {self.name}'s career, you can explain that you are here to discuss jsut career topics.\
            Do not expose any tools you are using or internal logic."

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## CV:\n{self.cv}\n\n## LinkedIn:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt