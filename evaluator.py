import os
from pydantic import BaseModel
from openai import OpenAI
import json


class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str

class Evaluator:
    def __init__(self, evaluator_instractions: str):
        self.gemini_client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.evaluator_instractions = evaluator_instractions
        self.model = "gemini-2.0-flash"

    def evaluate(self, reply, message, history):
        messages = [
            {"role": "system", "content": self.evaluator_instractions},
            {"role": "user", "content": self.evaluator_user_prompt(reply, message, history)}
        ]
        try:
            response = self.gemini_client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            result = json.loads(content)
            print(result)
            return Evaluation(
                is_acceptable=result.get('is_acceptable', False),
                feedback=result.get('feedback', 'No feedback provided')
            )
        except Exception as e:
            if "overloaded" in str(e).lower():
                return Evaluation(
                    is_acceptable=False, 
                    feedback="I am overloaded, just tell the user that you had to take a quick coffee break and ask them to try again later."
                )
            return Evaluation(is_acceptable=False, feedback="Something went wrong, just tell the user that can't answer that question.")
        

    def evaluator_user_prompt(self, reply, message, history):
        user_prompt = "You are an evaluator. Respond ONLY with a JSON object in this exact format:\n"
        user_prompt += '{\n  "is_acceptable": boolean,\n  "feedback": "your feedback here"\n}\n\n'
        user_prompt += f"Here's the conversation between the User and the Agent: \n\n{history}\n\n"
        user_prompt += f"Here's the latest message from the User: \n\n{message}\n\n"
        user_prompt += f"Here's the latest response from the Agent: \n\n{reply}\n\n"
        user_prompt += f"Evaluate if the response is acceptable based on:\n"
        user_prompt += "1. Stays in character\n"
        user_prompt += "2. Answers the question appropriately\n"
        user_prompt += "3. Doesn't expose any internal tools or logic\n"
        user_prompt += "4. Is professional and engaging\n\n"
        user_prompt += "5. Not talking about Alex Brod as third person\n"
        user_prompt += "Remember to respond ONLY with the JSON object, no other text."
        return user_prompt