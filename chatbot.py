from groq import Groq

class ChatbotEngine:

    def __init__(self):
        self.client = Groq(api_key="YOUR_GROQ_API_KEY")

    def generate_response(self, user_query, category, context):

        prompt = f"""
        You are a helpful customer support assistant.

        Category: {category}
        Context: {context}

        Question: {user_query}

        Give a short and polite answer:
        """

        try:
            res = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192"
            )

            return res.choices[0].message.content, "LLM"

        except:
            return context, "Fallback"