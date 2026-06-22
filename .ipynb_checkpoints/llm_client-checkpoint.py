from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
    conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # TODO: Define system prompt
    SYSTEM_PROMT = """
You are a helpful AI assistant who answers questions about NASA missions.
Use the provided context to answer the user's question.
If the context does not contain enough information, say so clearly.
Do not invent facts.
"""

    # TODO: Set context in messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMT},
        {"role": "system", "content": f"Context:\n{context}"}
    ]    
    
    # TODO: Add chat history
    for message in conversation_history:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant", "system"} and content:
            messages.append({"role": role, "content": content})
    # Add the last user message
    messages.append({"role": "user", "content": user_message})
    
    # TODO: Create OpenAI Client
    client = OpenAI(api_key = openai_key)
    
    # TODO: Send request to OpenAI
    response = client.chat.completions.create(model = model, messages = messages, temperature = 0.7)
    
    # TODO: Return response
    return response.choices[0].message.content