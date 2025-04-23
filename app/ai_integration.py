import os
from typing import Dict, List, Optional
from fastapi import HTTPException
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# OpenAI Client Setup
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class MentalHealthAI:
    """Handler for mental health AI model integration"""
    @staticmethod
    def generate_title(question: str) -> Optional[str]:
        """Generate a title for a chat session based on the first question"""
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": """Based on the mental health query below, generate a concise, 
                        empathetic title that does not exceed 30 characters. The title should be 
                        descriptive but maintain privacy, avoiding overly specific details. Focus on 
                        the general theme of the conversation rather than specific symptoms."""
                    },
                    {
                        "role": "user", 
                        "content": question
                    }
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating title: {e}")
            # Provide a default title if generation fails
            return "Mental Health Conversation"

    @staticmethod
    def get_conversation_chain():
        """Create a mental health focused conversational chain using the OpenAI API"""
        def process_query(question: str, history: List[Dict] = None):
            try:
                messages = []
                
                # System message to guide the AI as a mental health assistant
                messages.append({
                    "role": "system",
                    "content": """You are an empathetic mental health assistant. Your role is to listen carefully, 
                    understand the person's concerns, and provide supportive, helpful responses. Respond in a warm, 
                    conversational manner while maintaining professionalism. Remember to:
                    
                    1. Show empathy and understanding
                    2. Ask clarifying questions when appropriate
                    3. Provide evidence-based information and suggestions
                    4. Emphasize that you're an AI and not a replacement for professional mental health care
                    5. Encourage seeking professional help when needed
                    
                    Always prioritize the person's wellbeing and safety in your responses.
                    """
                })
                
                # Add conversation history if available
                if history:
                    for item in history:
                        messages.append({"role": "user", "content": item["query_text"]})
                        messages.append({"role": "assistant", "content": item["response_text"]})
                
                # Add the current question
                messages.append({"role": "user", "content": question})
                
                # Get response from OpenAI
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages
                )
                
                return {"answer": response.choices[0].message.content}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
        
        return {"invoke": process_query}

    @staticmethod
    def analyze_mental_health(clinical_text: str):
        """Analyze mental health concerns and provide recommendations"""
        try:
            system_prompt = """
            You are a compassionate mental health assistant. Your role is to:

            1. Carefully analyze the person's described mental health concerns
            2. Identify potential mental health issues they might be experiencing
            3. Provide thoughtful, evidence-based recommendations
            4. Suggest coping strategies and self-care practices
            5. Encourage professional help when appropriate

            Structure your response in a warm, conversational manner that includes:

            1. A gentle acknowledgment of their feelings and concerns
            2. A thoughtful analysis of what might be happening (avoiding definitive diagnosis)
            3. Practical, actionable suggestions that could help
            4. Encouragement to seek professional support if needed
            5. A compassionate closing note

            Remember that this is preliminary information based on limited text, not a clinical diagnosis. 
            Be supportive, empathetic, and helpful while maintaining appropriate boundaries.
            """
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"I've been experiencing these mental health concerns: '{clinical_text}'"}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            return {
                "analysis": response.choices[0].message.content.strip()
            }
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error analyzing mental health concerns: {str(e)}")