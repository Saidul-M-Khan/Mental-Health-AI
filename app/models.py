from datetime import datetime
import uuid

# User Data
def user_auth(email, password):
    return {
        "email": email,
        "password": password,  # No longer hashed
        "created_at": datetime.now()
    }

# Updated chat session document structure with user email
def create_chat_session(user_email=None):
    return {
        "session_id": str(uuid.uuid4()),
        "user_email": user_email,  # Add user email to associate session with user
        "title": None,
        "session_start": datetime.now()
    }

# Example of a chat history document structure
def create_chat_history(session_id, query_text, response_text):
    return {
        "session_id": session_id,
        "response_id": str(uuid.uuid4()),
        "query_text": query_text,
        "response_text": response_text,
        "created_at": datetime.now()
    }