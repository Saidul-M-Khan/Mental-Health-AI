from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime


class ChatSessionCreate(BaseModel):
    """Schema for creating a new chat session"""
    title: Optional[str] = None


class ChatSessionResponse(BaseModel):
    """Schema for chat session response"""
    session_id: str
    user_email: Optional[str] = None  # Added user email field
    title: Optional[str] = None
    session_start: datetime


class ChatHistoryCreate(BaseModel):
    """Schema for creating a new chat history entry"""
    query_text: str
    session_id: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    """Schema for chat history response"""
    response_id: str
    query_text: str
    response_text: str
    created_at: datetime


class SessionWithHistory(BaseModel):
    """Schema for session with history response"""
    session_id: str
    data: List[Dict]  # Using Dict instead of ChatHistoryResponse for flexibility


class SessionGrouped(BaseModel):
    """Schema for grouped sessions response"""
    today: Optional[List[Dict]] = None  # Using Dict instead of ChatSessionResponse for flexibility
    yesterday: Optional[List[Dict]] = None
    last_week: Optional[List[Dict]] = None


class SymptomInput(BaseModel):
    """Schema for symptom analysis input"""
    clinical_text: str = Field(..., description="Patient's description of mental health concerns")


class UserBase(BaseModel):
    """Base user schema"""
    email: EmailStr


class UserCreate(UserBase):
    """Schema for creating a new user"""
    password: str


class UserResponse(UserBase):
    """Schema for user response"""
    created_at: datetime