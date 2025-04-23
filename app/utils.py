from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class UserBase(BaseModel):
    email: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int
    
    class Config:
        from_attributes = True


class ChatSessionBase(BaseModel):
    title: Optional[str] = None


class ChatSessionCreate(ChatSessionBase):
    pass


class ChatSession(ChatSessionBase):
    session_id: str
    session_start: datetime
    
    class Config:
        from_attributes = True


class ChatHistoryBase(BaseModel):
    query_text: str


class ChatHistoryCreate(ChatHistoryBase):
    pass


class ChatHistory(ChatHistoryBase):
    response_id: str
    response_text: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class SessionWithHistory(BaseModel):
    session_id: str
    data: List[ChatHistory]


class SessionGrouped(BaseModel):
    today: Optional[List[ChatSession]] = None
    yesterday: Optional[List[ChatSession]] = None
    last_week: Optional[List[ChatSession]] = None


class SymptomInput(BaseModel):
    clinical_text: str = Field(..., description="Patient's description of symptoms or concerns")
    use_gpt: bool = Field(True, description="Whether to use GPT-4 for detailed analysis")
    similarity_threshold: Optional[float] = Field(0.5, description="Threshold for similarity matching")