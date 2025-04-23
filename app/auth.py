from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import os
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr

# Load environment variables
load_dotenv()

# Security configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key-keep-it-secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

# JWT token handling
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserInDB(BaseModel):
    email: EmailStr
    password: str  # No longer hashed
    created_at: datetime

# Helper functions
def get_user(db, email: str):
    """Get user from database"""
    # Use your MongoDB collection
    user = db.users.find_one({"email": email})
    return user

def authenticate_user(db, email: str, password: str):
    """Authenticate user with plaintext password"""
    user = get_user(db, email)
    if not user:
        return False
    if password != user["password"]:  # Simple plaintext comparison
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db=None):
    """Get user from token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    
    # Get database if not provided
    if db is None:
        from .database import get_db
        db = get_db()  # Fixed: now directly using the db object, not treating it as an iterator
        
    user = get_user(db, email=token_data.email)
    if user is None:
        raise credentials_exception
    return user

# User registration and management
def register_user(db, user: UserCreate):
    """Register a new user with plaintext password"""
    # Check if user exists
    existing_user = get_user(db, user.email)
    if existing_user:
        return False
    
    # Create new user with plaintext password
    user_data = {
        "email": user.email,
        "password": user.password,  # Store password as plaintext
        "created_at": datetime.now()
    }
    
    # Insert user into database
    db.users.insert_one(user_data)
    return True

def get_user_email_from_token(token: str):
    """Extract email from JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        return email
    except JWTError:
        return None