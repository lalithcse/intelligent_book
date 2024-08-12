from pydantic import BaseModel
from typing import List, Optional


class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserUpdate(BaseModel):
    email: Optional[str] = None
    password: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    username: str
    email: str


class BookCreate(BaseModel):
    title: str
    author: str
    genre: str
    year_published: int
    summary: str


class BookUpdate(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    genre: Optional[str] = None
    year_published: Optional[int] = None
    summary: Optional[str] = None


class BookResponse(BaseModel):
    id: int
    title: str
    author: str
    genre: str
    year_published: int
    summary: Optional[str] = None


class ReviewCreate(BaseModel):
    book_id: int
    user_id: int
    review_text: str
    rating: float


class BookOut(BookCreate):
    id: int
    reviews: List[ReviewCreate] = []

    class Config:
        from_attributes = True


class ReviewOut(ReviewCreate):
    id: int

    class Config:
        from_attributes = True


class ReviewResponse(BaseModel):
    id: int
    book_id: int
    user_id: int
    review_text: str
    rating: int
