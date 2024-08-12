import asyncio
import logging
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, delete, update
from contextlib import asynccontextmanager
import uvicorn
from app.llama3 import generate_books_summary
from app.database import init_db, shutdown_engine, get_db
from app.models import Book, Review, User
from app.auth import authenticate, get_password_hash
# from app.auth import authenticate
from app.schemas import *
from app.logging_config import setup_logging

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("in lifespan")
    await init_db()
    yield
    await shutdown_engine()


app = FastAPI(lifespan=lifespan)
logger = logging.getLogger("fastapi")


async def get_recommendations(genre, min_rating):
    # Implement asynchronous ML model here
    await asyncio.sleep(1)  # Simulate async operation
    return [{"id": 1, "title": "Recommended Book", "author": "Author Name"}]


@app.post("/users", response_model=UserResponse)
async def create_user(user: UserCreate, db: AsyncSession = Depends(get_db)):
    db_user = User(username=user.username, email=user.email,
                   hashed_password=get_password_hash(user.password))
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return UserResponse(**db_user.__dict__)


@app.get("/users", response_model=List[UserResponse])
async def read_users(skip: int = 0, limit: int = 100,
                     db: AsyncSession = Depends(get_db),
                     current_user: User = Depends(authenticate)):
    result = await db.execute(select(User).offset(skip).limit(limit))
    users = result.scalars().all()
    return [UserResponse(**user.__dict__) for user in users]


@app.get("/users/{user_id}", response_model=UserResponse)
async def read_user(user_id: int, db: AsyncSession = Depends(get_db),
                    current_user: User = Depends(authenticate)):
    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(**user.__dict__)


@app.put("/users/{user_id}", response_model=UserResponse)
async def update_user(user_id: int, user: UserUpdate,
                      db: AsyncSession = Depends(get_db),
                      current_user: User = Depends(authenticate)):
    if current_user.id != user_id:
        raise HTTPException(status_code=403,
                            detail="Not authorized to update this user")

    update_data = user.dict(exclude_unset=True)
    if "password" in update_data:
        update_data["hashed_password"] = get_password_hash(update_data.pop("password"))

    result = await db.execute(
        update(User).
        where(User.id == user_id).
        values(**update_data).
        returning(User)
    )
    updated_user = result.scalar_one_or_none()
    if updated_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    await db.commit()
    return UserResponse(**updated_user.__dict__)


@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: int, db: AsyncSession = Depends(get_db),
                      current_user: User = Depends(authenticate)):
    if current_user.id != user_id:
        raise HTTPException(status_code=403,
                            detail="Not authorized to delete this user")

    result = await db.execute(delete(User).where(User.id == user_id))
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="User not found")

    await db.commit()


@app.post("/books/", response_model=BookOut)
async def create_book(book: BookCreate, db: AsyncSession = Depends(get_db),
                      username: str = Depends(authenticate)):
    book_dict = book.model_dump()
    cleaned_summary = book_dict['summary'].strip()
    book_dict['summary'] = cleaned_summary

    new_book = Book(**book_dict)
    db.add(new_book)
    await db.commit()
    await db.refresh(new_book)
    # return db_book
    return BookResponse(**new_book.__dict__)


@app.get("/books/", response_model=List[BookOut])
async def get_books(skip: int = 0, limit: int = 10,
                    db: AsyncSession = Depends(get_db)):
    # result = await db.execute(select(Book).offset(skip).limit(limit))
    result = await db.execute(
        select(Book).options(selectinload(Book.reviews)).offset(skip).limit(limit))
    books = result.scalars().all()
    return [BookResponse(**book.__dict__) for book in books]


@app.get("/books/{book_id}", response_model=BookOut)
async def get_book(book_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Book).filter(Book.id == book_id))
    book = result.scalar_one_or_none()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return BookResponse(**book.__dict__)


@app.put("/books/{book_id}", response_model=BookOut)
async def update_book(book_id: int, book_update: BookUpdate,
                      db: AsyncSession = Depends(get_db),
                      username: str = Depends(authenticate)):
    result = await db.execute(select(Book).filter(Book.id == book_id))
    book = result.scalar_one_or_none()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    update_data = book_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(book, key, value)

    await db.commit()
    await db.refresh(book)
    return BookResponse(**book.__dict__)


@app.delete("/books/{book_id}")
async def delete_book(book_id: int, db: AsyncSession = Depends(get_db),
                      username: str = Depends(authenticate)):
    result = await db.execute(select(Book).filter(Book.id == book_id))
    db_book = result.scalar_one_or_none()
    if not db_book:
        raise HTTPException(status_code=404, detail="Book not found")

    await db.delete(db_book)
    await db.commit()
    return {"message": "Book deleted successfully"}


@app.post("/books/{book_id}/reviews", response_model=ReviewOut)
async def create_review(book_id: int, review: ReviewCreate,
                        db: AsyncSession = Depends(get_db)):
    new_review = Review(book_id=book_id, **review.model_dump())
    db.add(new_review)
    await db.commit()
    await db.refresh(new_review)
    return ReviewResponse(**new_review.__dict__)


@app.get("/books/{book_id}/reviews", response_model=List[ReviewOut])
async def read_reviews(book_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Review).filter(Review.book_id == book_id))
    reviews = result.scalars().all()
    return [ReviewResponse(**review.__dict__) for review in reviews]


@app.get("/books/{book_id}/summary")
async def get_book_summary(book_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Book).filter(Book.id == book_id))
    book = result.scalar_one_or_none()
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")

    reviews_result = await db.execute(select(Review).filter(Review.book_id == book_id))
    reviews = reviews_result.scalars().all()

    avg_rating = sum(review.rating for review in reviews) / len(
        reviews) if reviews else 0

    return {
        "title": book.title,
        "summary": book.summary,
        "average_rating": avg_rating,
        "num_reviews": len(reviews)
    }


@app.get("/recommendations")
async def get_book_recommendations(genre: str, min_rating: float = 0.0):
    recommendations = await get_recommendations(genre, min_rating)
    return recommendations


@app.post("/generate-summary")
async def generate_book_summary(book_id: int, db: AsyncSession = Depends(get_db),
                                username: str = Depends(authenticate)):
    logger.info("main: generate book summary")
    summary = await generate_books_summary(book_id, db)
    return {"summary": summary}


#
# @app.on_event("startup")
# async def on_startup():
#     await init_db()
#
#
# @app.on_event("shutdown")
# async def shutdown():
#     await shutdown_engine()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
