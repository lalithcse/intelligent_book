from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models import User
from app.database import get_async_session

security = HTTPBasic()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# async def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
#     correct_username = "admin"
#     correct_password = "password"
#     if (credentials.username != correct_username or
#             credentials.password != correct_password):
#         raise HTTPException(status_code=401, detail="Invalid credentials")
#     return credentials.username
#

async def get_user(db: AsyncSession, username: str):
    result = await db.execute(select(User).filter(User.username == username))
    return result.scalar_one_or_none()


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


async def authenticate(
    credentials: HTTPBasicCredentials = Depends(security),
    db: AsyncSession = Depends(get_async_session)
):
    user = await get_user(db, credentials.username)
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return user
