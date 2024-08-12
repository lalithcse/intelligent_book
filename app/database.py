from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import logging

DATABASE_URL = os.getenv('DATABASE_URL',
                         "postgresql+asyncpg://postgres:SendHub101@10.1.1.73/intelligent")

engine = create_async_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine,
                            class_=AsyncSession,
                            expire_on_commit=False,
                            autocommit=False,
                            autoflush=False)
Base = declarative_base()
logger = logging.getLogger("fastapi")


async def get_db():
    logger.info("in get_db")
    async with SessionLocal() as session:
        yield session


async def get_async_session() -> AsyncSession:
    async with SessionLocal() as session:
        yield session


async def init_db():
    async with engine.begin() as conn:
        logger.info("in init_db")
        await conn.run_sync(Base.metadata.create_all)


async def shutdown_engine():
    await engine.dispose()
