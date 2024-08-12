from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from fastapi import HTTPException, Depends
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.llama_model import CustomLlamaModel
import torch
import logging
import os
import json
from app.models import Book
from app.database import get_db

MODEL_PATH = os.path.abspath("meta-llama/Llama-Guard3/Meta-Llama-Guard-3-8B")
tokenizer = None
model = None
logger = logging.getLogger("fastapi")

if not os.path.exists(MODEL_PATH):
    logger.error(f"Model path does not exist: {MODEL_PATH}")
else:
    logger.info(f"Model path exists: {MODEL_PATH}")

with open(os.path.join(MODEL_PATH, "params.json"), "r") as f:
    params = json.load(f)

model = CustomLlamaModel(params)
state_dict = torch.load(os.path.join(MODEL_PATH, "consolidated.00.pth"),
                        map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

if torch.cuda.is_available():
    model = model.cuda()


def load_model():
    """
    Load model and tokenizer if not already loaded
    """
    global tokenizer, model
    if tokenizer is None or model is None:
        try:
            tokenizer_path = os.path.join(MODEL_PATH, "tokenizer.model")
            model_path = MODEL_PATH

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

            if torch.cuda.is_available():
                model = model.to('cuda')

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail="Model loading failed")


def generate_summary(book_content: str, max_length=500) -> str:
    """
    Generate summary for a book
    :param book_content: book content
    :param max_length: maximum length of summary
    :return: summary
    """
    global tokenizer, model
    logger.info("Generating summary")
    load_model()

    prompt = f"Summarize the following book content:\n\n{book_content}\n\nSummary:"
    inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True,
                              max_length=2048)

    outputs = model.generate(
        inputs.input_ids.to(model.device),
        max_length=max_length,
        temperature=0.7,
        top_p=0.95
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Generated summary: {summary}")

    return summary.split("Summary:")[
        1].strip() if "Summary:" in summary else summary.strip()


async def generate_books_summary(book_id: int, db: AsyncSession):
    """
    Generate summary for a book
    :param book_id: book id
    :param db: database session
    :return: summary
    """
    logger.info(f"Generating summary for book ID: {book_id}")
    result = await db.execute(select(Book).filter(Book.id == book_id))
    book = result.scalar_one_or_none()

    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    summary = generate_summary(book.content)
    book.summary = summary
    await db.commit()
    await db.refresh(book)
    return summary
