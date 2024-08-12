from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from fastapi import HTTPException, Depends
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
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
    print(f"Model path does not exist: {MODEL_PATH}")
else:
    print(f"Model path exists: {MODEL_PATH}")

with open(MODEL_PATH + "/params.json", "r") as f:
    params = json.load(f)

model = CustomLlamaModel(params)
state_dict = torch.load(MODEL_PATH + "/consolidated.00.pth", map_location='cpu')
model.load_state_dict(state_dict)

# Set to evaluation mode
model.eval()

# Move to GPU if available
if torch.cuda.is_available():
    model = model.cuda()

def load_model1():
    global tokenizer, model
    if tokenizer is None or model is None:
        try:
            # tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
            # model = LlamaForCausalLM.from_pretrained(
            #     MODEL_PATH,
            #     torch_dtype=torch.float16,
            #     device_map="auto" if torch.cuda.is_available() else None
            # )
            
            tokenizer_path = os.path.join(MODEL_PATH, "tokenizer.model")  # Adjust if your setup differs
            model_path = os.path.join(MODEL_PATH, "model_file_name")  # Specify your model file name here
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            
        
            # Load model directly
            # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            #
            # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
            # model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
            # Load model directly

            # from transformers import AutoTokenizer, AutoModelForMaskedLM
            #
            # tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
            # model = AutoModelForMaskedLM.from_pretrained(
            #     "google-bert/bert-base-uncased")

            # model_name = "facebook/bart-large-cnn"
            # model = BartForConditionalGeneration.from_pretrained(model_name)
            # tokenizer = BartTokenizer.from_pretrained(model_name)

        except Exception as e:
            print(f"Error loading model: {e}")
            raise


def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        try:
            tokenizer_path = os.path.dirname(os.path.join(MODEL_PATH, "tokenizer.model"))
            model_path = MODEL_PATH
            
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            
            if torch.cuda.is_available():
                model = model.to('cuda')
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise


def generate_summary(book_content: str, max_length=500) -> str:
    logger.info("generate summary")
    load_model()
    prompt = f"Summarize the following book content:\n\n{book_content}\n\nSummary:"
    
    logger.info("generate summary prompt : {}".format(prompt))
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    logger.info("inputs : {}".format(inputs))
    
    outputs = model.generate(
        inputs.input_ids.to(model.device),
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.95,
        do_sample=True
    )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # input_text = "Summarize the following content:"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_length=100)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    logger.info("summary : {}".format(summary))
    return summary.split("Summary:")[1].strip()

    # inputs = tokenizer.encode("summarize: " + prompt, return_tensors="pt",
    #                           max_length=1024, truncation=True)
    # summary_ids = model.generate(inputs, max_length=150, min_length=50,
    #                              length_penalty=2.0, num_beams=4, early_stopping=True)
    # summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # return summary


async def generate_books_summary(book_id: int, db: AsyncSession):
    logger.info("generate books summary")
    result = await db.execute(select(Book).filter(Book.id == book_id))
    book = result.scalar_one_or_none()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    summary = generate_summary(book.summary)
    book.summary = summary
    await db.commit()
    await db.refresh(book)
    return summary
