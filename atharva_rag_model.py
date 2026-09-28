# ============================================================
# GENERAL-PURPOSE DOCUMENT RAG ASSISTANT
# ONE-CELL GOOGLE COLAB VERSION  —  REDESIGNED UI
# ============================================================

# ============================================================
# 1. INSTALL SYSTEM PACKAGES
# ============================================================

print("Installing system packages...")

!apt-get update -qq
!apt-get install -y -qq poppler-utils tesseract-ocr

print("System packages installed.")


# ============================================================
# 2. INSTALL PYTHON PACKAGES
# ============================================================

print("\nInstalling Python packages...")

!pip install -q \
    langchain \
    langchain-community \
    langchain-huggingface \
    langchain-text-splitters \
    faiss-cpu \
    pypdf \
    pdf2image \
    pytesseract \
    pillow \
    sentence-transformers \
    transformers \
    accelerate \
    bitsandbytes \
    gradio

print("Package installation complete.")


# ============================================================
# 3. IMPORTS
# ============================================================

import os
import re
import gc
import json
import time
import uuid
import shutil
import sqlite3
import hashlib
import logging
import traceback

from pathlib import Path
from collections import defaultdict

import torch
import gradio as gr

from PIL import Image

from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    BlipForConditionalGeneration,
    BitsAndBytesConfig,
    pipeline
)

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 4. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("DocuMind")


# ============================================================
# 5. DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Speed: let cuDNN pick the fastest conv/attention kernels once it
# has seen a few shapes, and allow TF32 matmuls on Ampere+ GPUs.
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print("\n==========================================")
print("DEVICE INFORMATION")
print("==========================================")
print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU: Not available - using CPU")

print("==========================================\n")


# ============================================================
# 6. CONFIGURATION
# ============================================================

class Config:

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    DATA_PATH = Path("uploaded_docs")
    INDEX_PATH = Path("faiss_indexes")
    HISTORY_DB_PATH = Path("chat_history.db")

    DATA_PATH.mkdir(exist_ok=True)
    INDEX_PATH.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

    LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

    BLIP_MODEL = "Salesforce/blip-image-captioning-base"

    # Load the LLM in 4-bit on GPU for a large speed/VRAM win.
    # Automatically ignored on CPU (bitsandbytes needs CUDA).
    USE_4BIT = True

    # --------------------------------------------------------
    # Chunking
    # --------------------------------------------------------

    CHUNK_SIZE = 900
    CHUNK_OVERLAP = 150

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    PER_DOC_TOP_K = 2

    MAX_DOCS_IN_CONTEXT = 6

    MAX_CHARS_PER_CHUNK = 320

    SIMILARITY_THRESHOLD = 0.30

    # Batch size used when embedding chunks — bigger batches are
    # much faster than the HuggingFaceEmbeddings default of 1-by-1.
    EMBEDDING_BATCH_SIZE = 64

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    MAX_NEW_TOKENS = 220

    COMPARISON_MAX_NEW_TOKENS = 400

    GENERATION_TIMEOUT = 100

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    OCR_LANG = "eng"

    # --------------------------------------------------------
    # Session
    # --------------------------------------------------------

    DEFAULT_SESSION_ID = "default"


config = Config()

DEFAULT_SESSION_ID = config.DEFAULT_SESSION_ID


# ============================================================
# 7. SUPPORTED FILE TYPES
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif"
}


# ============================================================
# 8. BLIP IMAGE CAPTION MODEL
# ============================================================

print("\nLoading BLIP image captioning model...")

try:

    blip_processor = AutoProcessor.from_pretrained(
        config.BLIP_MODEL
    )

    blip_model = BlipForConditionalGeneration.from_pretrained(
        config.BLIP_MODEL,
        low_cpu_mem_usage=True
    )

    blip_model = blip_model.to(DEVICE)

    blip_model.eval()

    print("BLIP loaded successfully.")

except Exception as e:

    print("BLIP loading failed:")
    print(e)

    blip_processor = None
    blip_model = None


# ============================================================
# 9. IMAGE CAPTIONING
# ============================================================

def generate_image_caption(image):

    if blip_model is None or blip_processor is None:
        return ""

    try:

        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = blip_processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(DEVICE)
            for k, v in inputs.items()
        }

        with torch.no_grad():

            output = blip_model.generate(
                **inputs,
                max_new_tokens=60,
                num_beams=1,
                do_sample=False
            )

        caption = blip_processor.decode(
            output[0],
            skip_special_tokens=True
        )

        return caption.strip()

    except Exception as e:

        logger.warning(
            "BLIP caption failed: %s",
            e
        )

        return ""


# ============================================================
# 10. OCR
# ============================================================

def perform_ocr(image):

    try:

        if image.mode != "RGB":
            image = image.convert("RGB")

        text = pytesseract.image_to_string(
            image,
            lang=config.OCR_LANG
        )

        return text.strip()

    except Exception as e:

        logger.warning(
            "OCR failed: %s",
            e
        )

        return ""


# ============================================================
# 11. VISUAL DESCRIPTION
# ============================================================

def describe_image(image):

    ocr_text = perform_ocr(image)

    caption = generate_image_caption(image)

    result_parts = []

    if caption:
        result_parts.append(
            "Visual description:\n" + caption
        )

    if ocr_text:
        result_parts.append(
            "OCR text:\n" + ocr_text
        )

    return "\n\n".join(result_parts)


# ============================================================
# 12. PDF PROCESSING
# ============================================================

def process_pdf(file_path):

    documents = []

    file_name = Path(file_path).name

    print(
        f"Processing PDF: {file_name}"
    )

    # --------------------------------------------------------
    # Try native PDF extraction first
    # --------------------------------------------------------

    try:

        reader = PdfReader(file_path)

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            try:

                text = page.extract_text()

            except Exception:

                text = ""

            if text and text.strip():

                documents.append(
                    Document(
                        page_content=text.strip(),
                        metadata={
                            "source": file_name,
                            "file_path": str(file_path),
                            "page": page_number,
                            "content_type": "pdf_text"
                        }
                    )
                )

    except Exception as e:

        print(
            f"Native PDF extraction failed: {e}"
        )

    # --------------------------------------------------------
    # If native extraction failed / empty,
    # use OCR on rendered PDF pages
    # --------------------------------------------------------

    if not documents:

        print(
            f"No native text found in {file_name}."
        )

        print(
            "Rendering PDF pages for OCR..."
        )

        try:

            # Lower DPI = faster rasterization with little OCR
            # accuracy loss for typical scanned documents.
            pages = convert_from_path(
                file_path,
                dpi=120,
                thread_count=os.cpu_count() or 2
            )

            for page_number, image in enumerate(
                pages,
                start=1
            ):

                ocr_text = perform_ocr(
                    image
                )

                caption = generate_image_caption(
                    image
                )

                content_parts = []

                if ocr_text:
                    content_parts.append(
                        "OCR text:\n" + ocr_text
                    )

                if caption:
                    content_parts.append(
                        "Visual description:\n"
                        + caption
                    )

                content = "\n\n".join(
                    content_parts
                )

                if content.strip():

                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": file_name,
                                "file_path": str(file_path),
                                "page": page_number,
                                "content_type":
                                    "scanned_pdf"
                            }
                        )
                    )

        except Exception as e:

            print(
                "PDF OCR failed:",
                e
            )

    return documents


# ============================================================
# 13. IMAGE PROCESSING
# ============================================================

def process_image(file_path):

    file_name = Path(file_path).name

    print(
        f"Processing image: {file_name}"
    )

    try:

        image = Image.open(file_path)

        description = describe_image(
            image
        )

        if not description.strip():
            return []

        return [
            Document(
                page_content=description,
                metadata={
                    "source": file_name,
                    "file_path": str(file_path),
                    "content_type": "image"
                }
            )
        ]

    except Exception as e:

        print(
            f"Image processing failed: {e}"
        )

        return []


# ============================================================
# 14. TEXT / MARKDOWN PROCESSING
# ============================================================

def process_text(file_path):

    file_name = Path(file_path).name

    print(
        f"Processing text: {file_name}"
    )

    try:

        text = Path(
            file_path
        ).read_text(
            encoding="utf-8",
            errors="ignore"
        )

        if not text.strip():
            return []

        return [
            Document(
                page_content=text,
                metadata={
                    "source": file_name,
                    "file_path": str(file_path),
                    "content_type": "text"
                }
            )
        ]

    except Exception as e:

        print(
            f"Text processing failed: {e}"
        )

        return []


# ============================================================
# 15. GENERAL FILE LOADER
# ============================================================

def load_document(file_path):

    extension = Path(
        file_path
    ).suffix.lower()

    if extension == ".pdf":

        return process_pdf(
            file_path
        )

    elif extension in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif"
    }:

        return process_image(
            file_path
        )

    elif extension in {
        ".txt",
        ".md"
    }:

        return process_text(
            file_path
        )

    return []


# ============================================================
# 16. EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

try:

    # Speed: run embeddings on GPU when available (previously
    # hardcoded to CPU), and embed chunks in batches instead of
    # one at a time.
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={
            "device": DEVICE
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": config.EMBEDDING_BATCH_SIZE
        }
    )

    print(
        "Embedding model loaded."
    )

except Exception as e:

    print(
        "Embedding model failed:",
        e
    )

    raise


# ============================================================
# 17. TEXT SPLITTER
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)


# ============================================================
# 18. GLOBAL DOCUMENT INDEXES
# ============================================================

document_indexes = {}

document_metadata = {}


# ============================================================
# 19. FILE HASH
# ============================================================

def calculate_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(
        file_path,
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha256.update(
                chunk
            )

    return sha256.hexdigest()


# ============================================================
# 20. CREATE DOCUMENT INDEX
# ============================================================

def create_document_index(
    source_name,
    documents
):

    if not documents:
        return None

    # --------------------------------------------------------
    # Split documents
    # --------------------------------------------------------

    chunks = text_splitter.split_documents(
        documents
    )

    if not chunks:
        return None

    # --------------------------------------------------------
    # Add chunk metadata
    # --------------------------------------------------------

    for i, chunk in enumerate(chunks):

        chunk.metadata[
            "chunk_id"
        ] = i

        chunk.metadata[
            "document"
        ] = source_name

        # Keep chunks manageable
        if len(chunk.page_content) > \
                config.MAX_CHARS_PER_CHUNK:

            chunk.page_content = (
                chunk.page_content[
                    :config.MAX_CHARS_PER_CHUNK
                ]
                + "..."
            )

    # --------------------------------------------------------
    # Create FAISS index (embeddings are batched internally
    # via encode_kwargs["batch_size"] set above)
    # --------------------------------------------------------

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    document_indexes[
        source_name
    ] = vectorstore

    document_metadata[
        source_name
    ] = {
        "chunks": len(chunks),
        "source": source_name
    }

    return vectorstore


# ============================================================
# 21. SAVE FAISS INDEX
# ============================================================

def save_document_index(
    source_name,
    vectorstore
):

    safe_name = re.sub(
        r"[^a-zA-Z0-9_.-]",
        "_",
        source_name
    )

    index_directory = (
        config.INDEX_PATH
        / safe_name
    )

    index_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        vectorstore.save_local(
            str(index_directory)
        )

    except Exception as e:

        logger.warning(
            "Could not save FAISS index: %s",
            e
        )


# ============================================================
# 22. INGEST DOCUMENTS
# ============================================================

def ingest_files(
    uploaded_files
):

    if not uploaded_files:

        return (
            "No files selected."
        )

    logs = []

    processed_count = 0

    for uploaded_file in uploaded_files:

        try:

            original_path = Path(
                uploaded_file
            )

            file_name = (
                original_path.name
            )

            extension = (
                original_path.suffix.lower()
            )

            if extension not in \
                    SUPPORTED_EXTENSIONS:

                logs.append(
                    f"❌ Unsupported: "
                    f"{file_name}"
                )

                continue

            destination = (
                config.DATA_PATH
                / file_name
            )

            # ------------------------------------------------
            # Copy file
            # ------------------------------------------------

            shutil.copy2(
                original_path,
                destination
            )

            # ------------------------------------------------
            # Calculate hash
            # ------------------------------------------------

            file_hash = calculate_file_hash(
                destination
            )

            logs.append(
                f"\n📄 {file_name}"
            )

            logs.append(
                f"Hash: {file_hash[:12]}..."
            )

            # ------------------------------------------------
            # Process
            # ------------------------------------------------

            documents = load_document(
                destination
            )

            if not documents:

                logs.append(
                    "⚠️ No readable content."
                )

                continue

            logs.append(
                f"Extracted sections: "
                f"{len(documents)}"
            )

            # ------------------------------------------------
            # Create vector index
            # ------------------------------------------------

            vectorstore = create_document_index(
                file_name,
                documents
            )

            if vectorstore is None:

                logs.append(
                    "❌ Index creation failed."
                )

                continue

            save_document_index(
                file_name,
                vectorstore
            )

            logs.append(
                f"✅ Indexed successfully."
            )

            logs.append(
                f"Chunks: "
                f"{document_metadata[file_name]['chunks']}"
            )

            processed_count += 1

        except Exception as e:

            logs.append(
                f"❌ Error: {e}"
            )

            logger.exception(
                "Document ingestion failed"
            )

    logs.insert(
        0,
        (
            f"Processed "
            f"{processed_count} document(s)."
        )
    )

    return "\n".join(
        logs
    )


# ============================================================
# 23. RESET KNOWLEDGE BASE
# ============================================================

def reset_knowledge_base():

    global document_indexes
    global document_metadata

    document_indexes = {}

    document_metadata = {}

    try:

        if config.DATA_PATH.exists():

            shutil.rmtree(
                config.DATA_PATH
            )

        config.DATA_PATH.mkdir(
            exist_ok=True
        )

    except Exception as e:

        print(
            "Data cleanup error:",
            e
        )

    try:

        if config.INDEX_PATH.exists():

            shutil.rmtree(
                config.INDEX_PATH
            )

        config.INDEX_PATH.mkdir(
            exist_ok=True
        )

    except Exception as e:

        print(
            "Index cleanup error:",
            e
        )

    gc.collect()

    if DEVICE == "cuda":

        torch.cuda.empty_cache()

    return (
        "Knowledge base cleared.",
        [],
        []
    )


# ============================================================
# 24. SQLITE DATABASE
# ============================================================

def initialize_database():

    conn = sqlite3.connect(
        str(config.HISTORY_DB_PATH)
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()

    conn.close()


initialize_database()


# ============================================================
# 25. SAVE CHAT MESSAGE
# ============================================================

def save_turn(
    session_id,
    role,
    content
):

    try:

        conn = sqlite3.connect(
            str(config.HISTORY_DB_PATH)
        )

        conn.execute(
            """
            INSERT INTO chat_history
            (session_id, role, content)
            VALUES (?, ?, ?)
            """,
            (
                session_id,
                role,
                content
            )
        )

        conn.commit()

        conn.close()

    except Exception as e:

        logger.warning(
            "Chat history save error: %s",
            e
        )


# ============================================================
# 26. LOAD CHAT HISTORY
# IMPORTANT:
# Gradio expects dictionary messages in this environment.
# ============================================================

def load_session_history(
    session_id
):

    try:

        conn = sqlite3.connect(
            str(config.HISTORY_DB_PATH)
        )

        rows = conn.execute(
            """
            SELECT role, content
            FROM chat_history
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (
                session_id,
            )
        ).fetchall()

        conn.close()

        return [
            {
                "role": role,
                "content": content
            }
            for role, content in rows
        ]

    except Exception as e:

        print(
            "History load error:",
            e
        )

        return []


# ============================================================
# 27. CLEAR CHAT HISTORY
# ============================================================

def clear_chat_history():

    try:

        conn = sqlite3.connect(
            str(config.HISTORY_DB_PATH)
        )

        conn.execute(
            """
            DELETE FROM chat_history
            WHERE session_id = ?
            """,
            (
                DEFAULT_SESSION_ID,
            )
        )

        conn.commit()

        conn.close()

    except Exception as e:

        print(
            "History clear error:",
            e
        )

    return [], []


# ============================================================
# 28. LOAD QWEN MODEL
# ============================================================

print("\nLoading Qwen model...")

try:

    tokenizer = AutoTokenizer.from_pretrained(
        config.LLM_MODEL
    )

    if DEVICE == "cuda":

        # Speed + memory: 4-bit quantization roughly halves (or
        # better) generation latency on GPU versus fp16, with
        # minimal quality loss for a 1.5B instruct model.
        if config.USE_4BIT:

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            llm_model = AutoModelForCausalLM.from_pretrained(
                config.LLM_MODEL,
                quantization_config=quant_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                attn_implementation="sdpa"
            )

        else:

            llm_model = AutoModelForCausalLM.from_pretrained(
                config.LLM_MODEL,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                attn_implementation="sdpa"
            )

    else:

        llm_model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        llm_model = llm_model.to(
            "cpu"
        )

    llm_model.eval()

    print(
        "Qwen model loaded successfully."
    )

except Exception as e:

    print(
        "Qwen loading failed:"
    )

    traceback.print_exc()

    raise


# ============================================================
# 29. GENERATE TEXT
# ============================================================

def generate_answer(
    system_prompt,
    user_prompt,
    max_new_tokens=None
):

    if max_new_tokens is None:

        max_new_tokens = (
            config.MAX_NEW_TOKENS
        )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    try:

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    except Exception:

        prompt = (
            "System:\n"
            + system_prompt
            + "\n\nUser:\n"
            + user_prompt
            + "\n\nAssistant:"
        )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(
            llm_model.device
        )
        for key, value in inputs.items()
    }

    start_time = time.time()

    with torch.no_grad():

        # Speed: greedy decoding (do_sample=False) is faster than
        # sampling (no per-token softmax-sample overhead) and more
        # deterministic for a document-QA assistant. Repetition
        # penalty still guards against loops without needing
        # temperature/top-p sampling.
        output_ids = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    generated_ids = (
        output_ids[
            0,
            inputs["input_ids"].shape[1]:
        ]
    )

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    elapsed = (
        time.time()
        - start_time
    )

    logger.info(
        "Generation time: %.2fs",
        elapsed
    )

    return answer.strip()


# ============================================================
# 30. RETRIEVE FROM EACH DOCUMENT
# ============================================================

def retrieve_evidence(
    question
):

    evidence = []

    if not document_indexes:

        return evidence

    # --------------------------------------------------------
    # Search every document independently
    # --------------------------------------------------------

    for source_name, vectorstore in (
        document_indexes.items()
    ):

        try:

            results = (
                vectorstore.similarity_search_with_score(
                    question,
                    k=config.PER_DOC_TOP_K
                )
            )

            for doc, score in results:

                # FAISS distance:
                # lower generally means more similar
                similarity = 1 / (
                    1 + float(score)
                )

                evidence.append(
                    {
                        "document": source_name,
                        "page": doc.metadata.get(
                            "page"
                        ),
                        "content": (
                            doc.page_content
                        ),
                        "similarity": similarity,
                        "score": float(score)
                    }
                )

        except Exception as e:

            logger.warning(
                "Retrieval failed for %s: %s",
                source_name,
                e
            )

    # --------------------------------------------------------
    # Sort by similarity
    # --------------------------------------------------------

    evidence.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return evidence


# ============================================================
# 31. GROUP EVIDENCE BY DOCUMENT
# ============================================================

def group_evidence_by_document(
    evidence
):

    grouped = defaultdict(list)

    for item in evidence:

        grouped[
            item["document"]
        ].append(item)

    return grouped


# ============================================================
# 32. BUILD CONTEXT
# ============================================================

def build_context(
    evidence
):

    if not evidence:

        return (
            "No relevant evidence "
            "was retrieved."
        )

    grouped = (
        group_evidence_by_document(
            evidence
        )
    )

    blocks = []

    document_names = list(
        grouped.keys()
    )

    document_names = document_names[
        :config.MAX_DOCS_IN_CONTEXT
    ]

    for document_name in document_names:

        blocks.append(
            f"\n===== DOCUMENT: "
            f"{document_name} ====="
        )

        for item in grouped[
            document_name
        ]:

            page = item.get(
                "page"
            )

            if page:

                blocks.append(
                    f"[Page {page}]"
                )

            blocks.append(
                item["content"]
            )

    return "\n".join(
        blocks
    )


# ============================================================
# 33. DETECT COMPARISON / INVESTIGATION QUERY
# ============================================================

def is_comparison_query(
    question
):

    q = question.lower()

    comparison_words = [
        "compare",
        "comparison",
        "difference",
        "differences",
        "similar",
        "similarities",
        "contrast",
        "versus",
        "vs",
        "across",
        "between",
        "all documents",
        "each document",
        "multiple documents",
        "which document",
        "documents disagree",
        "conflict",
        "conflicting",
        "disagreement",
        "investigate",
        "investigation",
        "cross document"
    ]

    return any(
        word in q
        for word in comparison_words
    )


# ============================================================
# 34. BUILD COMPARISON EVIDENCE
# ============================================================

def build_comparison_context(
    evidence
):

    grouped = (
        group_evidence_by_document(
            evidence
        )
    )

    if not grouped:

        return (
            "No evidence found."
        )

    blocks = []

    for document_name, items in (
        grouped.items()
    ):

        blocks.append(
            f"\n### DOCUMENT: "
            f"{document_name}"
        )

        for item in items:

            page = item.get(
                "page"
            )

            location = ""

            if page:

                location = (
                    f" | Page {page}"
                )

            blocks.append(
                f"[Evidence{location}]\n"
                f"{item['content']}"
            )

    return "\n".join(
        blocks
    )


# ============================================================
# 35. NORMAL ANSWER SYSTEM PROMPT
# ============================================================

NORMAL_SYSTEM_PROMPT = """
You are a careful document intelligence assistant.

Answer the user's question using ONLY the supplied
document evidence.

Rules:

1. Do not invent facts.
2. Do not use outside knowledge when answering
   document-specific questions.
3. If the evidence does not contain the answer,
   clearly say that the documents do not provide
   enough information.
4. Distinguish facts from uncertainty.
5. If several documents contain relevant information,
   mention the document names.
6. Keep the answer clear and useful.
7. Never claim that two documents agree unless the
   supplied evidence actually supports that.
8. Never claim that documents conflict unless their
   supplied evidence gives materially different
   information about the same issue.
"""


# ============================================================
# 36. COMPARISON SYSTEM PROMPT
# ============================================================

COMPARISON_SYSTEM_PROMPT = """
You are a multi-document investigation assistant.

The user has asked a question that may require
comparing information across multiple documents.

Use ONLY the supplied evidence.

For every important claim:

- identify which document supports it
- preserve important differences in wording or values
- distinguish genuine contradictions from differences
  caused by context, dates, units, scope, or wording
- do not invent missing information

If documents disagree, explain the disagreement clearly.

If the documents do not actually disagree, say that
the apparent difference is not necessarily a contradiction.

If evidence is insufficient, say so.

A useful answer should contain:

1. Direct finding
2. Evidence by document
3. Comparison / disagreement analysis
4. Conclusion

Do not manufacture a winner between documents.
"""


# ============================================================
# 37. ANSWER QUESTION
# ============================================================

def answer_question(
    question
):

    if not question or not question.strip():

        return (
            "Please enter a question.",
            []
        )

    if not document_indexes:

        return (
            "No documents have been processed yet. "
            "Please upload and process documents first.",
            []
        )

    question = question.strip()

    # --------------------------------------------------------
    # Retrieve evidence independently from every document
    # --------------------------------------------------------

    evidence = retrieve_evidence(
        question
    )

    if not evidence:

        return (
            "I could not retrieve relevant evidence "
            "from the uploaded documents.",
            []
        )

    # --------------------------------------------------------
    # Filter weak evidence
    # --------------------------------------------------------

    useful_evidence = [
        item
        for item in evidence
        if item["similarity"]
        >= config.SIMILARITY_THRESHOLD
    ]

    # If threshold removes everything,
    # keep strongest evidence instead.
    if not useful_evidence:

        useful_evidence = evidence[
            :min(
                len(evidence),
                config.MAX_DOCS_IN_CONTEXT
            )
        ]

    # --------------------------------------------------------
    # Determine query type
    # --------------------------------------------------------

    comparison_mode = (
        is_comparison_query(
            question
        )
    )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    if comparison_mode:

        context = build_comparison_context(
            useful_evidence
        )

        system_prompt = (
            COMPARISON_SYSTEM_PROMPT
        )

        user_prompt = f"""
USER QUESTION:
{question}

DOCUMENT EVIDENCE:
{context}

Perform a careful multi-document investigation.

Identify the relevant evidence from each document.

Then answer the question.

If there are different values, dates, statements,
or conclusions, preserve them and explain the
difference.

Do not invent information.
"""

        max_tokens = (
            config.COMPARISON_MAX_NEW_TOKENS
        )

    else:

        context = build_context(
            useful_evidence
        )

        system_prompt = (
            NORMAL_SYSTEM_PROMPT
        )

        user_prompt = f"""
USER QUESTION:
{question}

DOCUMENT EVIDENCE:
{context}

Answer the question using the evidence above.

Mention the relevant document names when useful.

If the evidence is insufficient, say so clearly.
"""

        max_tokens = (
            config.MAX_NEW_TOKENS
        )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    try:

        answer = generate_answer(
            system_prompt,
            user_prompt,
            max_new_tokens=max_tokens
        )

    except Exception as e:

        logger.exception(
            "Generation failed"
        )

        answer = (
            "Generation error: "
            + str(e)
        )

    # --------------------------------------------------------
    # Build source list
    # --------------------------------------------------------

    source_names = []

    for item in useful_evidence:

        source = item[
            "document"
        ]

        page = item.get(
            "page"
        )

        if page:

            source_label = (
                f"{source} "
                f"(page {page})"
            )

        else:

            source_label = source

        if source_label not in source_names:

            source_names.append(
                source_label
            )

    return (
        answer,
        source_names
    )


# ============================================================
# 38. GRADIO PROCESS FUNCTION
# ============================================================

def ui_process(
    files
):

    if not files:

        return (
            "Please upload at least one "
            "PDF, TXT, MD, or image file.",
            get_document_table(),
            get_status_markdown()
        )

    try:

        result = ingest_files(
            files
        )

        return (
            result,
            get_document_table(),
            get_status_markdown()
        )

    except Exception as e:

        logger.exception(
            "UI processing error"
        )

        return (
            "Processing error:\n" + str(e),
            get_document_table(),
            get_status_markdown()
        )


# ============================================================
# 38b. UI HELPER FUNCTIONS (new — support the redesigned UI)
# ============================================================

def get_document_table():
    """Return a table of currently indexed documents for display."""

    if not document_metadata:
        return [["—", "—"]]

    rows = []

    for name, meta in document_metadata.items():
        rows.append([name, meta.get("chunks", 0)])

    return rows


def get_status_markdown():
    """Small status line showing how many documents are indexed."""

    count = len(document_indexes)

    if count == 0:
        return "🔴 **No documents indexed yet** — upload files to get started."

    return f"🟢 **{count} document(s) indexed** and ready to query."


def ui_reset_wrapper():
    """Wrap reset_knowledge_base to also refresh the new status widgets."""

    log_msg, chatbot_val, state_val = reset_knowledge_base()

    return (
        log_msg,
        chatbot_val,
        state_val,
        get_document_table(),
        get_status_markdown()
    )


# ============================================================
# 39. GRADIO CHAT FUNCTION
# IMPORTANT:
# HISTORY USES DICTIONARIES
# ============================================================

def ui_chat(
    message,
    history
):

    history = history or []

    if not message or not message.strip():

        return (
            history,
            history,
            ""
        )

    message = message.strip()

    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    save_turn(
        DEFAULT_SESSION_ID,
        "user",
        message
    )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    answer, sources = answer_question(
        message
    )

    # --------------------------------------------------------
    # Add sources
    # --------------------------------------------------------

    if sources:

        answer += (
            "\n\n**Sources:**\n"
            + "\n".join(
                f"- {source}"
                for source in sources
            )
        )

    # --------------------------------------------------------
    # Save assistant answer
    # --------------------------------------------------------

    save_turn(
        DEFAULT_SESSION_ID,
        "assistant",
        answer
    )

    # --------------------------------------------------------
    # Create new Gradio message history
    # --------------------------------------------------------

    new_history = list(
        history
    )

    new_history.append(
        {
            "role": "user",
            "content": message
        }
    )

    new_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return (
        new_history,
        new_history,
        ""
    )


# ============================================================
# 40. INITIAL CHAT HISTORY
# ============================================================

initial_history = load_session_history(
    DEFAULT_SESSION_ID
)


# ============================================================
# 41. GRADIO INTERFACE  —  REDESIGNED
# ============================================================

print(
    "\nCreating Gradio interface..."
)

try:

    gr.close_all()

except Exception:

    pass


CUSTOM_CSS = """
:root {
    --brand: #6366f1;
    --brand-dark: #4338ca;
    --surface: #ffffff;
    --surface-alt: #f8fafc;
    --border: #e5e7eb;
}

.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}

/* Hero header */
#hero {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 55%, #a855f7 100%);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 18px;
    color: white !important;
    box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.45);
}

#hero h1 { color: white !important; margin-bottom: 6px !important; font-size: 1.75rem !important; }
#hero p  { color: rgba(255,255,255,0.9) !important; margin: 2px 0 !important; }

.badge-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.badge {
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.8rem;
    color: white;
}

/* Panel cards */
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.section-title {
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    margin-bottom: 4px !important;
}

.section-sub {
    color: #6b7280 !important;
    font-size: 0.85rem !important;
    margin-bottom: 10px !important;
}

#status-line {
    padding: 10px 14px;
    border-radius: 10px;
    background: var(--surface-alt);
    border: 1px solid var(--border);
    font-size: 0.9rem;
}

#chatbot {
    border-radius: 16px !important;
    border: 1px solid var(--border) !important;
}

footer { display: none !important; }
"""

GRADIO_THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
)

with gr.Blocks(
    title="DocuMind"
) as demo:

    # --------------------------------------------------------
    # Hero header
    # --------------------------------------------------------

    with gr.Column(elem_id="hero"):

        gr.Markdown(
            f"""
# 📄 DocuMind
Upload PDFs, text files, or images and ask questions in plain English — every document is searched individually before answering.

<div class="badge-row">
<span class="badge">🧠 RAG</span>
<span class="badge">🔎 OCR</span>
<span class="badge">👁️ Image understanding</span>
<span class="badge">📚 Multi-document investigation</span>
<span class="badge">⚖️ Evidence comparison</span>
<span class="badge">💾 Persistent history</span>
</div>

<div class="badge-row">
<span class="badge">Model: {config.LLM_MODEL}</span>
<span class="badge">Embeddings: {config.EMBEDDING_MODEL}</span>
<span class="badge">Device: {DEVICE.upper()}</span>
</div>
"""
        )

    # --------------------------------------------------------
    # Main layout
    # --------------------------------------------------------

    with gr.Row(equal_height=False):

        # ====================================================
        # LEFT SIDE — knowledge base
        # ====================================================

        with gr.Column(scale=1):

            with gr.Column(elem_classes="panel"):

                gr.Markdown(
                    "### 1️⃣ Upload documents",
                    elem_classes="section-title"
                )
                gr.Markdown(
                    "PDF • TXT • MD • PNG • JPG • JPEG • WEBP • BMP • TIFF",
                    elem_classes="section-sub"
                )

                file_upload = gr.File(
                    label="Drag & drop files here",
                    file_count="multiple",
                    file_types=[
                        ".pdf", ".txt", ".md",
                        ".png", ".jpg", ".jpeg",
                        ".webp", ".bmp", ".tiff", ".tif"
                    ]
                )

                with gr.Row():
                    process_btn = gr.Button(
                        "⚙️ Process documents",
                        variant="primary",
                        scale=2
                    )
                    reset_btn = gr.Button(
                        "🗑️ Clear",
                        variant="stop",
                        scale=1
                    )

                status_line = gr.Markdown(
                    get_status_markdown(),
                    elem_id="status-line"
                )

                with gr.Accordion("📋 Processing log", open=False):
                    process_log = gr.Textbox(
                        label=None,
                        show_label=False,
                        lines=10,
                        interactive=False
                    )

                with gr.Accordion("📚 Indexed documents", open=True):
                    doc_table = gr.Dataframe(
                        headers=["Document", "Chunks"],
                        value=get_document_table(),
                        interactive=False,
                        row_count=(1, "dynamic"),
                        column_count=(2, "fixed")
                    )

        # ====================================================
        # RIGHT SIDE — chat
        # ====================================================

        with gr.Column(scale=2):

            with gr.Column(elem_classes="panel"):

                gr.Markdown(
                    "### 2️⃣ Ask questions",
                    elem_classes="section-title"
                )
                gr.Markdown(
                    "Try: *\"What is the main conclusion?\"* · "
                    "*\"Compare the findings across all documents.\"* · "
                    "*\"Do the documents disagree about this value?\"*",
                    elem_classes="section-sub"
                )

                chatbot = gr.Chatbot(
                    label=None,
                    show_label=False,
                    height=520,
                    avatar_images=(None, "📄"),
                    value=initial_history,
                    elem_id="chatbot"
                )

                chat_state = gr.State(initial_history)

                with gr.Row():
                    question_box = gr.Textbox(
                        label=None,
                        show_label=False,
                        placeholder="Ask something about your uploaded documents…",
                        scale=5,
                        container=False
                    )
                    ask_btn = gr.Button("Ask ➤", variant="primary", scale=1)

                clear_chat_btn = gr.Button(
                    "🗑️ Clear chat history",
                    variant="secondary",
                    size="sm"
                )

    # ========================================================
    # EVENTS
    # ========================================================

    process_btn.click(
        fn=ui_process,
        inputs=file_upload,
        outputs=[process_log, doc_table, status_line]
    )

    ask_btn.click(
        fn=ui_chat,
        inputs=[question_box, chat_state],
        outputs=[chatbot, chat_state, question_box]
    )

    question_box.submit(
        fn=ui_chat,
        inputs=[question_box, chat_state],
        outputs=[chatbot, chat_state, question_box]
    )

    reset_btn.click(
        fn=ui_reset_wrapper,
        outputs=[process_log, chatbot, chat_state, doc_table, status_line]
    )

    clear_chat_btn.click(
        fn=clear_chat_history,
        outputs=[chatbot, chat_state]
    )


# ============================================================
# 42. READY
# ============================================================

print("\n==========================================")
print("DOCUMIND READY")
print("==========================================")

print(
    f"Documents indexed: "
    f"{len(document_indexes)}"
)

print(
    f"Chat database: "
    f"{config.HISTORY_DB_PATH}"
)

print(
    "=========================================="
)


# ============================================================
# 43. LAUNCH GRADIO
# ============================================================

demo.launch(
    share=True,
    debug=False,
    theme=GRADIO_THEME,
    css=CUSTOM_CSS
)
