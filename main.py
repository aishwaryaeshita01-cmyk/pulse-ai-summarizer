import os
import io
import json
import logging
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from groq import Groq, GroqError
from pydub import AudioSegment

# Load environment variables from .env
load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("pulse.ai")

# Initialize FastAPI application
app = FastAPI(
    title="Pulse AI Audio & Transcript Summarizer",
    description="Enterprise-grade Speech-to-Text and Structured Summarization Pipeline",
    version="1.0.0",
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client (reads GROQ_API_KEY from environment)
groq_client = Groq()

# ==========================================
# 1. MODEL REGISTRY & SYSTEM CONFIGURATION
# ==========================================
AUDIO_MODELS = [
    "whisper-large-v3-turbo",
    "whisper-large-v3",
]

LLM_REGISTRY = {
    "high_tier_reasoning": [
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
        "qwen/qwen3.6-27b",
    ],
    "fast_summary": [
        "qwen/qwen3.6-27b",
        "openai/gpt-oss-20b",
    ],
}

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac"}
MAX_FILE_SIZE_MB = 100
CHUNK_THRESHOLD_MB = 24
CHUNK_DURATION_MS = 10 * 60 * 1000  # 10-minute audio segments


# ==========================================
# 2. STRICT PYDANTIC SCHEMAS
# ==========================================
class StructuredSummary(BaseModel):
    executive_summary: str = Field(..., description="A concise 2-3 sentence overview.")
    key_takeaways: List[str] = Field(..., description="Core bullet points extracted from the transcript.")
    action_items: List[str] = Field(default_factory=list, description="Explicit next steps or tasks mentioned.")
    sentiment: str = Field(..., description="Overall tone: Positive, Neutral, Urgent, or Critical.")


class TranscriptRequest(BaseModel):
    transcript_text: str = Field(..., min_length=1)
    tier: Optional[str] = "high_tier_reasoning"


class PipelineResponse(BaseModel):
    success: bool
    transcription_model: Optional[str] = None
    summarization_model: str
    transcript: str
    analysis: StructuredSummary


# ==========================================
# 3. CORE ORCHESTRATION FUNCTIONS
# ==========================================
def transcribe_audio_bytes(file_bytes: bytes, filename: str) -> tuple[str, str]:
    """Automatically chunks audio > 24MB and transcribes with Whisper fallback."""
    size_mb = len(file_bytes) / (1024 * 1024)
    ext = os.path.splitext(filename)[1].lstrip(".").lower() or "mp3"

    chunks = None
    if size_mb > CHUNK_THRESHOLD_MB:
        logger.info(f"Large file detected ({size_mb:.1f}MB). Splitting into 10-minute chunks...")
        try:
            audio = AudioSegment.from_file(io.BytesIO(file_bytes), format=ext)
            chunks = [
                audio[i : i + CHUNK_DURATION_MS]
                for i in range(0, len(audio), CHUNK_DURATION_MS)
            ]
        except Exception as e:
            logger.error(f"Audio chunking failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to decode or segment large audio file. Ensure ffmpeg is installed.",
            )

    for model_id in AUDIO_MODELS:
        try:
            logger.info(f"Attempting transcription with {model_id}")
            if not chunks:
                res = groq_client.audio.transcriptions.create(
                    file=(filename, file_bytes),
                    model=model_id,
                    response_format="json",
                    temperature=0.0,
                )
                return res.text, model_id

            # Process chunks sequentially and stitch transcript together
            full_transcript = []
            for idx, chunk in enumerate(chunks):
                logger.info(f"Transcribing chunk {idx + 1}/{len(chunks)} using {model_id}")
                buf = io.BytesIO()
                chunk.export(buf, format="mp3")
                res = groq_client.audio.transcriptions.create(
                    file=(f"chunk_{idx}.mp3", buf.getvalue()),
                    model=model_id,
                    response_format="json",
                    temperature=0.0,
                )
                full_transcript.append(res.text.strip())

            return " ".join(full_transcript), model_id

        except GroqError as e:
            logger.warning(f"Audio model {model_id} failed: {e}")
            continue

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="All upstream speech-to-text models failed.",
    )


def generate_structured_summary(transcript: str, tier: str) -> tuple[StructuredSummary, str]:
    """Generates and validates a strict JSON summary against the Pydantic schema."""
    candidates = LLM_REGISTRY.get(tier)
    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tier '{tier}'. Available tiers: {list(LLM_REGISTRY.keys())}",
        )

    schema_instruction = json.dumps(StructuredSummary.model_json_schema(), indent=2)
    system_prompt = (
        "You are an enterprise systems analyst. Analyze the provided transcript and return ONLY "
        "a valid JSON object strictly adhering to this JSON Schema:\n"
        f"{schema_instruction}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Transcript:\n{transcript}"},
    ]

    for model_id in candidates:
        try:
            logger.info(f"Generating structured summary with {model_id}")
            response = groq_client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            raw_json = response.choices[0].message.content
            validated_data = StructuredSummary.model_validate_json(raw_json)
            return validated_data, model_id

        except (GroqError, ValidationError, json.JSONDecodeError) as e:
            logger.warning(f"Model {model_id} failed execution or schema validation: {e}")
            continue

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="All summarization models failed or violated output schema.",
    )


# ==========================================
# 4. API ENDPOINTS
# ==========================================
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy", "service": "pulse-ai-summarizer"}


@app.post("/api/v1/summarize", response_model=PipelineResponse)
async def summarize_text(payload: TranscriptRequest):
    analysis, model_used = generate_structured_summary(payload.transcript_text, payload.tier)
    return PipelineResponse(
        success=True,
        summarization_model=model_used,
        transcript=payload.transcript_text,
        analysis=analysis,
    )


@app.post("/api/v1/process-audio", response_model=PipelineResponse)
async def process_audio_file(
    file: UploadFile = File(...),
    tier: str = Form("high_tier_reasoning"),
):
    # 1. Validate file extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}",
        )

    # 2. Read and validate file size
    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb == 0 or size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file must be between 1 byte and {MAX_FILE_SIZE_MB}MB.",
        )

    # 3. Execute Speech-to-Text + Structured Summarization
    transcript_text, stt_model = transcribe_audio_bytes(file_bytes, file.filename)
    analysis, llm_model = generate_structured_summary(transcript_text, tier)

    return PipelineResponse(
        success=True,
        transcription_model=stt_model,
        summarization_model=llm_model,
        transcript=transcript_text,
        analysis=analysis,
    )
