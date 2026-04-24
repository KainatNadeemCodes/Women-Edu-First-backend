rom fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional, List
import httpx, os, uuid
from datetime import datetime
import hashlib, hmac, base64
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="NextGenShe API", version="1.0.0")

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://women-edu-first.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-in-prod")
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# ── In-memory DB ──────────────────────────────────────────────────────────────
users_db: dict = {}
enrollments_db: dict = {}
contacts_db: list = []
progress_db: dict = {}

# ── MODELS ────────────────────────────────────────────────────────────────────
class UserSignup(BaseModel):
    name: str
    email: EmailStr
    password: str
    is_anonymous: bool = False

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class EnrollRequest(BaseModel):
    user_id: str
    course_id: int
    course_title: str

class ProgressUpdate(BaseModel):
    user_id: str
    course_id: int
    lesson_index: int
    completed: bool

class ChatMessage(BaseModel):
    message: str
    path: str
    level: str
    user_name: str
    history: List[dict] = []

class ContactForm(BaseModel):
    name: str
    email: EmailStr
    message: str

# ── HELPERS ───────────────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def make_token(user_id: str) -> str:
    payload = f"{user_id}:{datetime.utcnow().isoformat()}"
    sig = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
    token = base64.b64encode(f"{payload}:{sig}".encode()).decode()
    return token

def verify_token(token: str) -> Optional[str]:
    try:
        decoded = base64.b64decode(token.encode()).decode()
        parts = decoded.rsplit(":", 2)
        user_id, timestamp, sig = parts[0], parts[1], parts[2]
        expected = hmac.new(
            SECRET_KEY.encode(),
            f"{user_id}:{timestamp}".encode(),
            hashlib.sha256
        ).hexdigest()
        if hmac.compare_digest(sig, expected):
            return user_id
    except Exception:
        pass
    return None

def build_system_prompt(user_name: str, path: str, level: str) -> str:
    return f"""You are Zara, a warm, supportive AI mentor on NextGenShe — a platform for girls and women learning tech skills to earn from home, especially in Pakistan and South Asia.
User: {user_name} | Path: {path} | Level: {level}
Your personality:
- Warm and encouraging, like a big sister or mentor
- Mix English with Urdu/Roman Urdu naturally (e.g., "Bohat acha!", "Bilkul kar sakti ho!", "Koi baat nahi!")
- Never make them feel behind or incapable
- Give practical, actionable steps with realistic timelines
- Keep responses concise — 3-4 short paragraphs max
- When giving roadmaps, use numbered steps
- Focus on learning to earning pathway
- Everything can be done from home on a phone or basic laptop
For earning questions: give specific platforms (Fiverr, Upwork) and realistic income ranges.
For roadmap requests: give week-by-week steps.
For motivation: celebrate their courage first, then help."""

# ── AI: CLAUDE PRIMARY ────────────────────────────────────────────────────────
async def call_claude(data: ChatMessage) -> str:
    system_prompt = build_system_prompt(data.user_name, data.path, data.level)
    
    # Filter history to keep Anthropic happy (only user/assistant roles)
    filtered_history = [m for m in data.history if m["role"] in ["user", "assistant"]]
    messages = filtered_history[-10:] + [{"role": "user", "content": data.message}]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-3-haiku-20240307",
                "max_tokens": 800,
                "system": system_prompt,
                "messages": messages,
            },
        )
        
        if response.status_code != 200:
            error_detail = response.text
            print(f"Claude API error {response.status_code}: {error_detail}")
            raise Exception(f"Claude error: {response.status_code}")

        result = response.json()
        return result["content"][0]["text"]

# ── AI: HUGGING FACE FALLBACK ─────────────────────────────────────────────────
async def call_huggingface(data: ChatMessage) -> str:
    system_prompt = build_system_prompt(data.user_name, data.path, data.level)
    history_text = ""
    for msg in data.history[-6:]:
        role = "User" if msg["role"] == "user" else "Zara"
        history_text += f"{role}: {msg['content']}\n"
    
    prompt = f"<s>[INST] <<SYS>>{system_prompt}<</SYS>>{history_text}User: {data.message} [/INST]Zara:"
    
    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False,
                },
            },
        )
    
    if response.status_code == 503:
        raise Exception("HF model is loading, please wait...")
    if response.status_code != 200:
        raise Exception(f"HF error: {response.status_code}")
    
    result = response.json()
    if isinstance(result, list) and len(result) > 0:
        reply = result[0].get("generated_text", "").strip()
        if "Zara:" in reply:
            reply = reply.split("Zara:")[-1].strip()
        if "[/INST]" in reply:
            reply = reply.split("[/INST]")[-1].strip()
        return reply if reply else "Thodi mushkil aa gayi, please dobara try karein! 💜"
    raise Exception("Unexpected HF response")

# ── AI SMART ROUTER ───────────────────────────────────────────────────────────
async def get_ai_reply(data: ChatMessage) -> tuple[str, str]:
    if ANTHROPIC_API_KEY:
        try:
            reply = await call_claude(data)
            return reply, "claude"
        except Exception as e:
            print(f"⚠️ Claude failed ({e}) — switching to HuggingFace")
    
    if HF_TOKEN:
        try:
            reply = await call_huggingface(data)
            return reply, "huggingface"
        except Exception as e:
            print(f"⚠️ HuggingFace also failed: {e}")
            
    raise HTTPException(
        status_code=500,
        detail="AI service temporarily unavailable. Please try again! 💜"
    )

# ── HEALTH ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "status": "NextGenShe API is running! 💜",
        "version": "1.0.0",
        "ai_primary": "Claude (Anthropic)" if ANTHROPIC_API_KEY else "not configured",
        "ai_fallback": f"HuggingFace ({HF_MODEL})" if HF_TOKEN else "not configured",
    }

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "users": len(users_db),
        "enrollments": len(enrollments_db),
        "claude_configured": bool(ANTHROPIC_API_KEY),
        "hf_configured": bool(HF_TOKEN),
    }

# ── AUTH ──────────────────────────────────────────────────────────────────────
@app.post("/api/auth/signup")
async def signup(data: UserSignup):
    if data.email in users_db:
        raise HTTPException(status_code=400, detail="Email already registered")
    user_id = str(uuid.uuid4())
    users_db[data.email] = {
        "id": user_id,
        "name": data.name,
        "email": data.email,
        "password": hash_password(data.password),
        "is_anonymous": data.is_anonymous,
        "created_at": datetime.utcnow().isoformat(),
    }
    token = make_token(user_id)
    return {"token": token, "user": {"id": user_id, "name": data.name, "email": data.email}}

@app.post("/api/auth/login")
async def login(data: UserLogin):
    user = users_db.get(data.email)
    if not user or user["password"] != hash_password(data.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = make_token(user["id"])
    return {"token": token, "user": {"id": user["id"], "name": user["name"], "email": user["email"]}}

# ── ENROLLMENT ────────────────────────────────────────────────────────────────
@app.post("/api/enroll")
async def enroll(data: EnrollRequest):
    key = f"{data.user_id}:{data.course_id}"
    if key in enrollments_db:
        return {"message": "Already enrolled", "enrolled": True}
    enrollments_db[key] = {
        "user_id": data.user_id,
        "course_id": data.course_id,
        "course_title": data.course_title,
        "enrolled_at": datetime.utcnow().isoformat(),
        "progress": 0,
    }
    return {"message": "Successfully enrolled! 💜", "enrolled": True}

@app.get("/api/enrollments/{user_id}")
async def get_enrollments(user_id: str):
    return {"enrollments": [v for v in enrollments_db.values() if v["user_id"] == user_id]}

# ── PROGRESS ──────────────────────────────────────────────────────────────────
@app.post("/api/progress")
async def update_progress(data: ProgressUpdate):
    key = f"{data.user_id}:{data.course_id}"
    if key not in progress_db:
        progress_db[key] = {"completed_lessons": [], "last_updated": ""}
    if data.completed and data.lesson_index not in progress_db[key]["completed_lessons"]:
        progress_db[key]["completed_lessons"].append(data.lesson_index)
    progress_db[key]["last_updated"] = datetime.utcnow().isoformat()
    return {"progress": progress_db[key]}

@app.get("/api/progress/{user_id}/{course_id}")
async def get_progress(user_id: str, course_id: int):
    return progress_db.get(f"{user_id}:{course_id}", {"completed_lessons": [], "last_updated": ""})

# ── AI CHAT ───────────────────────────────────────────────────────────────────
@app.post("/api/ai/chat")
async def ai_chat(data: ChatMessage):
    reply, model_used = await get_ai_reply(data)
    return {"reply": reply, "model": model_used}

# ── CONTACT ───────────────────────────────────────────────────────────────────
@app.post("/api/contact")
async def contact_form(data: ContactForm):
    contacts_db.append({
        "name": data.name,
        "email": data.email,
        "message": data.message,
        "submitted_at": datetime.utcnow().isoformat(),
    })
    print(f"📬 Contact: {data.name} <{data.email}>: {data.message[:80]}")
    return {"success": True, "message": "Message received! We'll reply within 24 hours. 💜"}
