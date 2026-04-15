import os
import json
import re
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pypdf import PdfReader
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL_PRIORITY = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/", response_class=HTMLResponse)
async def root():
    for candidate in [
        os.path.join(BASE_DIR, "index.html"),
        os.path.join(BASE_DIR, "static", "index.html"),
    ]:
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                return f.read()
    files = os.listdir(BASE_DIR)
    raise HTTPException(status_code=404, detail=f"index.html not found. Files: {files}")


def repair_and_parse_json(raw: str) -> dict:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response")

    depth = 0
    end = -1
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        fragment = text[start:]
        if fragment.count('"') % 2 != 0:
            fragment += '"'
        fragment += "]" * max(0, fragment.count("[") - fragment.count("]"))
        fragment += "}" * max(0, fragment.count("{") - fragment.count("}"))
        try:
            return json.loads(fragment)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not repair JSON: {e}")

    return json.loads(text[start:end + 1])


def build_extraction_prompt(extracted_text: str) -> str:
    return (
        "You are a precise legal document data extractor for Philippine Articles of Incorporation (AOI).\n"
        "Extract ALL information from the document text below.\n"
        "Return ONLY a single valid JSON object — no markdown, no explanation, no preamble.\n"
        "Use EXACTLY the keys listed. Leave a field as empty string \"\" if not found — never omit a key.\n\n"
        "IMPORTANT INSTRUCTIONS FOR AMENDMENT DATES:\n"
        "- Look for phrases like 'As amended on', 'as amended on', '(As amended on ...)'.\n"
        "- Extract the date from the TITLE BLOCK (e.g. 'RIZAL POULTRY FARM CORPORATION (As amended on October 24, 2025)').\n"
        "- Also extract per-article amendment dates if present inline in the text.\n"
        "- All dates must be in YYYY-MM-DD format.\n\n"
        "IMPORTANT INSTRUCTIONS FOR SIGNATORIES:\n"
        "- The 'signatories' array must be extracted from the SIGNATURE BLOCK at the END of the document.\n"
        "- These are the people who physically sign the AOI — they have a printed name, a role/title below, and a TIN.\n"
        "- Do NOT confuse with the directors table. Signatories are only those listed under 'IN WITNESS WHEREOF'.\n"
        "- Example signatory: name='ALEXANDRA S. ANGLIONGTO', role='Director and President', tin='106-265-745'.\n\n"
        "REQUIRED JSON SCHEMA:\n"
        "{\n"
        '  "corporateName": "Full registered name in ALL CAPS",\n'
        '  "documentAmendedDate": "Date the entire AOI was last amended, YYYY-MM-DD or empty",\n'
        '  "article1AmendedDate": "Amendment date for Article I specifically, YYYY-MM-DD or empty",\n'
        '  "article2AmendedDate": "Amendment date for Article II specifically, YYYY-MM-DD or empty",\n'
        '  "article3AmendedDate": "Amendment date for Article III specifically, YYYY-MM-DD or empty",\n'
        '  "article4AmendedDate": "Amendment date for Article IV specifically, YYYY-MM-DD or empty",\n'
        '  "article6AmendedDate": "Amendment date for Article VI specifically, YYYY-MM-DD or empty",\n'
        '  "article7AmendedDate": "Amendment date for Article VII specifically, YYYY-MM-DD or empty",\n'
        '  "primaryPurpose": "Full verbatim primary purpose clause",\n'
        '  "secondaryPurposes": "Secondary purposes if any, else empty string",\n'
        '  "street": "Street number and name of principal office",\n'
        '  "barangay": "Barangay of principal office",\n'
        '  "city": "City or municipality of principal office",\n'
        '  "province": "Province or Metro Manila",\n'
        '  "term": "Corporate term in years as integer string, or 0 for perpetual",\n'
        '  "numberOfDirectors": "Number of directors as integer string",\n'
        '  "acsWords": "Authorized capital stock in WORDS e.g. TWENTY MILLION",\n'
        '  "acsAmount": "Authorized capital stock amount e.g. 20,000,000.00",\n'
        '  "numberOfShares": "Total number of shares e.g. 2,000,000",\n'
        '  "parValue": "Par value per share e.g. 10.00",\n'
        '  "treasurer": "Full name of Treasurer-in-Trust",\n'
        '  "treasurerTIN": "TIN of Treasurer e.g. 106-265-745",\n'
        '  "treasurerAddress": "Residential address of Treasurer",\n'
        '  "treasurerCitizenship": "Citizenship of Treasurer e.g. Filipino",\n'
        '  "secRegistrationNo": "SEC Registration Number e.g. CS201600162941",\n'
        '  "dateOfRegistration": "Date in YYYY-MM-DD format or empty string",\n'
        '  "incorporators": [\n'
        '    {"name": "Full name", "nationality": "e.g. Filipino", "residence": "Full address"}\n'
        '  ],\n'
        '  "directors": [\n'
        '    {"name": "Full name", "nationality": "e.g. Filipino", "residence": "Full address"}\n'
        '  ],\n'
        '  "subscribers": [\n'
        '    {"name": "Full name", "citizenship": "e.g. Filipino", "sharesSubscribed": "number", '
        '"amountSubscribed": "amount", "amountPaid": "amount"}\n'
        '  ],\n'
        '  "signatories": [\n'
        '    {"name": "Full printed name e.g. ALEXANDRA S. ANGLIONGTO", "role": "e.g. Director and President", "tin": "e.g. 106-265-745"}\n'
        '  ]\n'
        "}\n\n"
        "DOCUMENT TEXT:\n"
        f"{extracted_text[:14000]}"
    )


def call_model_with_fallback(prompt: str) -> dict:
    last_error = None
    for model_id in MODEL_PRIORITY:
        try:
            print(f"[extract] Trying: {model_id}")
            response = groq_client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a legal document data extractor. "
                            "Output ONLY raw valid JSON. No markdown. No explanation. "
                            "Every key in the schema must be present."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4096,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            if not raw or not raw.strip():
                raise ValueError("Empty response from model")

            parsed = repair_and_parse_json(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Response is not a JSON object")

            print(f"[extract] Success: {model_id}")
            return parsed

        except Exception as e:
            print(f"[extract] {model_id} failed: {e}")
            last_error = e
            continue

    raise ValueError(f"All models failed. Last error: {last_error}")


@app.post("/api/extract")
async def extract_document(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        pdf_bytes = await file.read()

        if len(pdf_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        extracted_text = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            reader = PdfReader(tmp_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        extracted_text = extracted_text.strip()

        if not extracted_text:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not extract text. The PDF may be scanned/image-based. "
                    "Please use a text-based PDF."
                ),
            )

        print(f"[extract] Extracted {len(extracted_text)} characters from PDF")

        prompt = build_extraction_prompt(extracted_text)
        result = call_model_with_fallback(prompt)

        for key in [
            "corporateName", "documentAmendedDate",
            "article1AmendedDate", "article2AmendedDate", "article3AmendedDate",
            "article4AmendedDate", "article6AmendedDate", "article7AmendedDate",
            "primaryPurpose", "secondaryPurposes",
            "street", "barangay", "city", "province", "term",
            "numberOfDirectors", "acsWords", "acsAmount", "numberOfShares",
            "parValue", "treasurer", "treasurerTIN", "treasurerAddress",
            "treasurerCitizenship", "secRegistrationNo", "dateOfRegistration",
        ]:
            if key not in result:
                result[key] = ""

        for arr_key in ("incorporators", "directors", "subscribers", "signatories"):
            if arr_key not in result or not isinstance(result[arr_key], list):
                result[arr_key] = []

        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"[extract] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


static_dir = os.path.join(BASE_DIR, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
