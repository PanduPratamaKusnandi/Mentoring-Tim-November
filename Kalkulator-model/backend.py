from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, SafetySetting, Part
import base64
from PIL import Image
import io
import os
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#path absolut ke credentials.json
current_dir = os.path.dirname(os.path.abspath(__file__))
credential_path = os.path.join(current_dir, "credentials", "credentials.json")

if os.path.exists(credential_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path
    logger.info(f"Using credentials from {credential_path}")
else:
    logger.error(f"Credentials file not found at {credential_path}")
    raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS not set in environment")

vertexai.init(project="inspired-gear-443512-t4", location="asia-southeast1")


system_instruction = """Anda adalah seorang guru matematika yang interaktif dan sabar. Anda mengajarkan siswa pada tingkat pendidikan berbeda: SD, SMP, atau SMA. Berdasarkan tingkat pendidikan pengguna, Anda harus:

1. **Memberikan soal matematika** sesuai dengan tingkat pendidikan pengguna:
  - SD
  - SMP
  - SMA

2. **Menyesuaikan tingkat kesulitan** soal (mudah, sedang, sulit) berdasarkan pilihan pengguna:
  - Mudah: Soal sederhana untuk pemahaman awal.
  - Sedang: Soal dengan langkah penyelesaian yang lebih kompleks.
  - Sulit: Soal menantang dengan banyak langkah.

3. Semua soal, jawaban, dan penjelasan harus diberikan dalam **bahasa Indonesia**. Pastikan penjelasan disampaikan dengan jelas dan mudah dipahami untuk membantu pengguna memahami langkah-langkah penyelesaian.

Selalu berikan respons yang relevan, jelas, dan mendidik. Pastikan untuk memotivasi pengguna untuk terus belajar."""


model = GenerativeModel(
    "gemini-1.5-flash-002",
    system_instruction=[system_instruction]
)

API_KEY = "my-super-secret-key"  

class QuestionRequest(BaseModel):
    level_pendidikan: str
    kesulitan: str


class QuestionResponse(BaseModel):
    soal: str


class AnswerRequest(BaseModel):
    soal: str
    jawaban_pengguna: str
    image_data: str = None  


class EvaluationResponse(BaseModel):
    umpan_balik: str


def convert_base64_to_image(image_base64):
    if image_base64:
        try:
            image_bytes = base64.b64decode(image_base64)
            return Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            logger.error(f"Error converting image: {e}")
            return None
    return None

@app.post("/generate-question", response_model=QuestionResponse)
def generate_question(
    request: QuestionRequest,
    authorization: str = Header(None)
):
    
    if authorization != f"Bearer {API_KEY}":
        logger.warning("Unauthorized access attempt.")
        raise HTTPException(status_code=403, detail="Forbidden")

    
    prompt = f"Berikan saya soal matematika untuk anak {request.level_pendidikan} dengan tingkat kesulitan {request.kesulitan}. Pastikan soal hanya mencakup matematika."

    try:
        
        response = model.generate_content(
            [prompt],
            generation_config=GenerationConfig(
                max_output_tokens=1024,
                temperature=0.7,
                top_p=0.95,
                response_mime_type="text/plain"
            ),
            safety_settings=[
                SafetySetting(
                    category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=SafetySetting.HarmBlockThreshold.OFF
                ),
                SafetySetting(
                    category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=SafetySetting.HarmBlockThreshold.OFF
                ),
                SafetySetting(
                    category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=SafetySetting.HarmBlockThreshold.OFF
                ),
                SafetySetting(
                    category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=SafetySetting.HarmBlockThreshold.OFF
                ),
            ]
        )

        soal = response.text.strip()
        logger.info(f"Generated question: {soal}")
        return {"soal": soal}
    except Exception as e:
        logger.error(f"Error generating question: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/evaluate-answer", response_model=EvaluationResponse)
def evaluate_answer(
    request: AnswerRequest,
    authorization: str = Header(None)
):
    
    if authorization != f"Bearer {API_KEY}":
        logger.warning("Unauthorized access attempt.")
        raise HTTPException(status_code=403, detail="Forbidden")

    
    evaluasi_prompt = f"Soal: {request.soal}\nJawaban Pengguna (teks): {request.jawaban_pengguna}\n"

    if request.image_data:
        
        image = convert_base64_to_image(request.image_data)
        if image:
            
            image_part = Part.from_data(
                mime_type="image/jpeg",  
                data=base64.b64decode(request.image_data)
            )
            
            evaluasi_prompt += "Jawaban Pengguna juga disertakan dalam bentuk gambar. Tolong interpretasikan angka dalam gambar tersebut sebagai jawaban pengguna dan bandingkan dengan jawaban teks.\n"
        else:
            raise HTTPException(status_code=400, detail="Invalid image data")
    else:
        image_part = None

    evaluasi_prompt += "Evaluasi jawaban tersebut dan berikan umpan balik."

    try:
        if image_part:
            
            evaluasi_response = model.generate_content(
                [evaluasi_prompt, image_part],
                generation_config=GenerationConfig(
                    max_output_tokens=1024,
                    temperature=0.7,
                    top_p=0.95,
                    response_mime_type="text/plain"
                ),
                safety_settings=[
                    SafetySetting(
                        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=SafetySetting.HarmBlockThreshold.OFF
                    ),
                    SafetySetting(
                        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=SafetySetting.HarmBlockThreshold.OFF
                    ),
                    SafetySetting(
                        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=SafetySetting.HarmBlockThreshold.OFF
                    ),
                    SafetySetting(
                        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=SafetySetting.HarmBlockThreshold.OFF
                    ),
                ]
            )
        else:
            
            evaluasi_response = model.generate_content(
                [evaluasi_prompt],
                generation_config=GenerationConfig(
                    max_output_tokens=1024,
                    temperature=0.7,
                    top_p=0.95,
                    response_mime_type="text/plain"
                ),
                safety_settings=[
                    SafetySetting(
                        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=SafetySetting.HarmBlockThreshold.OFF
                    ),
                    SafetySetting(
                        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=SafetySetting.HarmBlockThreshold.OFF
                    ),
                    SafetySetting(
                        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=SafetySetting.HarmBlockThreshold.OFF
                    ),
                    SafetySetting(
                        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=SafetySetting.HarmBlockThreshold.OFF
                    ),
                ]
            )

        umpan_balik = evaluasi_response.text.strip()
        logger.info(f"Feedback: {umpan_balik}")
        return {"umpan_balik": umpan_balik}
    except Exception as e:
        logger.error(f"Error evaluating answer: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
 
    port = int(os.environ.get("PORT", 8080))  
    uvicorn.run(app, host="0.0.0.0", port=port)
