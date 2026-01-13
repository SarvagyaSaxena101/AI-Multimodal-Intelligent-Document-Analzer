import io
from PIL import Image
import pytesseract
import os

def ocr_from_image(image_bytes: bytes) -> str:
    if os.getenv("TESSERACT_CMD"):
        pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    text = pytesseract.image_to_string(img)
    return text
