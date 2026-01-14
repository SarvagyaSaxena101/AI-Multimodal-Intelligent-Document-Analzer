import io
from PIL import Image
import easyocr
import numpy as np # Added for image conversion

# Initialize EasyOCR reader globally
# This will download models the first time it's run, if not present.
# You can specify languages, e.g., ['en', 'fr']
reader = easyocr.Reader(['en'])

def ocr_from_image(image_bytes: bytes) -> str:
    # Open the image using PIL
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convert PIL Image to NumPy array (required by EasyOCR)
    np_img = np.array(img)
    
    # Perform OCR
    # readtext returns a list of (bbox, text, confidence)
    results = reader.readtext(np_img)
    
    # Extract just the text from the results
    extracted_text = [text for (bbox, text, prob) in results]
    
    return "\n".join(extracted_text)