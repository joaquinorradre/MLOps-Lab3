"""
api.py
----------------
API for image processing and prediction using FastAPI.
"""

import io
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse
from mylib import logic

app = FastAPI(
    title="API for Image Processing (MLOps Lab 1)",
    description="API to predict image classes and resize images.",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR.parent / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

OUTPUTS_DIR = BASE_DIR.parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Serves the home.html page."""
    return templates.TemplateResponse(request, "home.html")


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to predict the class of a given image.
    Receives an image file and returns a JSON prediction.
    """
    try:
        image_bytes = await file.read()

        prediction = logic.predict(image_bytes)

        return {
            "filename": file.filename,
            "prediction": prediction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}") from e

@app.post("/resize")
async def resize_endpoint(
    file: UploadFile = File(...),
    width: int = Form(...),
    height: int = Form(...)
):
    """
    Endpoint to resize an image.
    Receives an image file, width, and height.
    Returns the resized image file.
    """
    try:
        image_bytes = await file.read()

        resized_bytes = logic.resize(image_bytes, width, height)

        output_filename = f"resized_{file.filename}"
        output_path = OUTPUTS_DIR / output_filename

        with open(output_path, "wb") as f:
            f.write(resized_bytes)

        return StreamingResponse(io.BytesIO(resized_bytes), media_type="image/png")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resizing image: {e}") from e

@app.post("/grayscale")
async def grayscale_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to convert an image to grayscale.
    Receives an image file and returns the processed image.
    """
    try:
        image_bytes = await file.read()

        grayscale_bytes = logic.convert_to_grayscale(image_bytes)

        output_filename = f"grayscale_{file.filename}"
        output_path = OUTPUTS_DIR / output_filename

        with open(output_path, "wb") as f:
            f.write(grayscale_bytes)

        return StreamingResponse(io.BytesIO(grayscale_bytes), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Error converting image to grayscale: {e}") from e


if __name__ == "__main__":
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True) # pragma: no cover
