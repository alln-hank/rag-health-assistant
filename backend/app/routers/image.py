from fastapi import APIRouter

from backend.app.schemas import ImageAnalyzeRequest, ImageAnalyzeResponse
from backend.app.services.image_service import analyze_image, is_image_analysis_error


router = APIRouter(prefix="/image", tags=["image"])


@router.post("/analyze", response_model=ImageAnalyzeResponse)
async def analyze(request: ImageAnalyzeRequest):
    description = analyze_image(request.image_path)
    return ImageAnalyzeResponse(
        description=description,
        success=not is_image_analysis_error(description),
    )
