"""Paper library endpoints; asset URLs only address registered evidence."""

from fastapi import APIRouter, File, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response

from app.core.errors import AppError
from app.schemas.models import DocumentTaskResponse
from app.schemas.papers import (
    Paper,
    PaperEvidence,
    PaperReadingRequest,
    PaperReadingResponse,
    PaperReadingHistory,
    PaperReadingHistoryItem,
)
from app.services.paper_library import PaperLibrary
from app.services.paper_reading import image_bytes

router = APIRouter(prefix="/papers", tags=["论文阅读"])


def library_for(request: Request) -> PaperLibrary:
    service = request.app.state.service
    service._require_ready()
    assert service.papers is not None
    return service.papers


@router.post("", response_model=DocumentTaskResponse, status_code=202)
async def upload_paper(
    request: Request, file: UploadFile = File(...)
) -> DocumentTaskResponse:
    return await request.app.state.service.submit_document(file, is_paper=True)


@router.get("", response_model=list[Paper])
async def list_papers(request: Request) -> list[Paper]:
    return library_for(request).list()


@router.post("/read", response_model=PaperReadingResponse)
async def read(payload: PaperReadingRequest, request: Request) -> PaperReadingResponse:
    return await request.app.state.service.read_paper(payload)


@router.get("/{paper_id}", response_model=Paper)
async def get_paper(paper_id: str, request: Request) -> Paper:
    return library_for(request).get(paper_id)


@router.get("/{paper_id}/content", response_model=list[PaperEvidence])
async def content(
    paper_id: str,
    request: Request,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[PaperEvidence]:
    library = library_for(request)
    return [
        library.public(row) for row in library.blocks(paper_id)[offset : offset + limit]
    ]


@router.get("/{paper_id}/readings", response_model=list[PaperReadingHistoryItem])
async def readings(
    paper_id: str,
    request: Request,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[PaperReadingHistoryItem]:
    return library_for(request).readings(paper_id, offset, limit)


@router.get("/{paper_id}/readings/{history_id}", response_model=PaperReadingHistory)
async def reading(
    paper_id: str, history_id: str, request: Request
) -> PaperReadingHistory:
    return library_for(request).reading(paper_id, history_id)


@router.get("/{paper_id}/assets/{evidence_id}")
async def asset(paper_id: str, evidence_id: str, request: Request) -> Response:
    data = image_bytes(library_for(request).asset(paper_id, evidence_id))
    if data is None:
        raise AppError("ASSET_UNAVAILABLE", "图片无法预览", 404)
    return Response(data, media_type="image/jpeg")


@router.get("/{paper_id}/original")
async def original(paper_id: str, request: Request) -> FileResponse:
    library = library_for(request)
    paper = library.get(paper_id)
    return FileResponse(library.original(paper_id), filename=paper.file_name)
