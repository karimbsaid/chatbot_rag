from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from models.schemas import Section
from utils.text_splitter import split_text
from services.retrieval import store_documents
from utils.extractor import extract_sections_by_font

from fastapi.responses import JSONResponse

router = APIRouter()

# @router.post("")
# async def store(sections: list[Section]):
#     try:
#         split_sections = []
#         for section in sections:
#             split_sections.extend(split_text(section.content, section.title))
#         store_documents(split_sections)
#         return {"message": f"{len(split_sections)} sections stored"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@router.post("/from-file")
async def store_from_file(
    file: UploadFile = File(...),
    heading_font_threshold: float = Form(12.0),
    debut_document: int = Form(0),
    space_threshold: float = Form(1.5)
):
    print(file)
    print(heading_font_threshold)
    print(debut_document)
    print(space_threshold)
    try:
        pdf_path = f"temp_{file.filename}"
        with open(pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        sections = extract_sections_by_font(pdf_path, heading_font_threshold, space_threshold, debut_document)
        split_sections = []
        
        for section in sections:
            # Use dictionary key access instead of attribute access
            split_sections.extend(split_text(section["content"], section["title"]))
            
        store_documents(split_sections)
        print(split_sections)
        return {"message": f"{len(split_sections)} sections stored"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))