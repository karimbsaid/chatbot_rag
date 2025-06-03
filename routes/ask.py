from fastapi import APIRouter, HTTPException
from models.schemas import Question
from services.retrieval import retrieve_optimized
from fastapi.responses import StreamingResponse
from services.llm import chain

router = APIRouter()

# @router.post("")
# async def ask(question: Question):
#     try:
#         docs = retrieve_optimized(question.question)
#         answer = get_llm_answer(question.question, "\n".join(docs))
#         return {"response": answer}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@router.post("/stream", response_class=StreamingResponse)
async def ask_stream(question: Question):
    try:
        docs = retrieve_optimized(question.question)
        documents = "\n".join(docs)
        print("docs", documents)


        async def stream():
            async for event in chain.astream_events({
                "question": question.question,
                "documents": documents
            }):
                print(event)
                if event["event"] == "on_parser_stream":
                    chunk = event["data"]["chunk"]
                    yield chunk


        return StreamingResponse(stream(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
