from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.ask import router as ask_router
from routes.store import router as store_router

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(ask_router, prefix="/ask")
app.include_router(store_router, prefix="/store")
