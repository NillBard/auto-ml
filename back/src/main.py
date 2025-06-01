import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from settings import settings
from routers import router
from db.initdb import initdb

app = FastAPI(debug=settings.SERVER_TEST)
app.include_router(router)
app.mount("/api/static", StaticFiles(directory="./"), name="streams")

origins = ['*']

app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )
initdb()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.SERVER_ADDR,
        port=settings.SERVER_PORT,
        reload=settings.SERVER_TEST,
        log_level="debug" if settings.SERVER_TEST else "info",
    )
