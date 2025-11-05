from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.routes.upload import router as upload_router
from app.db.session import init_engine_and_create_tables


def create_app() -> FastAPI:
    configure_logging(level=settings.logging_level, json_output=settings.logging_json)
    logger = get_logger()

    application = FastAPI(title=settings.app_name)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(upload_router)

    @application.on_event("startup")
    def on_startup() -> None:
        logger.info("app.startup", app=settings.app_name, env=settings.environment)
        init_engine_and_create_tables()

    @application.on_event("shutdown")
    def on_shutdown() -> None:
        logger.info("app.shutdown", app=settings.app_name)

    return application


app = create_app()


