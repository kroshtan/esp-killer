import logging

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from server import __version__
from server.db.engine import make_engine
from server.db.repository import Repository
from server.ingest import AppState, router
from server.orgconfig import ConfigStore
from server.ratelimit import TokenBucketLimiter
from server.settings import ServerSettings

logger = logging.getLogger(__name__)


def create_app(settings: ServerSettings | None = None, repo: Repository | None = None) -> FastAPI:
    """
    Build the API application.

    :param settings: settings; read from the environment if omitted
    :param repo: repository; built from ``settings.database_url`` if omitted
    :return: the app
    """
    settings = settings or ServerSettings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = FastAPI(title="espk", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.espk = AppState(
        settings=settings,
        config=ConfigStore(settings.config_path),
        repo=repo or Repository(make_engine(settings.database_url)),
        limiter=TokenBucketLimiter(settings.rate_limit_per_s, settings.rate_limit_burst),
    )
    app.include_router(router)

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        state: AppState = app.state.espk
        try:
            await run_in_threadpool(state.repo.ping)
        except Exception:
            logger.exception("health check: database unavailable")
            return JSONResponse({"status": "error", "database": "unavailable"}, status_code=503)
        return JSONResponse({"status": "ok", "version": __version__})

    return app
