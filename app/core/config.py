from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass
class Settings:
    app_name: str
    environment: str
    host: str
    port: int
    logging_level: str
    logging_json: bool
    upload_dir: Path
    db_user: str
    db_password: str
    db_host: str
    db_port: int
    db_name: str
    db_pool_size: int
    db_max_overflow: int

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


def _load_settings() -> Settings:
    config_path = Path("config/settings.toml")
    if not config_path.exists():
        raise FileNotFoundError("Missing configuration file at config/settings.toml")

    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    app_cfg = raw.get("app", {})
    log_cfg = raw.get("logging", {})
    storage_cfg = raw.get("storage", {})
    db_cfg = raw.get("database", {})

    upload_dir = Path(storage_cfg.get("upload_dir", "uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        app_name=app_cfg.get("name", "OCR Backend"),
        environment=app_cfg.get("environment", "development"),
        host=app_cfg.get("host", "0.0.0.0"),
        port=int(app_cfg.get("port", 8000)),
        logging_level=log_cfg.get("level", "INFO"),
        logging_json=bool(log_cfg.get("json", True)),
        upload_dir=upload_dir,
        db_user=db_cfg.get("user", "postgres"),
        db_password=db_cfg.get("password", "postgres"),
        db_host=db_cfg.get("host", "localhost"),
        db_port=int(db_cfg.get("port", 5432)),
        db_name=db_cfg.get("name", "ocr_db"),
        db_pool_size=int(db_cfg.get("pool_size", 10)),
        db_max_overflow=int(db_cfg.get("max_overflow", 10)),
    )


settings = _load_settings()


