from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    database_url: str = "postgresql+psycopg://poc:poc@localhost:5433/vidpoc"
    media_dir: str = "./data"

    vlm_provider: str = "fake"  # fake | gemini
    gcp_project: str = ""
    gcp_location: str = "us-central1"
    vertex_model: str = "gemini-2.5-flash"
    vlm_concurrency: int = 4

    # sampling knobs (Step-1 tuning surface)
    baseline_interval_s: float = 5.0
    dense_interval_s: float = 1.5
    t_fast_percentile: float = 75.0
    scan_fps: float = 4.0
    scan_width: int = 320
    max_frame_side: int = 1568
    jpeg_quality: int = 85
    thumb_width: int = 320
    blur_threshold: float = 100.0
    luma_min: float = 15.0
    luma_max: float = 240.0
    phash_max_distance: int = 8
    neighbor_window_s: float = 0.7


settings = Settings()
