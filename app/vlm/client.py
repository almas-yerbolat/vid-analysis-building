import logging
from dataclasses import dataclass
from typing import Protocol

from app.config import settings
from app.vlm.prompt import SYSTEM_PROMPT, batch_user_text
from app.vlm.schema import BatchResponse, Box, EquipmentCount, FrameAnalysis, VlmFinding

logger = logging.getLogger(__name__)


@dataclass
class VlmResult:
    parsed: BatchResponse
    raw_text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0


class VlmClient(Protocol):
    def analyze_batch(self, images: list[bytes], ts_ms: list[int],
                      low_quality: list[bool], project_name: str) -> VlmResult: ...
    def summarize(self, prompt: str) -> str: ...


class FakeVlmClient:
    """Deterministic stand-in: dev/tests without GCP credentials."""

    def analyze_batch(self, images, ts_ms, low_quality, project_name) -> VlmResult:
        frames = []
        for i, ts in enumerate(ts_ms):
            findings = []
            if i == 0:
                findings = [VlmFinding(
                    category="тб_от", subtype="отсутствие_каски", severity="high",
                    comment="Рабочий без каски в центре кадра.", confidence=0.8,
                    boxes=[Box(label="рабочий без каски", box_2d=[400, 400, 600, 600])],
                )]
            frames.append(FrameAnalysis(
                ts_ms=ts, stage="каркас", stage_confidence=0.9,
                activity="Монтаж опалубки",
                equipment=[EquipmentCount(type="башенный_кран", count=1)],
                findings=findings,
            ))
        parsed = BatchResponse(frames=frames)
        return VlmResult(parsed=parsed, raw_text=parsed.model_dump_json(), model="fake")

    def summarize(self, prompt: str) -> str:
        return ("На объекте ведутся работы стадии «каркас». Зафиксированы нарушения ТБ. "
                "Рекомендуется усилить контроль применения СИЗ.")


class GeminiVlmClient:
    def __init__(self):
        from google import genai
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_file(
            settings.gcp_credentials_file,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        self._client = genai.Client(
            vertexai=True,
            credentials=credentials,
            project=settings.gcp_project or credentials.project_id,
            location=settings.gcp_location,
        )
        self._model = settings.vertex_model

    def analyze_batch(self, images, ts_ms, low_quality, project_name) -> VlmResult:
        from google.genai import types
        contents = [batch_user_text(ts_ms, low_quality, project_name)]
        for ts, img in zip(ts_ms, images):
            contents.append(f"Кадр ts_ms={ts}:")
            contents.append(types.Part.from_bytes(data=img, mime_type="image/jpeg"))
        resp = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=BatchResponse,
                temperature=0.2,
            ),
        )
        usage = resp.usage_metadata
        return VlmResult(
            # resp.parsed is the SDK's schema-validated object; resp.text is kept
            # verbatim for analyses.raw_response (spec §5.1: debug without re-paying).
            parsed=BatchResponse.model_validate(resp.parsed),
            raw_text=resp.text or "",
            model=self._model,
            tokens_in=(usage.prompt_token_count or 0) if usage else 0,
            tokens_out=(usage.candidates_token_count or 0) if usage else 0,
        )

    def summarize(self, prompt: str) -> str:
        resp = self._client.models.generate_content(model=self._model, contents=prompt)
        return resp.text or ""


def get_client() -> VlmClient:
    if settings.vlm_provider == "gemini":
        return GeminiVlmClient()
    return FakeVlmClient()
