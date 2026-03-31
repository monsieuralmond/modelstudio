from __future__ import annotations

import asyncio
import json
import uuid
from urllib import request

from .config import Settings
from .models import GpuJob, GpuJobState, SubmitGpuTrainingRequest


class GpuTrainingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._jobs: dict[str, GpuJob] = {}

    def list_jobs(self) -> list[GpuJob]:
        return list(reversed(self._jobs.values()))

    def get_job(self, job_id: str) -> GpuJob | None:
        return self._jobs.get(job_id)

    async def submit_job(self, payload: SubmitGpuTrainingRequest) -> GpuJob:
        job_id = str(uuid.uuid4())
        has_raw = payload.dataset.has_raw_episodes
        initial_message = (
            "GPU 학습 작업을 대기열에 넣었습니다."
            if has_raw
            else "GPU 학습 작업을 대기열에 넣었습니다. 현재는 원본 episode 파일이 아니라 메타데이터 중심으로 제출됩니다."
        )
        job = GpuJob(
            job_id=job_id,
            session_name=payload.session_name,
            state=GpuJobState.QUEUED,
            message=initial_message,
            progress=5,
            dataset=payload.dataset,
            training_config=payload.training_config,
        )
        self._jobs[job_id] = job
        asyncio.create_task(self._run_job(job_id))
        return job

    async def _run_job(self, job_id: str) -> None:
        job = self._jobs[job_id]
        if self.settings.gpu_training_endpoint:
            try:
                submitted = await asyncio.to_thread(self._submit_remote_job, job)
                job.remote_job_id = submitted.get("job_id") or submitted.get("id")
                job.state = GpuJobState.RUNNING
                job.progress = int(submitted.get("progress", 20))
                job.message = submitted.get("message", "외부 GPU 서버에 학습 작업을 보냈습니다.")
                job.result = submitted
                return
            except Exception as exc:
                job.state = GpuJobState.ERROR
                job.progress = 100
                job.message = f"외부 GPU 서버 제출에 실패했습니다: {exc}"
                return

        job.state = GpuJobState.RUNNING
        job.progress = 25
        job.message = "원격 GPU 학습 작업을 준비하고 있습니다."
        await asyncio.sleep(1.2)
        job.progress = 60
        job.message = "GPU 서버에서 데이터셋 요약을 바탕으로 학습 설정을 구성하고 있습니다."
        await asyncio.sleep(1.2)
        job.progress = 100
        job.state = GpuJobState.COMPLETED
        job.message = "GPU 학습 작업이 완료되었습니다. 이제 실제 episode 업로드 파이프라인을 연결하면 원격 학습으로 확장할 수 있습니다."
        job.result = {
            "accuracy_hint": "메타데이터 기반 데모 완료",
            "next_step": "원본 영상/episode 파일 업로드 연결",
        }

    def _submit_remote_job(self, job: GpuJob) -> dict:
        payload = json.dumps(
            {
                "session_name": job.session_name,
                "dataset": job.dataset.model_dump(),
                "training_config": job.training_config,
            }
        ).encode("utf-8")
        req = request.Request(
            self.settings.gpu_training_endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
