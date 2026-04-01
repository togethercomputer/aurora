"""HTTP server Ray actor that receives training callbacks from external sglang servers.

Receives mooncake metadata via HTTP POST and pushes samples into the
AsyncTrainingController's sample_pool for training.
"""

import logging
import threading

import ray
import torch

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class TrainingExternalServer:
    """Ray actor that runs an HTTP server to receive training samples from external sglang.

    Receives POST /push_sample with mooncake metadata and pushes InferenceOutput
    samples into the training controller's sample pool.
    """

    def __init__(self, args, controller):
        self.controller = controller
        self.host = getattr(args, "online_serving_host", "0.0.0.0")
        self.port = getattr(args, "online_serving_port", 18080)
        dtype_str = getattr(args, "online_serving_hidden_states_dtype", "bfloat16")
        self.hidden_states_dtype = getattr(torch, dtype_str)
        self._server_thread = None
        self._server = None

    def run(self):
        """Start the HTTP server in a background thread."""
        from fastapi import FastAPI
        from pydantic import BaseModel

        app = FastAPI()
        server_self = self

        class PushSampleRequest(BaseModel):
            data_id: str
            mooncake_key: str
            tensor_shapes: dict
            prompt_tokens: int
            completion_tokens: int = 0

        @app.post("/push_sample")
        async def push_sample(req: PushSampleRequest):
            from aurora.data.utils import serialize_packed_loss_mask
            from aurora.utils.types import InferenceOutput

            # Convert shape lists back to tuples
            tensor_shapes = {k: tuple(v) for k, v in req.tensor_shapes.items()}

            # Build loss mask: [prompt_len, completion_len - 1]
            # completion_tokens - 1 because the last generated token has no hidden state
            if req.completion_tokens > 0:
                packed_loss_mask = serialize_packed_loss_mask(
                    [req.prompt_tokens, req.completion_tokens - 1]
                )
            else:
                packed_loss_mask = None

            sample = InferenceOutput(
                data_id=req.data_id,
                mooncake_key=req.mooncake_key,
                tensor_shapes=tensor_shapes,
                tensor_dtypes={
                    "hidden_states": server_self.hidden_states_dtype,
                    "last_hidden_states": server_self.hidden_states_dtype,
                },
                packed_loss_mask=packed_loss_mask,
            )

            pool_size = ray.get(server_self.controller.push_sglang_sample.remote(sample))
            logger.info(
                f"Received sample: data_id={req.data_id}, "
                f"key={req.mooncake_key}, pool_size={pool_size}"
            )
            return {"status": "ok", "pool_size": pool_size}

        @app.get("/health")
        async def health():
            pool_size = ray.get(server_self.controller.get_pool_size.remote())
            return {"status": "ok", "pool_size": pool_size}

        import uvicorn

        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)

        def _run_server():
            self._server.run()

        self._server_thread = threading.Thread(target=_run_server, daemon=True)
        self._server_thread.start()
        logger.info(f"TrainingExternalServer started at http://{self.host}:{self.port}")

    def stop(self):
        """Graceful shutdown."""
        if self._server is not None:
            self._server.should_exit = True
            if self._server_thread is not None:
                self._server_thread.join(timeout=5)
        logger.info("TrainingExternalServer stopped")
