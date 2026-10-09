from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream.resolve()))

    from sregym.conductor.conductor import Conductor
    from sregym.conductor.conductor_api import (
        app,
        request_shutdown,
        run_api,
        set_conductor,
    )
    from sregym.conductor.constants import StartProblemResult

    conductor = Conductor()
    conductor.problem_id = args.problem
    conductor.register_agent()
    conductor.start_k8s_proxy()
    kubeconfig = args.workspace / ".environment" / "kubeconfig"
    kubeconfig.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(conductor.get_agent_kubeconfig_path(), kubeconfig)
    result = asyncio.run(conductor.start_problem())
    if result != StartProblemResult.SUCCESS:
        raise RuntimeError(f"SREGym problem did not start: {result}")
    set_conductor(conductor)

    async def results():
        return dict(conductor.results)

    async def shutdown():
        if conductor.submission_state()[0] != "done":
            conductor.close_submissions()
            conductor.finish_problem_in_background()
            try:
                await conductor.wait_for_submission_work(timeout=300)
            except TimeoutError:
                conductor.abandon_submission_work()
        loop = asyncio.get_running_loop()
        conductor.stop_k8s_proxy()
        loop.call_later(0.1, request_shutdown)
        return {"status": "closing"}

    app.add_api_route("/results", results, methods=["GET"])
    app.add_api_route("/shutdown", shutdown, methods=["POST"])
    run_api(conductor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
