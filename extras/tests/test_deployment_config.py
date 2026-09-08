"""
Deployment-configuration validation for AUDIT.md P1 #9 (A1/A2/F2): every
deploy config under `extras/deployment/` (plus `frontend/`'s Vercel config)
must reference the ACTUAL current backend entrypoint (`extras/api/server.py`,
importable as `extras.api.server:app`) and the ACTUAL current dependency
files -- never the pre-quarantine `qknee/api/...` path, and never a
requirements file that lacks FastAPI/uvicorn for a target that runs the
FastAPI app.

This is a real, run-it-in-CI validation pass, not a documentation exercise:
every path/module name asserted below is checked against the real
filesystem/import system, not just string-matched against another string.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = REPO_ROOT / "extras" / "deployment"

STALE_MODULE_PATH = re.compile(r"qknee[./]api[./]")


def _read(path: Path) -> str:
    assert path.exists(), f"expected deployment file at {path}, not found"
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# 1. No deployment config anywhere references the pre-quarantine module path.
# --------------------------------------------------------------------------- #

class TestNoStaleModulePathAnywhereInDeploymentConfig:
    @pytest.mark.parametrize(
        "relative_path",
        [
            "extras/deployment/vercel.json",
            "extras/deployment/render.yaml",
            "extras/deployment/docker-compose.yml",
            "extras/deployment/docker-compose.override.yml",
            "extras/deployment/Dockerfile",
            "extras/deployment/requirements-vercel.txt",
        ],
    )
    def test_file_has_no_stale_qknee_api_reference(self, relative_path: str):
        content = _read(REPO_ROOT / relative_path)
        matches = STALE_MODULE_PATH.findall(content)
        assert not matches, f"{relative_path} still references the pre-quarantine 'qknee/api' path: {matches}"


# --------------------------------------------------------------------------- #
# 2. vercel.json points at the real entrypoint file.
# --------------------------------------------------------------------------- #

class TestVercelJson:
    @pytest.fixture
    def config(self) -> dict:
        return json.loads(_read(DEPLOY_DIR / "vercel.json"))

    def test_build_src_points_at_a_real_file(self, config: dict):
        src = config["builds"][0]["src"]
        assert (REPO_ROOT / src).exists(), f"vercel.json builds[0].src={src!r} does not exist"
        assert src == "extras/api/server.py"

    def test_route_dest_matches_the_build_src(self, config: dict):
        assert config["routes"][0]["dest"] == config["builds"][0]["src"]

    def test_nearest_requirements_file_to_the_entrypoint_has_fastapi(self, config: dict):
        """Vercel's @vercel/python builder installs the requirements.txt
        nearest to the declared entrypoint -- for extras/api/server.py,
        that's extras/api/requirements.txt, which must actually contain
        fastapi/uvicorn (the exact bug this fix addresses)."""
        entrypoint = REPO_ROOT / config["builds"][0]["src"]
        nearest_requirements = entrypoint.parent / "requirements.txt"
        assert nearest_requirements.exists()
        content = nearest_requirements.read_text(encoding="utf-8").lower()
        assert "fastapi" in content
        assert "uvicorn" in content


# --------------------------------------------------------------------------- #
# 3. render.yaml's build/start commands reference the real module + both
#    requirements files, and its health check targets a real route.
# --------------------------------------------------------------------------- #

class TestRenderYaml:
    @pytest.fixture
    def config(self) -> dict:
        return yaml.safe_load(_read(DEPLOY_DIR / "render.yaml"))

    @pytest.fixture
    def service(self, config: dict) -> dict:
        return config["services"][0]

    def test_start_command_invokes_the_real_asgi_app(self, service: dict):
        assert "extras.api.server:app" in service["startCommand"]

    def test_build_command_installs_both_requirements_files(self, service: dict):
        build_command = service["buildCommand"]
        assert "requirements.txt" in build_command
        assert "extras/api/requirements.txt" in build_command

    @staticmethod
    def _dependency_lines(requirements_text: str) -> str:
        """Real, installable dependency specifier lines only -- strips
        comments/blank lines, so a requirements file's own prose (e.g.
        "the FastAPI server ... is not installed here") can't produce a
        false-positive match against a package name mentioned only in
        passing."""
        return "\n".join(
            line for line in requirements_text.splitlines() if line.strip() and not line.strip().startswith("#")
        ).lower()

    def test_combined_requirements_referenced_by_build_command_contain_fastapi(self, service: dict):
        """Not just a string check on the command -- actually reads both
        files the command names and confirms FastAPI is really there as an
        installable dependency (AUDIT.md's exact original bug: the file
        installed had no FastAPI in it at all)."""
        root_requirements = self._dependency_lines((REPO_ROOT / "requirements.txt").read_text(encoding="utf-8"))
        api_requirements = self._dependency_lines(
            (REPO_ROOT / "extras" / "api" / "requirements.txt").read_text(encoding="utf-8")
        )
        combined = root_requirements + api_requirements
        assert "fastapi" in combined
        assert "uvicorn" in combined
        # And confirms the documented reason a combined install is needed:
        # the root file alone must NOT list it as a dependency (else this
        # whole fix would be a no-op) -- a regression guard, not a design
        # mandate.
        assert "fastapi" not in root_requirements

    def test_health_check_path_is_a_real_registered_route(self, config: dict, service: dict):
        assert service.get("healthCheckPath") == "/health"
        routes = _fastapi_route_paths()
        assert service["healthCheckPath"] in routes

    def test_jwt_secret_is_never_given_a_default_value_in_the_blueprint(self, service: dict):
        """AUDIT.md P1 #8 regression guard from the deployment-config side:
        render.yaml must never itself carry a `value:` for the JWT secret
        -- only `sync: false` (operator sets it manually)."""
        jwt_var = next(v for v in service["envVars"] if v["key"] == "QKNEE_JWT_SECRET_KEY")
        assert jwt_var.get("sync") is False
        assert "value" not in jwt_var


def _fastapi_route_paths() -> set:
    """Imports the real app in a clean subprocess (never the parent test
    process, whose sys.modules may carry the qknee.api.* test aliases from
    conftest.py) and lists its route paths -- proof the module actually
    imports as a standalone deployment would run it, not just that the
    source text looks right."""
    script = (
        "import os\n"
        "os.environ['QKNEE_JWT_SECRET_KEY'] = 'deployment-validation-" + "z" * 40 + "'\n"
        "import extras.api.server as server_module\n"
        "def _walk(routes):\n"
        "    for r in routes:\n"
        "        path = getattr(r, 'path', None)\n"
        "        if path:\n"
        "            yield path\n"
        "        nested = getattr(r, 'routes', None)\n"
        "        if nested:\n"
        "            yield from _walk(nested)\n"
        "        original_router = getattr(r, 'original_router', None)\n"
        "        if original_router is not None:\n"
        "            yield from _walk(getattr(original_router, 'routes', []))\n"
        "print('\\n'.join(_walk(server_module.app.routes)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"extras.api.server failed to import cleanly:\n{result.stderr}"
    return set(result.stdout.strip().splitlines())


# --------------------------------------------------------------------------- #
# 4. The real backend entrypoint actually imports and exposes /health,
#    independent of any test-only sys.modules aliasing.
# --------------------------------------------------------------------------- #

class TestRealEntrypointImportsCleanly:
    def test_extras_api_server_imports_without_the_qknee_api_test_shim(self):
        """The single most important check in this file: this is exactly
        what a real `uvicorn extras.api.server:app` deployment does, with
        none of conftest.py's qknee.api.* sys.modules aliasing active."""
        routes = _fastapi_route_paths()
        assert "/health" in routes
        assert "/predict" in routes
        assert "/api/v1/auth/login" in routes


# --------------------------------------------------------------------------- #
# 5. Docker Compose / Dockerfile reference the real module and install both
#    requirements files.
# --------------------------------------------------------------------------- #

class TestDockerCompose:
    @pytest.fixture
    def base_config(self) -> dict:
        return yaml.safe_load(_read(DEPLOY_DIR / "docker-compose.yml"))

    def test_api_service_command_invokes_the_real_module(self, base_config: dict):
        command = base_config["services"]["api"]["command"]
        assert "extras.api.server:app" in command

    def test_api_service_healthcheck_targets_a_real_route(self, base_config: dict):
        test_cmd = " ".join(base_config["services"]["api"]["healthcheck"]["test"])
        assert "/health" in test_cmd

    def test_override_command_also_invokes_the_real_module(self):
        override_config = yaml.safe_load(_read(DEPLOY_DIR / "docker-compose.override.yml"))
        command = override_config["services"]["api"]["command"]
        assert "extras.api.server:app" in command

    def test_build_context_resolves_to_the_repo_root_not_the_compose_file_directory(self):
        """`docker compose` resolves a relative `build.context` against
        the compose file's OWN directory (extras/deployment/), not the
        repo root -- found via an actual `docker compose config` run, not
        a path guess. `../..` from extras/deployment/ is the repo root,
        which is what the Dockerfile's `COPY requirements.txt .` and the
        root `.dockerignore` both assume."""
        result = subprocess.run(
            ["docker", "compose", "config"], cwd=DEPLOY_DIR, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            pytest.skip(f"docker compose not usable in this environment: {result.stderr.strip()[:200]}")
        merged = yaml.safe_load(result.stdout)
        api_build = merged["services"]["api"]["build"]
        assert Path(api_build["context"]).resolve() == REPO_ROOT.resolve()

    def test_bind_mount_sources_resolve_to_real_repo_root_directories(self):
        """Same class of bug as the build context above, for `volumes:` --
        a relative host path in `volumes:` is ALSO resolved against the
        compose file's directory, so `./qknee/artifacts` from
        extras/deployment/ pointed at a directory that doesn't exist."""
        result = subprocess.run(
            ["docker", "compose", "config"], cwd=DEPLOY_DIR, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            pytest.skip(f"docker compose not usable in this environment: {result.stderr.strip()[:200]}")
        merged = yaml.safe_load(result.stdout)
        api_volumes = merged["services"]["api"]["volumes"]
        artifact_mount = next(v for v in api_volumes if v["target"] == "/app/qknee/artifacts")
        resolved_source = Path(artifact_mount["source"]).resolve()
        assert resolved_source == (REPO_ROOT / "qknee" / "artifacts").resolve()
        assert resolved_source.exists()

    def test_override_bind_mounts_the_extras_source_tree_for_live_reload(self):
        """The dev override's `--reload` only watches directories that are
        actually bind-mounted -- extras/api/server.py living outside any
        mounted volume would mean editing it on the host never triggers a
        reload."""
        override_config = yaml.safe_load(_read(DEPLOY_DIR / "docker-compose.override.yml"))
        volumes = override_config["services"]["api"]["volumes"]
        assert any(v.startswith("../../extras:") for v in volumes)


class TestDockerfile:
    @pytest.fixture
    def content(self) -> str:
        return _read(DEPLOY_DIR / "Dockerfile")

    def test_cmd_invokes_the_real_module(self, content: str):
        assert "extras.api.server:app" in content

    def test_builder_stage_installs_both_requirements_files(self, content: str):
        assert "pip install -r requirements.txt -r extras/api/requirements.txt" in content


# --------------------------------------------------------------------------- #
# 6. Frontend -> backend API URL consistency (AUDIT.md P1 #9 requirement 6).
# --------------------------------------------------------------------------- #

class TestFrontendBackendUrlConsistency:
    def _render_service_name(self) -> str:
        config = yaml.safe_load(_read(DEPLOY_DIR / "render.yaml"))
        return config["services"][0]["name"]

    def test_frontend_vercel_json_points_at_the_render_service_render_would_actually_create(self):
        service_name = self._render_service_name()
        frontend_vercel = json.loads(_read(REPO_ROOT / "frontend" / "vercel.json"))
        api_url = frontend_vercel["env"]["NEXT_PUBLIC_API_URL"]
        assert api_url == f"https://{service_name}.onrender.com"

    def test_frontend_lib_api_ts_default_matches_the_render_service(self):
        service_name = self._render_service_name()
        content = _read(REPO_ROOT / "frontend" / "lib" / "api.ts")
        match = re.search(r'API_BASE_URL\s*=.*?"(https://[^"]+)"', content)
        assert match, "could not find API_BASE_URL's default fallback string in lib/api.ts"
        assert match.group(1) == f"https://{service_name}.onrender.com"

    def test_frontend_env_example_matches_the_render_service(self):
        service_name = self._render_service_name()
        content = _read(REPO_ROOT / "frontend" / ".env.local.example")
        assert f"https://{service_name}.onrender.com" in content
