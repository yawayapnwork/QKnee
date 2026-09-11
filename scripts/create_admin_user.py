"""
One-time admin bootstrap: provisions the project owner's OWN first real
account directly in the user store, bypassing `POST /api/v1/auth/register`'s
public registration flow entirely.

Why this exists: `extras/api/auth.py`'s `User` store ships with zero
pre-seeded accounts (see that module's docstring), and self-service
registration with `role="radiologist"` is deliberately gated behind
`$QKNEE_RADIOLOGIST_INVITE_CODE` (`UserRepository.create_user`) -- without
that env var set, EVERY radiologist registration request is silently
downgraded to `role="researcher"` (read-only, no inference access). That's
correct, intentional lockdown, not a bug -- but it also means there was no
path at all for the project owner to get their own working inference-
capable account without either (a) setting the invite-code env var AND
remembering to unset it again, or (b) reaching directly into the database.
This script is the deliberate, auditable alternative to both.

Safety model (read before running):
    - Refuses to run at all unless `$ADMIN_SEED_TOKEN` is set in this
      process's environment AND the `--token` argument matches it exactly.
      Neither the mere presence of the env var nor a plausible-looking
      `--token` alone is enough -- both are required, mirroring the
      invite-code pattern's "prove you meant it" design.
    - Refuses to run if ANY account already exists in the store (see
      `UserRepository.count_users`) -- this is a ONE-TIME bootstrap for the
      very first account, not a general-purpose admin-account factory.
      Once you have your first radiologist account, use the ordinary
      invite-code-gated `/register` flow for any additional accounts (set
      `$QKNEE_RADIOLOGIST_INVITE_CODE`, register with a matching
      `invite_code`) -- see extras/README.md's "Admin bootstrap" section.
    - Calls `UserRepository.create_user_unchecked` -- NOT `create_user` --
      so this script's own token check is the only gate; the invite-code
      logic that protects the public `/register` endpoint is completely
      untouched and unaffected by this script's existence.
    - Never reachable over HTTP: this is a CLI script, not a route. Nothing
      in `extras/api/server.py` imports or calls it.
    - The password is read via a hidden prompt (getpass) by default, never
      taken as a plain CLI argument, so it never lands in shell history or
      process-listing output. `--password` exists only for scripted/CI use
      and logs a warning when used.

Usage (see extras/README.md's "Admin bootstrap" section for the full,
copy-pasteable walkthrough):

    export ADMIN_SEED_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
    python scripts/create_admin_user.py \\
        --email you@example.com --full-name "Your Name" --token "$ADMIN_SEED_TOKEN"

Requires the same environment `extras/api/server.py` itself needs to boot
(a valid `$QKNEE_JWT_SECRET_KEY`/`$SECRET_KEY` and, if `$DATABASE_URL` isn't
set, the same local SQLite file the API would use) -- this script imports
`extras.api.auth` directly rather than reimplementing password hashing or
the user table schema.
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from extras.api.auth import ROLES, UserAlreadyExistsError, UserRepository  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--email", required=True, help="Email address for the new account.")
    parser.add_argument("--full-name", required=True, help="Display name for the new account.")
    parser.add_argument(
        "--role", default="radiologist", choices=ROLES,
        help="Role to grant (default: 'radiologist' -- the only role with inference access; "
             "see extras.api.auth.INFERENCE_ROLES).",
    )
    parser.add_argument(
        "--token", required=True,
        help="Must exactly match $ADMIN_SEED_TOKEN. Both this argument and the env var are "
             "required -- neither alone is sufficient.",
    )
    parser.add_argument(
        "--password", default=None,
        help="Account password. If omitted (recommended), you'll be prompted for it via a hidden "
             "input instead -- passing it here leaves it in your shell history and process list.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    seed_token = os.getenv("ADMIN_SEED_TOKEN")
    if not seed_token:
        raise SystemExit(
            "$ADMIN_SEED_TOKEN is not set in this environment -- refusing to run. "
            "See extras/README.md's 'Admin bootstrap' section."
        )
    if args.token != seed_token:
        raise SystemExit("--token does not match $ADMIN_SEED_TOKEN -- refusing to run.")

    repo = UserRepository()

    existing_count = repo.count_users()
    if existing_count > 0:
        raise SystemExit(
            f"Refusing to run: {existing_count} account(s) already exist in the store. "
            "This script only provisions the very first account. For additional accounts, "
            "use the invite-code-gated POST /api/v1/auth/register flow instead "
            "(set $QKNEE_RADIOLOGIST_INVITE_CODE and register with a matching invite_code) -- "
            "see extras/README.md's 'Admin bootstrap' section."
        )

    if repo.get_by_email(args.email) is not None:
        raise SystemExit(f"An account already exists for '{args.email}'.")

    password = args.password
    if password:
        print("WARNING: --password was passed on the command line; prefer the hidden prompt "
              "(omit --password) so it doesn't land in shell history.", file=sys.stderr)
    else:
        password = getpass.getpass(f"Password for {args.email}: ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            raise SystemExit("Passwords did not match.")

    try:
        user = repo.create_user_unchecked(
            email=args.email, password=password, full_name=args.full_name, role=args.role,
        )
    except UserAlreadyExistsError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Created account: {user.email} (role={user.role}, id={user.id})")
    print("Log in via POST /api/v1/auth/login to obtain a bearer token. "
          "Consider unsetting $ADMIN_SEED_TOKEN now that bootstrap is complete.")


if __name__ == "__main__":
    main()
