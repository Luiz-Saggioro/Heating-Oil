"""
manage_users.py — Local admin CLI for the Energy Intelligence Dashboard.

Run locally (NOT on Streamlit Cloud). Changes update users.json on disk;
commit and push to GitHub to deploy the changes.

Usage:
    python manage_users.py list
    python manage_users.py add
    python manage_users.py deactivate "Login Name"
    python manage_users.py activate "Login Name"
    python manage_users.py reset-password "Login Name"
    python manage_users.py reset-totp "Login Name"
"""

import sys
import json
import os
import hashlib
import secrets
import string
import getpass
import datetime

USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")
PBKDF2_ITERS = 260000
TOTP_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_db():
    if not os.path.exists(USERS_FILE):
        return {"users": []}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_db(db):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)
    print(f"[OK] Saved to {USERS_FILE}")


def hash_password(password: str):
    salt = os.urandom(16)
    key  = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERS)
    return key.hex(), salt.hex()


def new_totp_secret():
    return "".join(secrets.choice(TOTP_ALPHABET) for _ in range(32))


def find_user(db, login: str):
    for i, u in enumerate(db["users"]):
        if u["login"].strip().lower() == login.strip().lower():
            return i, u
    return None, None


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_list(db):
    users = db.get("users", [])
    if not users:
        print("No users found.")
        return
    print(f"\n{'Login':<30} {'Admin':<8} {'Active':<8} {'2FA':<6} {'Created'}")
    print("-" * 68)
    for u in users:
        print(
            f"{u['login']:<30} "
            f"{'YES' if u.get('is_admin') else 'no':<8} "
            f"{'YES' if u.get('is_active') else 'NO':<8} "
            f"{'ON' if u.get('totp_enabled') else 'off':<6} "
            f"{u.get('created_at', '—')}"
        )
    print()


def cmd_add(db):
    print("\n── Add New User ──────────────────────────────────")
    login = input("Login name (e.g. 'John Smith'): ").strip()
    if not login:
        print("[ERR] Login name cannot be empty.")
        return

    _, existing = find_user(db, login)
    if existing:
        print(f"[ERR] User '{login}' already exists.")
        return

    password = getpass.getpass("Password: ")
    confirm  = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("[ERR] Passwords do not match.")
        return
    if len(password) < 8:
        print("[ERR] Password must be at least 8 characters.")
        return

    is_admin = input("Grant admin privileges? (y/N): ").strip().lower() == "y"
    pwd_hash, salt = hash_password(password)
    totp_secret    = new_totp_secret()

    new_user = {
        "login":         login,
        "password_hash": pwd_hash,
        "salt":          salt,
        "is_admin":      is_admin,
        "is_active":     True,
        "totp_secret":   totp_secret,
        "totp_enabled":  False,
        "created_at":    str(datetime.date.today()),
    }
    db["users"].append(new_user)
    save_db(db)
    print(f"[OK] User '{login}' created. They will be prompted to set up 2FA on first login.")
    if is_admin:
        print("[!]  Admin privileges granted.")


def cmd_deactivate(db, login: str):
    idx, user = find_user(db, login)
    if user is None:
        print(f"[ERR] User '{login}' not found.")
        return
    if not user.get("is_active"):
        print(f"[!]  User '{login}' is already inactive.")
        return
    db["users"][idx]["is_active"] = False
    save_db(db)
    print(f"[OK] User '{login}' deactivated. They cannot log in until reactivated.")


def cmd_activate(db, login: str):
    idx, user = find_user(db, login)
    if user is None:
        print(f"[ERR] User '{login}' not found.")
        return
    if user.get("is_active"):
        print(f"[!]  User '{login}' is already active.")
        return
    db["users"][idx]["is_active"] = True
    save_db(db)
    print(f"[OK] User '{login}' reactivated.")


def cmd_reset_password(db, login: str):
    idx, user = find_user(db, login)
    if user is None:
        print(f"[ERR] User '{login}' not found.")
        return
    password = getpass.getpass(f"New password for '{login}': ")
    confirm  = getpass.getpass("Confirm: ")
    if password != confirm:
        print("[ERR] Passwords do not match.")
        return
    if len(password) < 8:
        print("[ERR] Password must be at least 8 characters.")
        return
    pwd_hash, salt = hash_password(password)
    db["users"][idx]["password_hash"] = pwd_hash
    db["users"][idx]["salt"]          = salt
    save_db(db)
    print(f"[OK] Password for '{login}' reset.")


def cmd_reset_totp(db, login: str):
    idx, user = find_user(db, login)
    if user is None:
        print(f"[ERR] User '{login}' not found.")
        return
    confirm = input(f"Reset 2FA for '{login}'? They will re-enroll on next login. (y/N): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return
    db["users"][idx]["totp_secret"]  = new_totp_secret()
    db["users"][idx]["totp_enabled"] = False
    save_db(db)
    print(f"[OK] 2FA reset for '{login}'. They will be prompted to re-enroll on next login.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    db = load_db()
    args = sys.argv[1:]

    if not args or args[0] == "list":
        cmd_list(db)

    elif args[0] == "add":
        cmd_add(db)

    elif args[0] == "deactivate":
        if len(args) < 2:
            print("Usage: python manage_users.py deactivate \"Login Name\"")
            sys.exit(1)
        cmd_deactivate(db, args[1])

    elif args[0] == "activate":
        if len(args) < 2:
            print("Usage: python manage_users.py activate \"Login Name\"")
            sys.exit(1)
        cmd_activate(db, args[1])

    elif args[0] == "reset-password":
        if len(args) < 2:
            print("Usage: python manage_users.py reset-password \"Login Name\"")
            sys.exit(1)
        cmd_reset_password(db, args[1])

    elif args[0] == "reset-totp":
        if len(args) < 2:
            print("Usage: python manage_users.py reset-totp \"Login Name\"")
            sys.exit(1)
        cmd_reset_totp(db, args[1])

    else:
        print(__doc__)
        sys.exit(1)

    print("\nRemember: commit and push users.json to GitHub to deploy changes.")
    print("  git add users.json && git commit -m 'chore: update user credentials' && git push")


if __name__ == "__main__":
    main()
