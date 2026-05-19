"""
Smoke test — fires a set of known spam and ham messages at the running
inference server and asserts the predictions are correct.

Usage:
    python src/smoke_test.py
    python src/smoke_test.py --url http://localhost:8080
"""

import argparse
import sys

import requests

SPAM_MESSAGES = [
    "Congratulations! You've won a FREE iPhone 15. Click here to claim: http://prize-claim.xyz",
    "URGENT: Your account has been suspended. Verify now at http://secure-bank-login.xyz/verify",
    "You have been selected for a £1000 Tesco gift card. Call 07543210987 to claim.",
    "Free entry in our weekly competition! Text WIN to 80085. T&Cs apply.",
    "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575.",
]

HAM_MESSAGES = [
    "Hey, are we still on for lunch tomorrow? Let me know.",
    "Can you pick up some milk on your way home?",
    "The meeting has been moved to 3pm. See you there.",
    "Happy birthday! Hope you have a wonderful day.",
    "I'll be there in 10 minutes, just stuck in traffic.",
]

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"


def run(base_url: str, api_key: str | None = None) -> bool:
    print(f"Target: {base_url}\n")

    headers = {"X-API-Key": api_key} if api_key else {}

    # Liveness check
    try:
        r = requests.get(f"{base_url}/ping", timeout=10)
        r.raise_for_status()
        print(f"  /ping{PASS}\n")
    except Exception as e:
        print(f"  /ping{FAIL} — server not reachable: {e}")
        return False

    failures = 0

    def check(text: str, expected: str) -> None:
        nonlocal failures
        r = requests.post(f"{base_url}/predict", json={"text": text}, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        prediction = data["prediction"]
        confidence = data["confidence"]
        ok = prediction == expected
        status = PASS if ok else FAIL
        label = f"[{expected.upper():8}]"
        snippet = text[:60] + ("…" if len(text) > 60 else "")
        print(f"  {status} {label} conf={confidence:.2f}  \"{snippet}\"")
        if not ok:
            print(f"           ^ expected={expected}, got={prediction}")
            failures += 1

    print("── Spam messages ───────────────────────────────────────────")
    for msg in SPAM_MESSAGES:
        check(msg, "spam")

    print()
    print("── Ham messages ────────────────────────────────────────────")
    for msg in HAM_MESSAGES:
        check(msg, "not-spam")

    print()
    total = len(SPAM_MESSAGES) + len(HAM_MESSAGES)
    passed = total - failures
    print(f"Result: {passed}/{total} passed", end="  ")
    if failures == 0:
        print("\033[92mall good\033[0m")
    else:
        print(f"\033[91m{failures} failure(s)\033[0m")

    return failures == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the spam classifier API")
    parser.add_argument("--url", default="http://localhost:8080", help="Base URL of the inference server")
    parser.add_argument("--api-key", default=None, help="X-API-Key header value")
    args = parser.parse_args()

    success = run(args.url.rstrip("/"), api_key=args.api_key)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
