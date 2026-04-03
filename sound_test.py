"""
sound_test.py  —  DrowseGuard Alert System Diagnostic
Run with:  python sound_test.py
Each test prints PASS / FAIL so you know exactly which layer is broken.
"""
import sys, time, threading

print("=" * 60)
print("  DrowseGuard Sound Alert Diagnostic")
print("=" * 60)

# ── Test 1: winsound import ─────────────────────────────────────
print("\n[TEST 1] Importing winsound...")
try:
    import winsound
    print("  ✅ PASS — winsound imported successfully")
except ImportError as e:
    print(f"  ❌ FAIL — cannot import winsound: {e}")
    print("  → This is a non-Windows platform or winsound is broken.")
    sys.exit(1)

# ── Test 2: Direct beep from main thread ───────────────────────
print("\n[TEST 2] Direct winsound.Beep(1000, 600) from MAIN thread...")
print("  >>> You should hear a 1-second beep NOW <<<")
try:
    winsound.Beep(1000, 600)
    print("  ✅ PASS — Beep call returned without error")
except Exception as e:
    print(f"  ❌ FAIL — Beep raised: {e}")

# ── Test 3: Beep from a daemon thread ──────────────────────────
print("\n[TEST 3] winsound.Beep(1400, 600) from a DAEMON THREAD...")
print("  >>> You should hear a different beep NOW <<<")

thread_result = {"ok": False, "err": None}

def _thread_beep():
    try:
        winsound.Beep(1400, 600)
        thread_result["ok"] = True
    except Exception as e:
        thread_result["err"] = str(e)

t = threading.Thread(target=_thread_beep, daemon=True)
t.start()
t.join(timeout=3.0)

if thread_result["ok"]:
    print("  ✅ PASS — Beep from thread worked")
elif thread_result["err"]:
    print(f"  ❌ FAIL — Beep from thread raised: {thread_result['err']}")
else:
    print("  ❌ FAIL — Thread timed out (possible deadlock or audio device hang)")

# ── Test 4: Continuous loop for 3 seconds (simulates alert worker) ─
print("\n[TEST 4] Continuous alert loop for 3 seconds (sleeping pattern)...")
print("  >>> You should hear rapid beeps for 3 seconds <<<")

stop_event = threading.Event()

def _continuous_worker():
    cycle = 0
    while not stop_event.is_set():
        cycle += 1
        print(f"    [worker] cycle {cycle}")
        try:
            winsound.Beep(1100, 150)
            if stop_event.is_set(): break
            time.sleep(0.05)
            winsound.Beep(1400, 150)
            if stop_event.is_set(): break
            time.sleep(0.05)
            winsound.Beep(1100, 150)
            if stop_event.is_set(): break
            time.sleep(0.2)
        except Exception as e:
            print(f"    [worker] ERROR: {e}")
            break
    print("    [worker] exited cleanly")

t2 = threading.Thread(target=_continuous_worker, daemon=True)
t2.start()
time.sleep(3.0)
stop_event.set()
t2.join(timeout=2.0)
print("  ✅ Continuous loop test complete")

# ── Test 5: Drowsy pattern for 3 seconds ──────────────────────
print("\n[TEST 5] Drowsy pattern for 3 seconds (slow double-beep)...")
print("  >>> You should hear slow double-beeps for 3 seconds <<<")

stop_event2 = threading.Event()

def _drowsy_worker():
    while not stop_event2.is_set():
        try:
            winsound.Beep(700, 250)
            if stop_event2.is_set(): break
            time.sleep(0.1)
            winsound.Beep(900, 250)
            if stop_event2.is_set(): break
            time.sleep(0.6)
        except Exception as e:
            print(f"    [drowsy worker] ERROR: {e}")
            break

t3 = threading.Thread(target=_drowsy_worker, daemon=True)
t3.start()
time.sleep(3.0)
stop_event2.set()
t3.join(timeout=2.0)
print("  ✅ Drowsy pattern test complete")

print("\n" + "=" * 60)
print("  All tests complete.")
print("  If all tests show ✅ PASS but app.py makes no sound,")
print("  the bug is in the trigger logic — check terminal output")
print("  in app.py for [AlertEngine] log lines.")
print("=" * 60)
