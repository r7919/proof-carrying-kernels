import importlib.util

# GPU tests need the `gpu` extra; skip collecting them entirely without it.
collect_ignore_glob = [] if importlib.util.find_spec("torch") else ["test_*.py"]
