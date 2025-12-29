import sys

print(f"Python: {sys.version}")

print("Importing numpy...")
try:
    import numpy

    print("OK")
except Exception as e:
    print(f"FAIL: {e}")

print("Importing pandas...")
try:
    import pandas

    print("OK")
except Exception as e:
    print(f"FAIL: {e}")

print("Importing duckdb...")
try:
    import duckdb

    print("OK")
except Exception as e:
    print(f"FAIL: {e}")

print("Importing polars...")
try:
    import polars

    print("OK")
except Exception as e:
    print(f"FAIL: {e}")

print("Importing reflex...")
try:
    import reflex

    print("OK")
except Exception as e:
    print(f"FAIL: {e}")
