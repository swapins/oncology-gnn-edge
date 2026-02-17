#!/usr/bin/env python3
"""Minimal import test to isolate torch DLL issue."""

print("Step 1: Import torch directly...")
try:
    import torch
    print(f"  ✓ torch {torch.__version__}")
except Exception as e:
    print(f"  ✗ torch: {e}")
    exit(1)

print("Step 2: Import gnn_edge.config...")
try:
    from gnn_edge.config import GNNConfig
    print(f"  ✓ GNNConfig imported")
except Exception as e:
    print(f"  ✗ gnn_edge.config: {e}")
    exit(1)

print("Step 3: Create config...")
try:
    config = GNNConfig(input_dim=64, hidden_dim=32)
    print(f"  ✓ Config created (device: {config.resolve_device()})")
except Exception as e:
    print(f"  ✗ Config creation: {e}")
    exit(1)

print("Step 4: Import GNNInference...")
try:
    from gnn_edge.inference import GNNInference
    print(f"  ✓ GNNInference imported")
except Exception as e:
    print(f"  ✗ GNNInference: {e}")
    exit(1)

print("Step 5: Create engine...")
try:
    engine = GNNInference(config)
    print(f"  ✓ Engine created (device: {engine.device})")
except Exception as e:
    print(f"  ✗ Engine creation: {e}")
    exit(1)

print("\n✓ All imports successful!")
