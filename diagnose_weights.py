#!/usr/bin/env python3
"""Diagnose model weight loading issues."""

import os
import sys
import torch

sys.path.insert(0, 'pretrained_model')

def check_weights():
    hf_model_dir = os.environ.get('HF_MODEL_DIR', 'pretrained_model')

    # 1. Check weight file
    weights_path = os.path.join(hf_model_dir, 'pytorch_model.bin')
    print(f"=== Checking {weights_path} ===")

    state_dict = torch.load(weights_path, map_location='cpu')
    print(f"Total keys: {len(state_dict)}")

    # 2. Check key prefixes
    prefixes = set()
    for k in state_dict.keys():
        prefix = k.split('.')[0]
        prefixes.add(prefix)
    print(f"Key prefixes: {prefixes}")

    # 3. Check planner weights
    planner_keys = [k for k in state_dict.keys() if 'planner' in k.lower()]
    print(f"\nPlanner keys ({len(planner_keys)}):")
    for k in planner_keys[:10]:
        v = state_dict[k]
        print(f"  {k}: shape={v.shape}, mean={v.float().mean():.6f}, std={v.float().std():.6f}")

    # 4. Check alpha_task buffer
    alpha_keys = [k for k in state_dict.keys() if 'alpha' in k.lower()]
    print(f"\nAlpha keys: {alpha_keys}")
    for k in alpha_keys:
        v = state_dict[k]
        print(f"  {k}: {v}")

    # 5. Check proj weights
    proj_keys = [k for k in state_dict.keys() if 'proj' in k.lower() and 'planner' not in k.lower()]
    print(f"\nProjector keys ({len(proj_keys)}):")
    for k in proj_keys[:5]:
        v = state_dict[k]
        print(f"  {k}: shape={v.shape}, mean={v.float().mean():.6f}, std={v.float().std():.6f}")

    # 6. Try loading the model and check if weights are applied
    print("\n=== Loading model and checking weights ===")
    try:
        from pretrained_model.modeling_open_trackvla import OpenTrackVLAForWaypoint
        model = OpenTrackVLAForWaypoint.from_pretrained(hf_model_dir)

        # Check alpha_task
        if hasattr(model.model, 'alpha_task'):
            print(f"model.model.alpha_task: {model.model.alpha_task}")
        else:
            print("WARNING: model.model.alpha_task not found!")

        # Check planner weights
        if hasattr(model.model, 'planner'):
            planner = model.model.planner
            for name, param in planner.named_parameters():
                print(f"planner.{name}: shape={param.shape}, mean={param.float().mean():.6f}, std={param.float().std():.6f}")

        print("\nModel loaded successfully!")

    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    check_weights()
