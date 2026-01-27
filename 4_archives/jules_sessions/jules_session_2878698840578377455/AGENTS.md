# SESSION 2878 - PHYSICS-INFORMED AUGMENTATION

**Context**: Minimal session focused solely on MRI-realistic data augmentation.

## OVERVIEW

Implements physics-based augmentations that simulate real MRI artifacts instead of generic image transforms.

## STRUCTURE

```text
jules_session_2878.../
├── src/physics_informed_augmentation.py   # Main implementation
└── tests/verify_augmentation.py            # Verification
```

## WHERE TO LOOK

| Task | File | Notes |
| :--- | :--- | :--- |
| **Augmentation Class** | `src/physics_informed_augmentation.py` | `PhysicsInformedAugmentation` |
| **Verification** | `tests/verify_augmentation.py` | Visual test |

## KEY INNOVATIONS

- **Elastic Deformation**: Simulates tissue displacement during MRI acquisition
- **Rician Noise**: MRI-specific noise model (not Gaussian)

## MERGE PRIORITY

**HIGH** - This augmentation should be merged to root `src/` for training pipeline.
