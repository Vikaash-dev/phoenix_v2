# Pull Request References

This document tracks the source PRs for each version.

## Version 1 (Root Directory)
- **Source**: Main branch (current state)
- **PR #1**: Initial implementation - "Implement AI brain tumor detection system with research paper and CNN implementation"
- **PR #2**: "Implement Phoenix Protocol: Complete NeuroSnake Implementation with ALL Features"
- **PR #3**: "Add research experimental validation framework for publication-ready results"
- **Status**: Production-ready in root directory
- **Commit**: 88683fc
- **Location**: Root directory (you're here by default)

## Version 2 (SOTA Upgrade)
- **Source PR**: #11
- **Title**: "Phoenix Protocol SOTA Upgrade: AMP, k-Fold, & Advanced Loss Functions"
- **Branch**: `phoenix-protocol-sota-upgrade-13225564203933682807`
- **Status**: Open (ready to merge)
- **Key Changes**:
  - Mixed Precision Training (AMP)
  - K-Fold Cross-Validation
  - SEVector Attention
  - Log-Cosh Dice Loss
  - ONNX export and serving
  - KAN layers integration
- **Files Changed**: 24 files (1,688 additions, 1,727 deletions)
- **Branch SHA**: ef915c20e40c35066e63bc3e11e9d8ce2da485cc

## Version 3 (Spectral-Snake)
- **Source PR**: #12
- **Title**: "feat: Add Spectral-Snake Architecture and AI Scientist Research Framework"
- **Branch**: `research/neurosnake-spectral-7226522193343951134`
- **Status**: Open (ready to merge)
- **Key Changes**:
  - Spectral Gating Blocks (FFT-based)
  - AI Scientist research framework
  - Bug fixes in Dynamic Snake Convolution
  - Complete research paper and peer review simulation
- **Files Changed**: 50 files (4,515 additions, 394 deletions)
- **Branch SHA**: 976fbef2b790ce60d621e8f2e6b85ad72cde395a

## Additional Performance PRs

These PRs contain performance optimizations that can be applied to any version:

### PR #4: CLAHE Optimization
- **Title**: "Performance: Optimize CLAHE using LAB color space"
- **Branch**: `perf-clahe-optimization-5385895825720896997`
- **Focus**: Preprocessing optimization

### PR #6: Parameter Counting
- **Title**: "⚡ Optimize trainable parameter counting in CNN model"
- **Branch**: `perf-optimize-params-count-9382974215360769563`
- **Focus**: Model initialization optimization

### PR #7: EfficientQuant Stats
- **Title**: "Optimize EfficientQuant layer statistics collection"
- **Branch**: `perf-efficient-quant-layer-stats-4729608087426733500`
- **Focus**: Quantization optimization

### PR #8: Dataset Loading
- **Title**: "⚡ Optimize dataset loading with tf.data streaming"
- **Branch**: `perf-optimize-dataset-loading-2113112794973398003`
- **Focus**: Data pipeline optimization

### PR #9: Parallel Loading
- **Title**: "⚡ Bolt: Parallelize data loading with ProcessPoolExecutor"
- **Branch**: `bolt/parallel-data-loading-4497968694327660098`
- **Focus**: Parallel data loading

### PR #10: INT8 Data Loading
- **Title**: "Implement data loading for INT8 quantization"
- **Branch**: `jules/int8-quantization-data-loading-11366970919926156499`
- **Focus**: Quantization data pipeline

### PR #5: Research Analysis
- **Title**: "Research Agent Analysis of Phoenix Protocol"
- **Branch**: `research-agent-analysis-7686887063306407004`
- **Focus**: Analysis and documentation

## Merging Strategy

To incorporate these versions into main:

1. **v1**: Already in main
2. **v2**: Review PR #11, then merge to main
3. **v3**: Review PR #12, then merge to main
4. **Performance PRs**: Cherry-pick relevant optimizations as needed

## GitHub PR Links

- [PR #11 - v2 SOTA Upgrade](https://github.com/Vikaash-dev/Ai-research-paper-and-implementation-of-brain-tumor-detection-/pull/11)
- [PR #12 - v3 Spectral-Snake](https://github.com/Vikaash-dev/Ai-research-paper-and-implementation-of-brain-tumor-detection-/pull/12)
- [All Open PRs](https://github.com/Vikaash-dev/Ai-research-paper-and-implementation-of-brain-tumor-detection-/pulls)
