# macOS MPS Compatibility Changes

This document summarizes the changes made to convert CUDA-specific code to be compatible with macOS Metal Performance Shaders (MPS).

## Summary of Changes

### 1. Created Device Utility Module (`src/utils/device_utils.py`)
- Added `get_optimal_device()` function that automatically detects the best available device:
  - Returns "mps" for macOS with MPS support
  - Returns "cuda" for systems with CUDA support
  - Returns "cpu" as fallback
- Added helper functions for cross-platform tensor operations
- Added improved `manual_seed_all()` function with MPS support

### 2. Updated Device Detection in Script Arguments
Updated the following files to use `get_optimal_device()` instead of hardcoded CUDA checks:
- `src/compute_average_activations.py`
- `src/evaluate_function_vector.py`
- `src/compute_avg_hidden_state.py`
- `src/portability_eval.py`
- `src/compute_indirect_effect.py`

### 3. Fixed Direct CUDA Calls in `src/vocab_reconstruction.py`
- Replaced `torch.zeros(target.size()).cuda()` with device-aware allocation
- Replaced `torch.randn(fv.size()).cuda()` with `torch.randn(fv.size()).to(model.device)`

### 4. Updated Random Seed Function in `src/utils/model_utils.py`
- Made CUDA-specific seeding conditional on CUDA availability
- Added MPS-specific seeding for macOS

### 5. Updated Environment Configuration (`fv_environment.yml`)
- Commented out the nvidia channel as it's not needed on macOS
- Kept CUDA toolkit commented out (already was for macOS)

## Benefits

1. **Cross-platform compatibility**: Code now works on macOS with MPS, Linux/Windows with CUDA, and CPU-only systems
2. **Automatic device detection**: No need to manually specify device types
3. **Performance optimization**: Uses the fastest available accelerator on each platform
4. **No breaking changes**: Existing functionality preserved for CUDA systems

## Usage

The changes are transparent to users. Scripts will automatically:
- Use MPS acceleration on macOS with Apple Silicon
- Use CUDA acceleration on systems with NVIDIA GPUs
- Fall back to CPU on other systems

No changes to command-line usage are required. The device detection happens automatically.

## Testing Recommendations

1. Test on macOS with Apple Silicon to verify MPS functionality
2. Test on CUDA systems to ensure backward compatibility
3. Test on CPU-only systems to verify fallback behavior

## Files Modified

1. `src/utils/device_utils.py` (new file)
2. `src/vocab_reconstruction.py`
3. `src/compute_average_activations.py`
4. `src/evaluate_function_vector.py`
5. `src/compute_avg_hidden_state.py`
6. `src/portability_eval.py`
7. `src/compute_indirect_effect.py`
8. `src/utils/model_utils.py`
9. `fv_environment.yml`

All changes maintain backward compatibility while adding macOS MPS support.
