from lx_anonymizer.frame_cleaner import FrameCleaner
from pathlib import Path

# Test that FrameCleaner can properly initialize with MaskApplication
frame_cleaner = FrameCleaner(use_llm=False)
print('✅ FrameCleaner initialized successfully')

# Test that the masking methods are properly accessible
print(f'✅ _mask_video_streaming method: {hasattr(frame_cleaner, "_mask_video_streaming")}')
print(f'✅ _create_mask_config_from_roi method: {hasattr(frame_cleaner, "_create_mask_config_from_roi")}')

# Test creating a mask config
test_roi = {
    'x': 550, 'y': 0, 'width': 1350, 'height': 1080,
    'image_width': 1920, 'image_height': 1080
}
mask_config = frame_cleaner._create_mask_config_from_roi(test_roi)
print(f'✅ Mask config created: {mask_config}')