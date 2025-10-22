import subprocess
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


class VideoEncoder:

    def __init__(self, mask_video_streaming: bool, create_mask_config_from_roi: bool):
        # Hardware acceleration
        self.nvenc_available = self._detect_nvenc_support()
        self.preferred_encoder = self._get_preferred_encoder()
        
        # Encoder selection
    def _detect_nvenc_support(self) -> bool:
        """
        Detect if NVIDIA NVENC hardware acceleration is available.
        
        Returns:
            True if NVENC is available, False otherwise
        """
        try:
            # Test NVENC availability with a minimal command (minimum size for NVENC)
            cmd = [
                'ffmpeg', '-nostdin', '-f', 'lavfi', '-i', 'testsrc=duration=1:size=256x256:rate=1',
                '-c:v', 'h264_nvenc', '-preset', 'p1', '-f', 'null', '-'
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=15,
                check=False
            )
            
            if result.returncode == 0:
                logger.debug("NVENC h264 encoding test successful")
                return True
            else:
                logger.debug(f"NVENC test failed: {result.stderr}")
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug(f"NVENC detection failed: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error during NVENC detection: {e}")
            return False
    
    def _get_preferred_encoder(self) -> Dict[str, Any]:
        """
        Get the preferred video encoder configuration based on available hardware.
        
        Returns:
            Dictionary with encoder configuration
        """
        if self.nvenc_available:
            return {
                'name': 'h264_nvenc',
                'preset_param': '-preset',
                'preset_value': 'p4',  # Medium quality/speed for NVENC (P1=fastest, P7=best quality)
                'quality_param': '-cq',
                'quality_value': '20',  # NVENC CQ mode (lower = better quality)
                'type': 'nvenc',
                'fallback_preset': 'p1'  # Fastest NVENC preset for fallback
            }
        else:
            return {
                'name': 'libx264',
                'preset_param': '-preset',
                'preset_value': 'veryfast',  # CPU preset
                'quality_param': '-crf',
                'quality_value': '18',  # CPU CRF mode
                'type': 'cpu',
                'fallback_preset': 'ultrafast'  # Fastest CPU preset for fallback
            }
    
    def _build_encoder_cmd(self, 
                          quality_mode: str = 'balanced',
                          fallback: bool = False) -> List[str]:
        """
        Build encoder command arguments based on available hardware and quality requirements.
        
        Args:
            quality_mode: 'fast', 'balanced', or 'quality'
            fallback: Whether to use fallback settings for compatibility
            
        Returns:
            List of FFmpeg encoder arguments
        """
        encoder = self.preferred_encoder
        
        if encoder['type'] == 'nvenc':
            # NVIDIA NVENC configuration
            if fallback:
                preset = encoder['fallback_preset']  # p1 - fastest
                quality = '28'  # Lower quality for speed
            elif quality_mode == 'fast':
                preset = 'p2'  # Faster preset
                quality = '25'
            elif quality_mode == 'quality':
                preset = 'p6'  # Higher quality preset
                quality = '18'
            else:  # balanced
                preset = encoder['preset_value']  # p4
                quality = encoder['quality_value']  # 20
            
            return [
                '-c:v', encoder['name'],
                encoder['preset_param'], preset,
                encoder['quality_param'], quality,
                '-gpu', '0',  # Use first GPU
                '-rc', 'vbr',  # Variable bitrate
                '-profile:v', 'high'
            ]
        else:
            # CPU libx264 configuration
            if fallback:
                preset = encoder['fallback_preset']  # ultrafast
                quality = '23'  # Lower quality for speed
            elif quality_mode == 'fast':
                preset = 'faster'
                quality = '20'
            elif quality_mode == 'quality':
                preset = 'slow'
                quality = '15'
            else:  # balanced
                preset = encoder['preset_value']  # veryfast
                quality = encoder['quality_value']  # 18
            
            return [
                '-c:v', encoder['name'],
                encoder['preset_param'], preset,
                encoder['quality_param'], quality,
                '-profile:v', 'high'
            ]