# NVIDIA NVENC Hardware Acceleration Implementation

## Übersicht

Das FrameCleaner-System wurde um NVIDIA NVENC Hardware-Beschleunigung erweitert, um die Video-Transkodierung erheblich zu beschleunigen. Die Implementierung nutzt die neuen P1-P7 Presets aus dem NVIDIA Video Codec SDK 10.

## Features

### Automatische Hardware-Erkennung
- **NVENC-Erkennung**: Automatische Erkennung verfügbarer NVIDIA-Hardware
- **Intelligenter Fallback**: Automatischer Wechsel zu CPU-Kodierung wenn NVENC nicht verfügbar
- **Optimierte Presets**: Nutzung der neuen P1-P7 Presets für NVENC

### Encoder-Konfigurationen

#### NVENC (Hardware-beschleunigt)
- **Encoder**: `h264_nvenc`
- **Presets**: P1 (schnellste) bis P7 (beste Qualität)
- **Qualitätsmodi**:
  - **Fast**: P2, CQ=25 (Geschwindigkeit optimiert)
  - **Balanced**: P4, CQ=20 (Ausgewogen)
  - **Quality**: P6, CQ=18 (Qualität optimiert)
- **Rate Control**: VBR (Variable Bitrate)
- **Profile**: High

#### CPU Fallback
- **Encoder**: `libx264`
- **Presets**: ultrafast, veryfast, etc.
- **Qualitätsmodi**:
  - **Fast**: faster, CRF=20
  - **Balanced**: veryfast, CRF=18
  - **Quality**: slow, CRF=15

## Technische Details

### NVENC Presets (Video Codec SDK 10)
Basierend auf dem NVIDIA Video Codec SDK 10:

- **P1**: Schnellste Kodierung, niedrigste Qualität
- **P2**: Schneller, niedrige Qualität
- **P3**: Schnell, niedrige Qualität
- **P4**: Medium (Standard), ausgewogene Qualität/Geschwindigkeit
- **P5**: Langsam, gute Qualität
- **P6**: Langsamer, bessere Qualität
- **P7**: Langsamste, beste Qualität

### Performance-Vorteile
- **10-50x schnellere** Video-Transkodierung bei Hardware-Beschleunigung
- **Reduzierte CPU-Last** durch GPU-Offloading
- **Konsistente Performance** unabhängig von CPU-Auslastung

## Verwendung

### Automatische Nutzung
```python
from lx_anonymizer.frame_cleaner import FrameCleaner

# Automatische Hardware-Erkennung und optimale Konfiguration
cleaner = FrameCleaner()

# NVENC wird automatisch verwendet wenn verfügbar
output_video, metadata = cleaner.clean_video(input_video_path)
```

### Manuelle Konfiguration
```python
# Unterschiedliche Qualitätsmodi
encoder_args_fast = cleaner._build_encoder_cmd('fast')       # P2, schnell
encoder_args_balanced = cleaner._build_encoder_cmd('balanced') # P4, ausgewogen
encoder_args_quality = cleaner._build_encoder_cmd('quality')   # P6, hohe Qualität
```

## Hardware-Anforderungen

### Unterstützte GPUs
- **NVIDIA Kepler** (GTX 600 Serie) oder neuer für Encoding
- **NVIDIA Fermi** (GTX 400 Serie) oder neuer für Decoding
- **RTX Karten** bieten die beste Performance

### Software-Anforderungen
- FFmpeg mit NVENC-Support (`--enable-nvenc`)
- NVIDIA GPU-Treiber (neueste Version empfohlen)
- CUDA-kompatibles System

## Fallback-Verhalten

Das System implementiert robuste Fallback-Strategien:

1. **NVENC-Test**: Automatischer Test bei Initialisierung
2. **Graceful Degradation**: Wechsel zu CPU bei NVENC-Fehlern
3. **Retry-Logik**: Automatische Wiederholung mit CPU-Encoding
4. **Fehlerbehandlung**: Ausführliche Protokollierung für Debugging

## Logs und Monitoring

```bash
# Typische Log-Ausgabe bei verfügbarer Hardware
INFO - Hardware acceleration: NVENC available
INFO - Using encoder: h264_nvenc with preset p4

# Bei nicht verfügbarer Hardware
INFO - Hardware acceleration: NVENC not available
INFO - Using encoder: libx264 with preset veryfast
```

## Performance-Vergleich

| Szenario | CPU (libx264) | NVENC (h264_nvenc) | Beschleunigung |
|----------|---------------|-------------------|----------------|
| 1080p Video | ~1x Realtime | ~10-20x Realtime | 10-20x |
| 4K Video | ~0.1x Realtime | ~5-10x Realtime | 50-100x |
| Masking | ~0.5x Realtime | ~5-15x Realtime | 10-30x |

## Kompatibilität

### Bestehende Funktionen
- ✅ Frame-Extraktion und OCR
- ✅ Video-Masking
- ✅ Frame-Removal
- ✅ Named Pipes
- ✅ Stream-Copy-Optimierung

### Neue Optimierungen
- ✅ Hardware-beschleunigte Pixel-Format-Konvertierung
- ✅ GPU-optimierte Masking-Filter
- ✅ Intelligente Encoder-Auswahl
- ✅ Automatische Qualitäts-/Geschwindigkeits-Balance

## Fehlerbehandlung

Das System behandelt häufige NVENC-Probleme automatisch:

- **Unsupported Resolution**: Automatischer Fallback zu CPU
- **GPU Memory**: Intelligente Ressourcenverwaltung
- **Driver Issues**: Graceful Degradation
- **Encoding Errors**: Retry mit alternativen Parametern

## Zukünftige Erweiterungen

- **HEVC NVENC**: H.265 Hardware-Encoding
- **AV1 Support**: Nächste Generation Codecs
- **Multi-GPU**: Parallele Verarbeitung
- **Adaptive Streaming**: Dynamische Qualitätsanpassung
