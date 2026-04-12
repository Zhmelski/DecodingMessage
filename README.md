# Декодирование последовательности гармонических импульсов / Decoding of harmonic pulse sequence

Данный проект реализует декодирование бинарных данных, переданных в виде последовательности синусоидальных импульсов на фоне аддитивного шума. Программа анализирует WAV-файл (моно, частота дискретизации 8 кГц), автоматически определяет частоту несущей и длительность бита, вычисляет корреляцию с опорным сигналом и преобразует полученную битовую последовательность в ASCII-строку.

This project decodes binary data transmitted as a sequence of sinusoidal pulses in additive noise. The program analyses a mono WAV file (8 kHz sampling rate), automatically determines the carrier frequency and bit duration, computes correlation with a reference sine wave, and converts the resulting bit sequence into an ASCII string.

## Требования / Requirements

- Python 3.8+
- Установленные зависимости / installed dependencies:
  - `numpy`
  - `scipy`
  - `scikit-learn`

Все зависимости можно установить командой / All dependencies can be installed with:

```bash
pip install -r requirements.txt