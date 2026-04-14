import numpy as np
from scipy.io import wavfile
from sklearn.cluster import KMeans

def find_carrier_frequency(samples, fs, freq_range=(300, 3000)):
    window_len = int(0.1 * fs)
    rms = np.array([np.sqrt(np.mean(samples[i:i + window_len] ** 2))
                    for i in range(0, len(samples) - window_len, window_len // 2)])
    start = np.argmax(rms) * (window_len // 2)
    segment = samples[start:start + window_len]
    spectrum = np.fft.rfft(segment * np.hanning(len(segment)))
    freqs = np.fft.rfftfreq(len(segment), 1 / fs)
    magnitude = np.abs(spectrum)
    idx = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
    if len(idx) == 0:
        raise ValueError("Частота не найдена")
    peak_idx = idx[np.argmax(magnitude[idx])]
    return freqs[peak_idx]


def complex_correlation(segment, freq, fs, center=True):
    t = np.arange(len(segment)) / fs
    exp = np.exp(-2j * np.pi * freq * t)
    if center:
        segment = segment - np.mean(segment)
    corr = np.sum(segment * exp)
    energy = np.sum(segment ** 2)
    if energy == 0:
        return 0
    amp = np.abs(corr) / np.sqrt(energy * len(segment))
    return amp


def find_best_shift_and_duration(samples, fs, freq, min_dur=0.04, max_dur=0.135, step=0.005):
    best_dur = min_dur
    best_shift = 0
    best_sep = 0
    print("Поиск оптимальной длительности и сдвига...")
    for dur in np.arange(min_dur, max_dur + step, step):
        spb = int(dur * fs)
        if spb < 1:
            continue
        shifts = range(0, spb, max(1, spb // 20))
        for shift in shifts:
            num_bits = (len(samples) - shift) // spb
            if num_bits < 20:
                continue
            truncated = samples[shift: shift + num_bits * spb]
            corrs = []
            for i in range(num_bits):
                seg = truncated[i * spb:(i + 1) * spb]
                corrs.append(complex_correlation(seg, freq, fs, center=True))
            # Оценка разделения через KMeans
            X = np.array(corrs).reshape(-1, 1)
            if len(np.unique(X)) < 2:
                continue
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X)
            labels = kmeans.labels_
            mean0, mean1 = np.mean(X[labels == 0]), np.mean(X[labels == 1])
            if mean0 > mean1:
                mean_high, mean_low = mean0, mean1
                std_high, std_low = np.std(X[labels == 0]), np.std(X[labels == 1])
            else:
                mean_high, mean_low = mean1, mean0
                std_high, std_low = np.std(X[labels == 1]), np.std(X[labels == 0])
            sep = (mean_high - mean_low) / (std_high + std_low + 1e-6)
            if sep > best_sep:
                best_sep = sep
                best_dur = dur
                best_shift = shift
                print(f"  dur={dur * 1000:.1f}ms shift={shift} ({shift / fs * 1000:.2f}ms) sep={sep:.3f}")
    return best_dur, best_shift


def bits_to_ascii(bits):
    if len(bits) < 8:
        return ""
    bits = bits[:-(len(bits) % 8)] if len(bits) % 8 != 0 else bits
    chars = []
    for i in range(0, len(bits), 8):
        code = int(bits[i:i + 8], 2)
        if 32 <= code <= 126 or code in (9, 10, 13):
            chars.append(chr(code))
        else:
            chars.append('?')
    return ''.join(chars)


def main():
    # === НАСТРОЙКИ (для ручной фиксации параметров) ===
    FORCE_FREQ = None  # укажите нужную частоту в Гц (например, 2100.0) или None для автоматического поиска
    FORCE_SHIFT = None  # укажите желаемое смещение в отсчётах (например, 0) или None для автоматического поиска
    FORCE_DUR = None  # укажите желаемую длительность бита в секундах (например, 0.115) или None для автоматического поиска
    # =================================================

    wav_file = "../Data/5_2_49.wav"
    fs, samples = wavfile.read(wav_file)
    if samples.dtype != np.float64:
        samples = samples.astype(np.float64) / np.max(np.abs(samples))
    if len(samples.shape) > 1:
        samples = samples[:, 0]

    print(f"Обработка файла: {wav_file}, частота дискретизации: {fs} Гц")
    freq = find_carrier_frequency(samples, fs)
    print(f"Определена частота синусоиды: {freq:.1f} Гц")

    dur, shift = find_best_shift_and_duration(samples, fs, freq)

    # === Принудительная установка параметров, если они заданы вручную ===
    if FORCE_FREQ is not None:
        freq = FORCE_FREQ
        print(f"** Частота синусоиды принудительно установлена: {freq:.1f} Гц **")
    if FORCE_DUR is not None:
        dur = FORCE_DUR
        print(f"** Длительность бита принудительно установлена: {dur * 1000:.1f} мс **")
    if FORCE_SHIFT is not None:
        shift = FORCE_SHIFT
        print(f"** Смещение принудительно установлено: {shift} отсч. ({shift / fs * 1000:.2f} мс) **")
    # =================================================

    spb = int(dur * fs)
    num_bits = (len(samples) - shift) // spb
    truncated = samples[shift: shift + num_bits * spb]
    corrs = [complex_correlation(truncated[i * spb:(i + 1) * spb], freq, fs, center=True) for i in range(num_bits)]

    # --- Порог: используем KMeans для разделения ---
    X = np.array(corrs).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X)
    labels = kmeans.labels_
    mean0, mean1 = np.mean(X[labels == 0]), np.mean(X[labels == 1])
    # Определяем, какой кластер соответствует "1" (большее среднее)
    if mean0 > mean1:
        threshold = (mean0 + mean1) / 2
        bits = ''.join(['1' if c > threshold else '0' for c in corrs])
    else:
        threshold = (mean1 + mean0) / 2
        bits = ''.join(['1' if c > threshold else '0' for c in corrs])

    text = bits_to_ascii(bits)

    print("\nКорреляции по сегментам:")
    for i, c in enumerate(corrs):
        print(f"{i:2d}: {c:.4f} -> {bits[i]}")

    print("\n=== РЕЗУЛЬТАТ ДЕКОДИРОВАНИЯ ===")
    print(f"Файл: {wav_file}")
    print(f"Частота дискретизации: {fs} Гц")
    print(f"Частота синусоиды: {freq:.1f} Гц")
    print(f"Длительность бита: {dur * 1000:.1f} мс")
    print(f"Смещение: {shift} отсчётов ({shift / fs * 1000:.2f} мс)")
    print(f"\nБитовая последовательность ({len(bits)} бит):")
    print(bits)
    print(f"\nДекодированное сообщение: {text}")


if __name__ == "__main__":
    main()