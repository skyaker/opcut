import ffmpeg
import numpy as np
import scipy.signal
import os
from config import SAMPLE_RATE, TEMP_DIR


def extract_audio(video_path, output_name):
    audio_path = os.path.join(TEMP_DIR, f"{output_name}.wav")
    if os.path.exists(audio_path):
        return audio_path
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec='pcm_s16le', ac=1, ar=SAMPLE_RATE)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return audio_path
    except ffmpeg.Error as e:
        print(f"Ошибка FFmpeg при извлечении звука: {e.stderr.decode('utf-8')}")
        return None


def load_audio_data(audio_path):
    if not audio_path:
        return None
    out, _ = (
        ffmpeg
        .input(audio_path)
        .output('pipe:', format='s16le', acodec='pcm_s16le', ac=1, ar=SAMPLE_RATE)
        .run(quiet=True)
    )
    return np.frombuffer(out, dtype=np.int16).astype(np.float32)


def _normalize(sig):
    std = np.std(sig)
    if std < 1e-6:
        return sig - np.mean(sig)
    return (sig - np.mean(sig)) / std


def _find_template_in_stream(template, stream, mask_ranges=None):
    """
    Ищет template в stream через NCC.
    mask_ranges: список [(start_sample, end_sample), ...] — регионы которые
    исключаются из поиска (обнуляются в корреляции). Используется чтобы
    найденный опенинг не мешал поиску эндинга.
    Возвращает (best_start_sec, best_score).
    """
    t = _normalize(template)
    corr = scipy.signal.fftconvolve(stream, t[::-1], mode='valid') / len(t)

    if mask_ranges:
        for s, e in mask_ranges:
            # корреляция в позиции i соответствует шаблону на [i, i+len(t)]
            # маскируем все позиции где шаблон перекрывается с регионом
            mask_s = max(0, s - len(t))
            mask_e = min(len(corr), e)
            if mask_s < mask_e:
                corr[mask_s:mask_e] = -np.inf

    best_idx = int(np.argmax(corr))
    best_score = float(corr[best_idx])
    best_start_sec = best_idx / SAMPLE_RATE
    return best_start_sec, best_score


def _extract_template(data, start_sec, duration_sec, margin_sec=5):
    """Вырезает сердцевину сегмента, отступая margin_sec от краёв."""
    s = int((start_sec + margin_sec) * SAMPLE_RATE)
    e = int((start_sec + duration_sec - margin_sec) * SAMPLE_RATE)
    if e <= s or e > len(data):
        s = int(start_sec * SAMPLE_RATE)
        e = int((start_sec + duration_sec) * SAMPLE_RATE)
    return data[s:e]


def _similarity_at(data_a, data_b, start_a_sec, start_b_sec, window_sec=3):
    w = int(window_sec * SAMPLE_RATE)
    s_a = int(start_a_sec * SAMPLE_RATE)
    s_b = int(start_b_sec * SAMPLE_RATE)
    if s_a < 0 or s_b < 0:
        return 0.0, False
    if s_a + w > len(data_a) or s_b + w > len(data_b):
        return 0.0, False
    chunk_a = data_a[s_a: s_a + w]
    chunk_b = data_b[s_b: s_b + w]
    silence_threshold = np.sqrt(np.mean(data_a ** 2)) * 0.01
    rms_a = np.sqrt(np.mean(chunk_a ** 2))
    rms_b = np.sqrt(np.mean(chunk_b ** 2))
    if (rms_a < silence_threshold) and (rms_b < silence_threshold):
        return 1.0, True
    a = _normalize(chunk_a)
    b = _normalize(chunk_b)
    score = float(np.dot(a, b) / len(a))
    return max(0.0, score), True


def _build_similarity_curve(data_a, data_b, start_a, start_b,
                             max_back_sec=120, max_fwd_sec=300, window_sec=3):
    shift = start_b - start_a
    half = window_sec / 2.0
    seconds = []
    scores = []
    for s in range(-int(max_back_sec), int(max_fwd_sec)):
        t_a = start_a + s - half
        t_b = t_a + shift
        score, valid = _similarity_at(data_a, data_b, t_a, t_b, window_sec)
        if not valid:
            continue
        seconds.append(s)
        scores.append(score)
    return np.array(seconds), np.array(scores)


def _find_boundaries_from_curve(seconds, scores, anchor_sec=0,
                                 low_threshold=0.25, gap_tolerance=5):
    sec_to_score = dict(zip(seconds.astype(int), scores))

    start_offset = 0
    gap = 0
    sec = anchor_sec
    while True:
        sec -= 1
        if sec not in sec_to_score:
            break
        if sec_to_score[sec] >= low_threshold:
            start_offset = sec
            gap = 0
        else:
            gap += 1
            if gap > gap_tolerance:
                break

    end_offset = 0
    gap = 0
    sec = anchor_sec
    while True:
        sec += 1
        if sec not in sec_to_score:
            break
        if sec_to_score[sec] >= low_threshold:
            end_offset = sec
            gap = 0
        else:
            gap += 1
            if gap > gap_tolerance:
                break

    return start_offset, end_offset


def _refine_boundary(data_a, data_b, rough_t_a, shift, direction,
                     search_sec=2.0, step=0.1, window_sec=1.0):
    half = window_sec / 2.0
    best_t = rough_t_a
    steps = np.arange(0, search_sec + step, step)
    if direction == 'start':
        for s in steps:
            t_a = rough_t_a - s
            t_b = t_a + shift
            score, valid = _similarity_at(data_a, data_b, t_a - half, t_b - half, window_sec)
            if valid and score >= 0.25:
                best_t = t_a
            else:
                break
    else:
        for s in steps:
            t_a = rough_t_a + s
            t_b = t_a + shift
            score, valid = _similarity_at(data_a, data_b, t_a - half, t_b - half, window_sec)
            if valid and score >= 0.25:
                best_t = t_a
            else:
                break
    return best_t


def _find_segment(data_a, data_b, template_len_sec=30, step_sec=10,
                  min_duration_sec=30, mask_ranges_a=None, label='segment',
                  low_threshold=0.25, gap_tolerance=5):
    """
    Универсальный поиск повторяющегося сегмента в паре серий.
    mask_ranges_a: [(start_sample, end_sample), ...] — регионы data_a которые
                   исключаются из поиска (уже найденный опенинг/эндинг).
    low_threshold, gap_tolerance: параметры границ — для эндинга используются
                   более мягкие значения чтобы захватить речевое вступление.
    Возвращает (start_a, start_b, duration) или (None, None, 0).
    """
    template_len = int(template_len_sec * SAMPLE_RATE)
    total_len = min(len(data_a), len(data_b))

    best_score = -np.inf
    anchor_a = None
    anchor_b = None

    s = 0
    while s + template_len <= total_len:
        if mask_ranges_a:
            in_mask = any(ms <= s < me for ms, me in mask_ranges_a)
            if in_mask:
                s += int(step_sec * SAMPLE_RATE)
                continue

        template = _normalize(data_a[s: s + template_len])
        start_b_sec, score = _find_template_in_stream(template, data_b,
                                                       mask_ranges=mask_ranges_a)
        if score > best_score:
            best_score = score
            anchor_a = s / SAMPLE_RATE + template_len_sec / 2
            anchor_b = start_b_sec + template_len_sec / 2

        s += int(step_sec * SAMPLE_RATE)

    if best_score < 0.4 or anchor_a is None:
        return None, None, 0

    print(f"  [DBG {label}] best_score={best_score:.3f} anchor_a={anchor_a:.1f}s anchor_b={anchor_b:.1f}s")

    seconds, scores = _build_similarity_curve(
        data_a, data_b, anchor_a, anchor_b,
        max_back_sec=120, max_fwd_sec=300, window_sec=3
    )
    if len(scores) == 0:
        return None, None, 0

    start_offset, end_offset = _find_boundaries_from_curve(
        seconds, scores, anchor_sec=0,
        low_threshold=low_threshold, gap_tolerance=gap_tolerance
    )

    actual_start_a = anchor_a + start_offset
    actual_end_a = anchor_a + end_offset
    shift = anchor_b - anchor_a

    actual_start_a = _refine_boundary(data_a, data_b, actual_start_a, shift, 'start')
    actual_end_a = _refine_boundary(data_a, data_b, actual_end_a, shift, 'end')

    duration = actual_end_a - actual_start_a

    print(f"  [DBG {label}] start_offset={start_offset} end_offset={end_offset} "
          f"→ start={actual_start_a:.1f}s end={actual_end_a:.1f}s dur={duration:.1f}s")
    if start_offset <= -118:
        print(f"  [DBG {label}] ⚠️  УПЁРСЯ В ЛИМИТ max_back_sec!")

    if duration < min_duration_sec:
        return None, None, 0

    actual_start_b = actual_start_a + shift
    return actual_start_a, actual_start_b, duration


def find_segments_in_pair(data_a, data_b, min_duration_sec=30,
                          search_only_ending=False, op_start=None, op_dur=None):
    """
    Ищет опенинг и/или эндинг в паре серий.
    search_only_ending=True — пропускает поиск опенинга, ищет только эндинг.
    op_start/op_dur — координаты опенинга для маскировки при поиске эндинга.
    Возвращает (op_a, op_b, op_dur, ed_a, ed_b, ed_dur).
    """
    if data_a is None or data_b is None:
        return None, None, 0, None, None, 0

    mask_ranges_a = None

    if search_only_ending:
        op_a = op_b = None
        op_dur_found = 0
        if op_start is not None and op_dur is not None:
            s = int(max(0, op_start - 30) * SAMPLE_RATE)
            e = int((op_start + op_dur + 30) * SAMPLE_RATE)
            mask_ranges_a = [(s, e)]
    else:
        op_a, op_b, op_dur_found = _find_segment(data_a, data_b,
                                                   min_duration_sec=min_duration_sec,
                                                   label='opening')
        if op_a is not None:
            s = int(max(0, op_a - 30) * SAMPLE_RATE)
            e = int((op_a + op_dur_found + 30) * SAMPLE_RATE)
            mask_ranges_a = [(s, e)]

    ed_a, ed_b, ed_dur_found = _find_segment(data_a, data_b,
                                              min_duration_sec=min_duration_sec,
                                              mask_ranges_a=mask_ranges_a,
                                              label='ending',
                                              low_threshold=0.15,
                                              gap_tolerance=15)

    if search_only_ending:
        return None, None, 0, ed_a, ed_b, ed_dur_found

    return op_a, op_b, op_dur_found, ed_a, ed_b, ed_dur_found


def find_segment_start_by_template(data_ref, start_ref_sec, duration_sec,
                                   data_target, label='segment',
                                   mask_ranges_target=None,
                                   low_threshold=0.25, gap_tolerance=5):
    """
    Зная сегмент в эталонной серии, находит его старт в data_target.
    mask_ranges_target: исключить регионы в data_target.
    low_threshold, gap_tolerance: для эндинга передаются мягкие значения.
    Возвращает (start_target_sec, duration_sec) или (None, 0).
    """
    if data_ref is None or data_target is None:
        return None, 0

    template_raw = _extract_template(data_ref, start_ref_sec, duration_sec, margin_sec=5)
    if len(template_raw) < 10 * SAMPLE_RATE:
        return None, 0

    template = _normalize(template_raw)
    anchor_target_sec, score = _find_template_in_stream(template, data_target,
                                                         mask_ranges=mask_ranges_target)
    if score < 0.4:
        return None, 0

    margin_sec = 5
    anchor_ref_sec = start_ref_sec + margin_sec + (len(template_raw) / SAMPLE_RATE) / 2
    anchor_target_sec = anchor_target_sec + (len(template_raw) / SAMPLE_RATE) / 2

    print(f"  [DBG {label}] score={score:.3f} anchor_ref={anchor_ref_sec:.1f}s "
          f"anchor_target={anchor_target_sec:.1f}s")

    seconds, scores = _build_similarity_curve(
        data_ref, data_target, anchor_ref_sec, anchor_target_sec,
        max_back_sec=120, max_fwd_sec=300, window_sec=3
    )
    if len(scores) == 0:
        return None, 0

    start_offset, end_offset = _find_boundaries_from_curve(
        seconds, scores, anchor_sec=0,
        low_threshold=low_threshold, gap_tolerance=gap_tolerance
    )

    shift = anchor_target_sec - anchor_ref_sec
    target_start = max(0.0, anchor_target_sec + start_offset)
    target_start = _refine_boundary(data_ref, data_target, target_start, shift, 'start')

    found_duration = (anchor_ref_sec + end_offset) - (anchor_ref_sec + start_offset)

    print(f"  [DBG {label}] start_offset={start_offset} end_offset={end_offset} "
          f"→ target_start={target_start:.1f}s found_dur={found_duration:.1f}s")

    if found_duration < 30:
        return None, 0

    return target_start, duration_sec


def cut_segments(video_path, segments, output_path):
    """
    Вырезает список сегментов из видео и склеивает оставшиеся куски.
    segments: список (start_sec, end_sec), отсортированных по времени.
    Если segments пустой — просто копирует файл.
    """
    import subprocess

    if not segments:
        import shutil
        shutil.copy2(video_path, output_path)
        return

    segments = sorted(segments)

    parts = []
    prev_end = 0.0

    for i, (seg_start, seg_end) in enumerate(segments):
        part = output_path + f"_part{i}.mp4"
        parts.append(part)
        subprocess.run([
            'ffmpeg', '-y', '-i', video_path,
            '-ss', str(prev_end), '-to', str(seg_start),
            '-c', 'copy', part
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        prev_end = seg_end

    tail = output_path + f"_part{len(parts)}.mp4"
    parts.append(tail)
    subprocess.run([
        'ffmpeg', '-y', '-ss', str(prev_end), '-i', video_path,
        '-c', 'copy', tail
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    list_file = output_path + "_concat_list.txt"
    try:
        with open(list_file, 'w') as f:
            for p in parts:
                f.write(f"file '{os.path.abspath(p)}'\n")

        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file,
            '-c', 'copy', output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print(f"✓ Серия успешно сохранена: {output_path}")

    except subprocess.CalledProcessError as e:
        print(f"✕ Ошибка FFmpeg: {e}")
    finally:
        for f in parts + [list_file]:
            if os.path.exists(f):
                os.remove(f)
