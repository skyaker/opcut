import os
import glob
import shutil
from config import INPUT_DIR, OUTPUT_DIR
import audio_utils


def _iter_pairs(n):
    """
    Генерирует пары индексов (i, j) для поиска эталона.
    Сначала соседи (шаг 1), потом через одну (шаг 2), и т.д.
    Это решает проблему чередования 1,0,1,0 — когда опенинг
    есть только в каждой второй серии.
    Останавливаться нужно снаружи как только нашли.
    """
    for step in range(1, n):
        for i in range(n - step):
            yield i, i + step


def _find_ref(series_files):
    """
    Перебирает пары серий с увеличивающимся шагом пока не найдёт опенинг.
    Возвращает (ref_idx, data_ref, op_start, op_dur) или None если не нашёл.
    Аудио кэшируется в словаре чтобы не грузить одно и то же дважды.
    """
    audio_cache = {}

    def get_data(idx):
        if idx not in audio_cache:
            name = os.path.splitext(os.path.basename(series_files[idx]))[0]
            path = audio_utils.extract_audio(series_files[idx], name)
            audio_cache[idx] = audio_utils.load_audio_data(path)
        return audio_cache[idx]

    n = len(series_files)
    for i, j in _iter_pairs(n):
        name_a = os.path.splitext(os.path.basename(series_files[i]))[0]
        name_b = os.path.splitext(os.path.basename(series_files[j]))[0]
        print(f"  Пробуем пару (шаг {j-i}): {name_a} 🔍 {name_b}")

        data_a = get_data(i)
        data_b = get_data(j)

        op_a, _, op_dur, _, _, _ = audio_utils.find_segments_in_pair(data_a, data_b)

        if op_dur > 0:
            print(f"  ✓ Опенинг найден! Эталон: серия '{name_a}', "
                  f"старт={op_a:.1f}s длит={op_dur:.1f}s")
            return i, data_a, op_a, op_dur

        print(f"  Опенинг не найден в этой паре.")

    return None


def _find_ed_ref(series_files, data_ref, op_start_ref, op_dur_ref):
    """
    Отдельный проход для поиска эндинга — тоже перебирает пары с шагом.
    Эталонная серия для эндинга может отличаться от эталона опенинга.
    Возвращает (ed_ref_idx, data_ed_ref, ed_start, ed_dur) или None.
    """
    audio_cache = {}

    def get_data(idx):
        if idx not in audio_cache:
            name = os.path.splitext(os.path.basename(series_files[idx]))[0]
            path = audio_utils.extract_audio(series_files[idx], name)
            audio_cache[idx] = audio_utils.load_audio_data(path)
        return audio_cache[idx]

    op_mask_ref = [(
        int(max(0, op_start_ref - 30) * audio_utils.SAMPLE_RATE),
        int((op_start_ref + op_dur_ref + 30) * audio_utils.SAMPLE_RATE)
    )]

    n = len(series_files)
    for i, j in _iter_pairs(n):
        name_a = os.path.splitext(os.path.basename(series_files[i]))[0]
        name_b = os.path.splitext(os.path.basename(series_files[j]))[0]
        print(f"  Пробуем пару (шаг {j-i}): {name_a} 🔍 {name_b}")

        data_a = get_data(i)
        data_b = get_data(j)

        _, _, _, ed_a, _, ed_dur = audio_utils.find_segments_in_pair(
            data_a, data_b,
            search_only_ending=True,
            op_start=op_start_ref,
            op_dur=op_dur_ref
        )

        if ed_dur > 0:
            print(f"  ✓ Эндинг найден! Эталон: серия '{name_a}', "
                  f"старт={ed_a:.1f}s длит={ed_dur:.1f}s")
            return i, data_a, ed_a, ed_dur

        print(f"  Эндинг не найден в этой паре.")

    return None


def main():
    video_extensions = ('*.mkv', '*.mp4', '*.avi')
    series_files = []
    for ext in video_extensions:
        series_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    series_files.sort()

    if len(series_files) < 2:
        print(f"Пожалуйста, положите хотя бы 2 серии в папку {INPUT_DIR}")
        return

    print(f"Найдено серий для обработки: {len(series_files)}")

    print("\n[1/4] Поиск эталона опенинга...")
    result = _find_ref(series_files)
    if result is None:
        print("\n[ОШИБКА] Опенинг не найден ни в одной паре серий.")
        return
    op_ref_idx, data_op_ref, op_start_ref, op_dur_ref = result

    print("\n[2/4] Поиск эталона эндинга...")
    ed_result = _find_ed_ref(series_files, data_op_ref, op_start_ref, op_dur_ref)
    if ed_result is None:
        print("  Эндинг не найден ни в одной паре — будем вырезать только опенинги.")
        ed_ref_idx = None
        data_ed_ref = None
        ed_start_ref = None
        ed_dur_ref = 0
    else:
        ed_ref_idx, data_ed_ref, ed_start_ref, ed_dur_ref = ed_result
        if ed_ref_idx == op_ref_idx and ed_start_ref < op_start_ref:
            op_start_ref, op_dur_ref, ed_start_ref, ed_dur_ref = (
                ed_start_ref, ed_dur_ref, op_start_ref, op_dur_ref
            )

    print("\n[3/4] Обработка всего сезона...")

    for idx, video_path in enumerate(series_files):
        filename = os.path.basename(video_path)
        output_path = os.path.join(OUTPUT_DIR, f"no_op_{filename}")
        print(f"\nАнализ [{idx + 1}/{len(series_files)}]: {filename}")

        current_name = os.path.splitext(filename)[0]
        current_audio = audio_utils.extract_audio(video_path, current_name)
        current_data = audio_utils.load_audio_data(current_audio)

        if idx == op_ref_idx:
            op_start = op_start_ref
        else:
            op_start, _ = audio_utils.find_segment_start_by_template(
                data_op_ref, op_start_ref, op_dur_ref, current_data, label='opening'
            )

        if op_start is None:
            print(f"⚠️  Опенинг не найден — копируем целиком.")
            shutil.copy2(video_path, output_path)
            continue

        op_end = op_start + op_dur_ref

        ed_start = None
        ed_end = None
        if ed_dur_ref > 0:
            op_mask = [(
                int(max(0, op_start - 30) * audio_utils.SAMPLE_RATE),
                int((op_end + 30) * audio_utils.SAMPLE_RATE)
            )]
            if idx == ed_ref_idx:
                ed_start = ed_start_ref
            else:
                ed_start, _ = audio_utils.find_segment_start_by_template(
                    data_ed_ref, ed_start_ref, ed_dur_ref, current_data,
                    label='ending', mask_ranges_target=op_mask,
                    low_threshold=0.15, gap_tolerance=15
                )
            if ed_start is not None:
                ed_end = ed_start + ed_dur_ref
            else:
                print(f"⚠️  Эндинг не найден в этой серии — вырезаем только опенинг.")

        segments = [(op_start, op_end)]
        if ed_start is not None:
            segments.append((ed_start, ed_end))

        info = f"опенинг {op_start:.1f}s–{op_end:.1f}s"
        if ed_start is not None:
            info += f", эндинг {ed_start:.1f}s–{ed_end:.1f}s"
        print(f"➔ Вырезаем: {info}")

        audio_utils.cut_segments(video_path, segments, output_path)

    print(f"\n[4/4] Готово! Файлы в папке: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
