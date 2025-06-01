import os
import subprocess
import threading
import time
from typing import Optional, Dict

class HLSGenerator:
    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self.frames_dir = os.path.join("/app/streams", stream_id, "frames")
        self.hls_dir = os.path.join("/app/streams", stream_id, "hls")
        self.process: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.is_running = False
        self.last_frame = 0
        print(f"Инициализирован HLSGenerator для потока {stream_id}")

    def start(self):
        """Запускает генерацию HLS потока в отдельном потоке"""
        if self.is_running:
            print(f"HLSGenerator для потока {self.stream_id} уже запущен")
            return

        # Создаем директорию для HLS если её нет
        os.makedirs(self.hls_dir, exist_ok=True)
        print(f"Создана директория HLS для потока {self.stream_id} по пути {self.hls_dir}")

        self.is_running = True
        self.thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self.thread.start()
        print(f"Запущен поток генерации HLS для потока {self.stream_id}")

    def stop(self):
        """Останавливает генерацию HLS потока"""
        print(f"Останавливаем HLSGenerator для потока {self.stream_id}")
        self.is_running = False
        if self.process:
            print(f"Завершаем процесс GStreamer для потока {self.stream_id}")
            self.process.terminate()
            self.process.wait()
        if self.thread:
            print(f"Ожидаем завершения потока генерации для потока {self.stream_id}")
            self.thread.join()

    def _get_latest_frame_number(self) -> int:
        """Получает номер последнего доступного кадра"""
        try:
            files = [f for f in os.listdir(self.frames_dir) if f.startswith('frame_') and f.endswith('.jpg')]
            if not files:
                print(f"Кадры не найдены в {self.frames_dir}")
                return 0
            numbers = [int(f.split('_')[1].split('.')[0]) for f in files]
            latest = max(numbers)
            print(f"Последний номер кадра для потока {self.stream_id}: {latest}")
            return latest
        except Exception as e:
            print(f"Ошибка при получении последнего номера кадра для потока {self.stream_id}: {str(e)}")
            return 0

    def _create_empty_playlist(self):
        """Создает пустой HLS плейлист"""
        playlist_path = os.path.join(self.hls_dir, "playlist.m3u8")
        try:
            with open(playlist_path, 'w') as f:
                f.write("#EXTM3U\n")
                f.write("#EXT-X-VERSION:3\n")
                f.write("#EXT-X-TARGETDURATION:10\n")
                f.write("#EXT-X-MEDIA-SEQUENCE:0\n")
            print(f"Создан пустой плейлист для потока {self.stream_id}")
        except Exception as e:
            print(f"Ошибка при создании пустого плейлиста для потока {self.stream_id}: {str(e)}")

    def _run_pipeline(self):
        """Запускает GStreamer pipeline в отдельном процессе"""
        print(f"Запускаем цикл pipeline для потока {self.stream_id}")
        while self.is_running:
            try:
                # Проверяем, есть ли новые кадры
                latest_frame = self._get_latest_frame_number()
                
                if latest_frame == 0:
                    # Если кадров нет, создаем пустой плейлист
                    print(f"Нет доступных кадров для потока {self.stream_id}, создаем пустой плейлист")
                    self._create_empty_playlist()
                    time.sleep(1)  # Ждем появления новых кадров
                    continue

                # Всегда начинаем с первого кадра
                self.last_frame = 0

                print(f"Обрабатываем кадры для потока {self.stream_id} с {self.last_frame} по {latest_frame}")
                # Создаем pipeline для обработки новых кадров
                pipeline = [
                    "gst-launch-1.0",
                    f"multifilesrc location={self.frames_dir}/frame_%d.jpg index=0 caps=\"image/jpeg,framerate=1/1\"",
                    "! jpegdec ! videoconvert",
                    "! x264enc tune=zerolatency bitrate=2000 speed-preset=superfast",
                    "! h264parse ! mpegtsmux",
                    f"! hlssink location={self.hls_dir}/segment_%05d.ts playlist-location={self.hls_dir}/playlist.m3u8 target-duration=10 max-files=5"
                ]

                # Запускаем pipeline
                print(f"Запускаем GStreamer pipeline для потока {self.stream_id}")
                self.process = subprocess.Popen(
                    " ".join(pipeline),
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                self.process.wait()

                # Обновляем номер последнего обработанного кадра
                self.last_frame = latest_frame
                print(f"Успешно обработаны кадры до {self.last_frame} для потока {self.stream_id}")

            except Exception as e:
                print(f"Ошибка в HLS pipeline для потока {self.stream_id}: {str(e)}")
                time.sleep(1)  # Ждем перед повторной попыткой

class HLSManager:
    def __init__(self):
        self.generators: Dict[str, HLSGenerator] = {}
        print("Инициализирован HLSManager")

    def start_generator(self, stream_id: str) -> HLSGenerator:
        """Запускает генератор HLS для указанного потока"""
        if stream_id in self.generators:
            print(f"Генератор для потока {stream_id} уже существует")
            raise Exception(f"Генератор HLS для потока {stream_id} уже существует")

        print(f"Запускаем новый HLS генератор для потока {stream_id}")
        generator = HLSGenerator(stream_id)
        generator.start()
        self.generators[stream_id] = generator
        return generator

    def stop_generator(self, stream_id: str):
        """Останавливает генератор HLS для указанного потока"""
        if stream_id in self.generators:
            print(f"Останавливаем HLS генератор для потока {stream_id}")
            self.generators[stream_id].stop()
            del self.generators[stream_id]
        else:
            print(f"Попытка остановить несуществующий генератор для потока {stream_id}")

    def get_generator(self, stream_id: str) -> Optional[HLSGenerator]:
        """Получает генератор HLS по ID потока"""
        return self.generators.get(stream_id) 