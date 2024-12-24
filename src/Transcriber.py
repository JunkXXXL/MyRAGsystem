from abc import ABC, abstractmethod
import speech_recognition as sr
import librosa, torch, pathlib
import subprocess
import whisper


class RecognizeStrategyAudio(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def _set_device(self):
        pass

    @abstractmethod
    def recognize(self, path: pathlib.Path):
        pass


class RecognizeStrategyVideo(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def _set_device(self):
        pass

    @abstractmethod
    def transcribe(self, path: pathlib.Path) -> str | None:
        pass

    @abstractmethod
    def convert_to_audio(self, audio_path: pathlib.Path) -> str | None:
        pass


class RecognizeAudioSpeech(RecognizeStrategyAudio):
    def __init__(self):
        super().__init__()
        self.recognizer = sr.Recognizer()
        self.device = self._set_device()

    def _set_device(self) -> str:
        device = "cuda:0"
        if not torch.cuda.is_available():
            device = "cpu"
        return device

    def recognize(self, path: pathlib.Path):
        path = str(path.absolute())
        try:
            with sr.AudioFile(path) as source:
                audio = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio, language="ru-RU")

        except sr.UnknownValueError:
            text = "Не удалось распознать речь"

        except sr.RequestError as e:
            text = f"Ошибка сервиса Google Speech Recognition: {e}"

        return text


class RecognizeAudioWhisper(RecognizeStrategyAudio):
    def __init__(self):
        super().__init__()

        self.device = self._set_device()

        self.model = whisper.load_model("turbo")
        self.model.to(self.device)

    def _set_device(self):
        device = "cuda:0"
        if not torch.cuda.is_available():
            device = "cpu"
        return device

    def recognize(self, audio_path: pathlib.Path) -> str | None:
        result = self.model.transcribe(str(audio_path.absolute()), temperature=1)
        return result.text


class TranscribeVideo(RecognizeStrategyVideo):
    def __init__(self):
        super().__init__()
        self.recognizer = sr.Recognizer()

    def _set_device(self):
        pass

    def recognize(self, video_path: pathlib.Path):
        audio_path = self.convert_to_audio(video_path)
        path = str(audio_path.absolute())

        try:
            with sr.AudioFile(path) as source:
                audio = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio, language="ru-RU")

        except sr.UnknownValueError:
            text = "Не удалось распознать речь"

        except sr.RequestError as e:
            text = f"Ошибка сервиса Google Speech Recognition: {e}"

        return text

    def convert_to_audio(self, video_path: pathlib.Path) -> pathlib.Path:
        input_mp4 = str(video_path.absolute())
        output_wav = (video_path.parent / video_path.stem).with_suffix(".wav")

        if not output_wav.is_file():
            subprocess.call(['ffmpeg', '-i', input_mp4, '-vn', 'copy', str(output_wav.absolute())])

            if not subprocess.check_output(
                    ['ffprobe', '-v', 'error', '-show_entries', 'format=duration,size,format_name', str(output_wav.absolute())],
                    stderr=subprocess.PIPE):
                raise RuntimeError(f'Conversion video to audio failed. Path {str(video_path.absolute())}')

        return output_wav


if __name__ == "__main__":
    path = pathlib.Path(r"C:\Users\SCII4\Desktop\Andrain\FastApi\logistic1.wav")

    recognizer = RecognizeAudioSpeech()
    text = recognizer.recognize(path)
    print(text)
    exit()

    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"
        print("Cuda is not avaliable")

    model = whisper.load_model("turbo")
    model.to(device)
    result = model.transcribe(str(path.absolute()))

    for segment in result["segments"]:
        start = segment["start"]  # Время начала сегмента
        end = segment["end"]  # Время окончания сегмента
        text = segment["text"]  # Распознанный текст сегмента
        print(f"[{start:.2f} - {end:.2f}] {text}")
