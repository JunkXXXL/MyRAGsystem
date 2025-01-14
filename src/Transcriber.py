from abc import ABC, abstractmethod
import speech_recognition as sr
import torch, pathlib
import subprocess
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class RecognizeStrategyAudio(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def recognize(self, path: pathlib.Path):
        pass


class RecognizeStrategyVideo(ABC):
    def __init__(self):
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

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3-turbo"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)

        model.to(self.device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self.device,
        )

    def _set_device(self):
        device = "cuda:0"
        if not torch.cuda.is_available():
            device = "cpu"
        return device

    def recognize(self, audio_path: pathlib.Path) -> dict:
        """Вернётся словарь с полями 'text' - текст записи,
        'language' - язык распознанной речи,
        'segments' - массив сегментов, каждый сегмент это словарь,
        в котором есть ключи начала и конца сегмента ('start', 'end')
        и ключ 'text'"""
        result = self.pipe(str(audio_path.absolute()), return_timestamps=True)
        return result


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
    path = pathlib.Path(r"D:\Users\aleksandr.kovalev\Desktop\Andrey_stazher\MyRAG\MyRAGsystem\logistic1.wav")

    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(str(path.absolute()), return_timestamps=True)

    for segment in result["chunks"]:
        start = segment["timestamp"][0]
        end = segment["timestamp"][1]
        text = segment["text"]
        print(f"[{start:.2f} - {end:.2f}] {text}")
