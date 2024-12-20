import Summarizer, Transcriber
import database.orm as orm


class Context:
    def __init__(self):
        self.summarizer = Summarizer.SummarizerSBert()
        self.transcriber = Transcriber.TranscribeAudio()
        self.client = orm.client
