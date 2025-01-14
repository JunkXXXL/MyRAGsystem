import Summarizer, Transcriber
import database.orm as orm
from TextSplitter import StandartSplitter

class Context:
    def __init__(self):
        self.summarizer = Summarizer.SummarizerSBert()
        self.transcriber = Transcriber.RecognizeAudioWhisper()
        self.client = orm.client
        self.text_splitter = StandartSplitter()
