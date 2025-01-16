import Summarizer, Transcriber
import database.orm as orm
from TextSplitter import StandartSplitter, TextSplitter
from clickhouse_connect.driver import Client

class Context:
    def __init__(self):
        self.summarizer: Summarizer.SummarizerStrategy = Summarizer.SummarizerSBert()
        self.transcriber: Transcriber.RecognizeStrategyAudio = Transcriber.RecognizeAudioWhisper()
        self.client: Client = orm.client
        self.text_splitter: TextSplitter = StandartSplitter()
