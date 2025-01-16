from fastapi import FastAPI
from pathlib import Path
from Context import Context
from database import orm
from LLM import LlmGigaChat

app = FastAPI()

context = Context()

summarization_handler = orm.SummarizationHandler(orm.client)
sentences_handler = orm.SentenceHandler(orm.client)

llm = LlmGigaChat("Y2MyMTYxY2QtZTY0Zi00OGQyLWI0MTktMzM5OWFlNjJjODgwOjBmMTAzMDBiLTRkMTMtNGVhZS04NTIzLTVhZWE5ZjEyNDk0ZQ==")

@app.get("/")
async def root():
    return {"message": "Hello world"}


@app.get("/Summary_audio")
async def summary_audio(file_name: str):
    audiofile_path = Path(file_name)

    if audiofile_path.exists() and audiofile_path.is_file():
        text_from_audio = context.transcriber.recognize(audiofile_path)
        summary_vector = context.summarizer.summarize(text_from_audio["chunks"])
        text_sentences = context.text_splitter.split(text_from_audio["text"])
        summary_sentences = [context.summarizer.summarize([sentence]) for sentence in text_sentences]

        try:
            summarization_handler.insert_record({"file_name": file_name, "summary_vector": summary_vector})

            for index, sentence in enumerate(summary_sentences):
                if len(text_from_audio["chunks"][index]["text"]) != 0:
                    sentences_handler.insert_record({'master_name': file_name,
                                                     'summary_vector': sentence,
                                                     'sentence': text_from_audio["chunks"][index]["text"]})

        except BaseException as exception:
            error = "Error: \n" + str(exception)
            return {"execution": error}

        return {"execution": "ok"}

    else:
        return {"execution": "file not found"}

@app.get("/Summary_text")
async def summary_text(file_name: str):
    text_file = Path(file_name)

    if text_file.exists() and text_file.is_file():

        text = text_file.read_text(encoding='utf-8')
        text_sentences = context.text_splitter.split(text)
        summary_vector = context.summarizer.summarize(text_sentences)
        summary_sentences = [context.summarizer.summarize([sentence]) for sentence in text_sentences]

        try:
            summarization_handler.insert_record({"file_name": file_name, "summary_vector": summary_vector})

            for index, sentence in enumerate(summary_sentences):
                if len(text_sentences[index]) > 1:
                    sentences_handler.insert_record({'master_name': file_name,
                                                     'summary_vector': sentence,
                                                     'sentence': text_sentences[index]})

        except BaseException as exception:
            error = "Error: \n" + str(exception)
            return {"execution": error}

        return {"execution": "ok"}

    else:
        return {"execution": "file_not_found"}

@app.get("/tableinfo")
async def get_info():

    try:
        table_info = summarization_handler.info()

    except BaseException as exception:
        error = "Error: \n" + str(exception)
        return {"execution": error}

    return {"execution": table_info.result_rows}


@app.get("/hnsw_search")
async def hnsw_search(request_txt: str):
        summary_vector = context.summarizer.summarize([request_txt])

        try:
            sentences = []
            file_path_to_similar = summarization_handler.hnsw_search(summary_vector)

            for index, element in enumerate(file_path_to_similar):
                for sentence in sentences_handler.hnsw_search(summary_vector, file_path_to_similar[index][0]):
                    sentences.append(sentence[0])

        except BaseException as exception:
            error = "Error: " + str(exception)
            return {"execution ": error, "answer": 0}

        try:
            llm_result = llm.get_answer(request_txt, sentences)

        except BaseException as exception:
            error = "Error: " + str(exception)
            return {"execution ": error, "answer": 0}

        return {"execution ": sentences, "answer ": llm_result}
