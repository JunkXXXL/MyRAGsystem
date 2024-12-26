from fastapi import FastAPI
from pathlib import Path
from Context import Context
from database import orm

app = FastAPI()
context = Context()

summarization_handler = orm.SummarizationHandler(orm.client)
sentences_handler = orm.SentenceHandler(orm.client)

@app.get("/")
async def root():
    return {"message": "Hello world"}


@app.get("/Summarize_audio")
async def summarize_audio(file_name: str):
    audiofile_path = Path(file_name)

    if audiofile_path.exists() and audiofile_path.is_file():
        text_from_audio = context.transcriber.recognize(audiofile_path)
        summary_vector, summary_sentences = context.summarizer.summarize(text_from_audio["chunks"])

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


@app.get("/tableinfo")
async def get_info():

    try:
        table_info = summarization_handler.info()

    except BaseException as exception:
        error = "Error: \n" + str(exception)
        return {"execution": error}

    return {"clickhouse": table_info.result_rows}


@app.get("/hnsw_search")
async def hnsw_search(request_txt: str):
        summary_vector, _ = context.summarizer.summarize([request_txt])

        try:
            sentences = []
            file_path_to_similar = summarization_handler.hnsw_search(summary_vector)

            for index, element in enumerate(file_path_to_similar):
                sentences.append(sentences_handler.hnsw_search(summary_vector, file_path_to_similar[index][0]))

        except BaseException as exception:
            error = "Error: " + str(exception)
            return {"execution ": error}

        return {"execution ": file_path_to_similar, "sentences ": sentences}
