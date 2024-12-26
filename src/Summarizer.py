from pathlib import Path
from abc import ABC, abstractmethod

import torch
from transformers import AutoTokenizer, AutoModel
from numpy import array


class SummarizerStrategy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def summarize(self, file_path: Path):
        pass


class SummarizerSBert(SummarizerStrategy):
    def __init__(self):
        super().__init__()
        self.device = self._set_device()
        self.tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
        self.model = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru")
        self.max_length_input_text = 512
        self.model.to(self.device)
        self.model.eval()

    def _set_device(self):
        device = "cuda:0"
        if not torch.cuda.is_available():
            device = "cpu"
        return device

    def summarize(self, text_to_summarize: list) -> [list, list]:
        """
        :param text_to_summarize:  массив предложений
        :return: вернёт summarized вектор, размерностью 1024; вернёт n векторов размерностью 1024
        """

        embeddings_list = []
        with torch.no_grad():
            for paragraph in text_to_summarize:
                str_paragraph = "".join(paragraph)
                input_ids = self.tokenizer.encode(str_paragraph, return_tensors="pt", add_special_tokens=True,
                                                  truncation=True, max_length=self.max_length_input_text)

                outputs = self.model(input_ids.to(self.device))
                cls_of_text = outputs[0][:, 0, :] #outputs[0][:, 0, :].tolist()
                embeddings_list.append(cls_of_text[0])

        averaged_vector = (torch.stack(embeddings_list).mean(dim=0))
        return averaged_vector.tolist(), [i.tolist() for i in embeddings_list]

    def avarage_summarize_vector(self, summarised_sentences: list) -> list:
        return list(array(summarised_sentences).mean(axis=0))



if __name__ == "__main__":
    text = """Посетил ОМСК ради соревнования, подвожу итоги:
    1. Омск самый жуткий город, который я видел
    2. Сибирские леса от пожаров спасти не удалось, заняли 10 место из 15. Зато мы заняли 10 место только из-за моего ночного решения
    3. Омск возвели за неделю до хакатона и разберут через месяц
    4. Еда попробована не вся, к сожалению люди слишком быстро наедаются
    5. В Омске людей нет, всё население выживает в бункерах т к на поверхности лишь заводы и кислотные дожди. Иногда в город прилетает гуманитарная помощь, но её обычно сбивают на границе региона. Еще глубже под бункером находится филиал ада.
    
    В прочем, поездка оказалась классной, если еще возместят половину от потраченной суммы, то готов буду хоть каждую неделю на хакатоны гонять
    """
    import Transcriber, pathlib

    path = pathlib.Path(r"D:\Users\aleksandr.kovalev\Desktop\Andrey_stazher\MyRAG\MyRAGsystem\logistic1.wav")

    whisper = Transcriber.RecognizeAudioWhisper()
    recognized_dict = whisper.recognize(path)

    summar = SummarizerSBert()
    vector, vectors = summar.summarize(recognized_dict["chunks"])
    print(len(vector))
