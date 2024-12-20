from pathlib import Path
from abc import ABC, abstractmethod

import torch
from transformers import AutoTokenizer, AutoModel
from torch import sum, clamp
import wtpsplit


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

        self.sat = wtpsplit.SaT("sat-3l")
        self.sat.half().to(self.device)

    def _set_device(self):
        device = "cuda:0"
        if not torch.cuda.is_available():
            device = "cpu"
            print("Cuda is not available")
        return device

    def split_text(self, whole_text: str):
        split_text = self.sat.split(whole_text, do_paragraph_segmentation=True)
        return split_text

    def summarize(self, text_to_summarize: str) -> list:
        split_text = self.split_text(text_to_summarize)

        embeddings_list = []
        with torch.no_grad():
            for paragraph in split_text:
                str_paragraph = "".join(paragraph)
                input_ids = self.tokenizer.encode(str_paragraph, return_tensors="pt", add_special_tokens=True, truncation=True,
                                                      max_length=self.max_length_input_text)

                outputs = self.model(input_ids)
                cls_of_text = outputs[0][:, 0, :] #outputs[0][:, 0, :].tolist()
                embeddings_list.append(cls_of_text[0])

        averaged_vector = (torch.stack(embeddings_list).mean(dim=0))
        return list(averaged_vector)


if __name__ == "__main__":
    text = """Посетил ОМСК ради соревнования, подвожу итоги:
    1. Омск самый жуткий город, который я видел
    2. Сибирские леса от пожаров спасти не удалось, заняли 10 место из 15. Зато мы заняли 10 место только из-за моего ночного решения
    3. Омск возвели за неделю до хакатона и разберут через месяц
    4. Еда попробована не вся, к сожалению люди слишком быстро наедаются
    5. В Омске людей нет, всё население выживает в бункерах т к на поверхности лишь заводы и кислотные дожди. Иногда в город прилетает гуманитарная помощь, но её обычно сбивают на границе региона. Еще глубже под бункером находится филиал ада.
    
    В прочем, поездка оказалась классной, если еще возместят половину от потраченной суммы, то готов буду хоть каждую неделю на хакатоны гонять
    """
    summar = SummarizerSBert()
    vector = summar.summarize(text)
    print(len(vector))
