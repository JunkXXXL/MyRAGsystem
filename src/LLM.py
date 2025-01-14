from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat
from langchain.retrievers import BM25Retriever
from abc import ABC, abstractmethod

class LlmStrategy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_answer(self, question: str, retriver_data: list) -> str:
        pass


class LlmGigaChat(LlmStrategy, ABC):
    def __init__(self, Authorization_Key):

        super().__init__()
        self.model = GigaChat(
        credentials=Authorization_Key,
        scope="GIGACHAT_API_PERS",
        model="GigaChat",
        streaming=False,
        verify_ssl_certs=False,
        )

        self.messages = [SystemMessage(
                content="Ты помощник, который использует предоставленные данные."
                        " Если в данных нет ответа на вопрос, ответь 'Я не знаю.'. "
                        "Отвечай кратко")]

    def get_answer(self, question: str, retriver_data: list) -> str:
        self.messages.append(HumanMessage(content="Данные: ".join(retriver_data) + " Вопрос: " + question))
        #self.messages.append(HumanMessage(content=question))

        res = self.model.invoke(self.messages)
        self.messages.append(res)
        return res.content


if __name__ == "__main__":
    Authorization_Key = "Y2MyMTYxY2QtZTY0Zi00OGQyLWI0MTktMzM5OWFlNjJjODgwOjBmMTAzMDBiLTRkMTMtNGVhZS04NTIzLTVhZWE5ZjEyNDk0ZQ=="
    client_secret = "0f10300b-4d13-4eae-8523-5aea9f12494e"
    client_ID = "cc2161cd-e64f-48d2-b419-3399ae62c880"

    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

    llm = LlmGigaChat(Authorization_Key)
    retriver_data = ["""Объекты VectorStore не являются Runnable-объектами и поэтому их нельзя использовать в LCEL-цепочках напрямую.
                        В то же время ретриверы GigaChain (Retrievers) — являются экземплярами Runnable, поэтому они реализуют стандартный набор методов (например, синхронные и асинхронные операции invoke и batch) и предназначены для включения в цепочки LCEL.
                        Вы можете самостоятельно создать ретривер, не прибегая к классу Retriever. Для этого нужно выбрать метод, который будет использоваться для извлечения документов, и создать Runnable. Пример ниже показывает, как создать ретривер, который использует метод similarity_search, на основе Runnable:"""]

    res =llm.get_answer("Кто такие тигры?", retriver_data)
    print(res)
    exit()
