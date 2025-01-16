from abc import ABC, abstractmethod

class TextSplitter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def split(self, text: str):
        pass

class StandartSplitter(TextSplitter):
    def __init__(self):
        super().__init__()

    def split(self, text: str) -> list:
        """
        :param text: Принимает монолитный текст, чтобы разбить его на предложения
        :return: Вернёт list, состоящий из предложений текста
        """
        text = text.replace('!', '.')
        text = text.replace('?', '.')
        return text.split('.')


if __name__ == "__main__":
    from pathlib import Path
    p = Path(r"D:\Users\aleksandr.kovalev\Desktop\Andrey_stazher\MyRAG\MyRAGsystem\dino.txt")
    n = p.read_text(encoding='utf-8')
    print(n)
