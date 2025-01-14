import clickhouse_connect
from abc import ABC, abstractmethod


class TableHandler(ABC):
    def __init__(self, client: clickhouse_connect.driver.Client):
        pass

    @abstractmethod
    def insert_record(self, parameters: dict):
        pass

    @abstractmethod
    def delete_record(self):
        pass

    @abstractmethod
    def delete_table(self):
        pass

    @abstractmethod
    def _create_table(self):
        pass


class SentenceHandler(TableHandler):
    def __init__(self, client: clickhouse_connect.driver.Client) -> None:
        super().__init__(client)
        self.client = client
        self.table_name = "sentences"
        self._create_table()

    def _create_table(self):
        self.client.command("SET allow_experimental_vector_similarity_index = 1;")
        self.client.command(f"CREATE TABLE IF NOT EXISTS {self.table_name}"
                            f" ("
                            f" created_at DateTime DEFAULT now(),"
                            f" master_name String, summary_vector Array(Float32), sentence String,"
                            f" INDEX idx_HNSW summary_vector TYPE vector_similarity('hnsw', 'L2Distance')"
                            f" )"
                            f" ENGINE MergeTree"
                            f" ORDER BY created_at")

    def insert_record(self, parameters: dict):
        """
        Принимает аргумент словарь с ключами 'master_name', 'summary_vector' и 'sentence'
        """
        file_name = parameters["master_name"]
        summary_vector = parameters["summary_vector"]
        sentence = parameters["sentence"]
        self.client.command(
            f" INSERT INTO {self.table_name} (master_name, summary_vector,"
            f" sentence) VALUES ('{file_name}', {summary_vector}, '{sentence}')")

    def delete_table(self) -> None:
        self.client.command(f"DROP TABLE IF EXISTS {self.table_name}")

    def delete_record(self) -> None:
        """Не реализована"""
        pass

    def hnsw_search(self, vector: list, file_path: list) -> clickhouse_connect.driver.client.QuerySummary:
        return self.client.query(f"select sentence from summarization "
                                 f"left join sentences on summarization.file_name == sentences.master_name "
                                 f"where sentences.master_name == '{file_path}' "
                                 f"AND cosineDistance (sentences.summary_vector, {vector}) < 0.4 "
                                 f"ORDER BY cosineDistance (sentences.summary_vector, {vector}) ASC LIMIT 10").result_rows

    def info(self) -> clickhouse_connect.driver.client.QueryResult:
        return self.client.query(f"SELECT * FROM {self.table_name}")


class SummarizationHandler(TableHandler):
    def __init__(self, client: clickhouse_connect.driver.Client) -> None:
        super().__init__(client)
        self.client = client
        self.table_name = "summarization"
        self._create_table()

    def _create_table(self) -> None:
        self.client.command("SET allow_experimental_vector_similarity_index = 1;")
        self.client.command(f"CREATE TABLE IF NOT EXISTS {self.table_name}"
                            f" ("
                            f" created_at DateTime DEFAULT now(),"
                            f" file_name String, summary_vector Array(Float32),"
                            f" INDEX idx_HNSW summary_vector TYPE vector_similarity('hnsw', 'L2Distance')"
                            f" )"
                            f" ENGINE MergeTree"
                            f" ORDER BY created_at")


    def insert_record(self, parameters: dict) -> None:
        """
        Принимает аргумент словарь с ключами 'file_name' и 'summary_vector'
        """
        file_name = parameters["file_name"]
        summary_vector = parameters["summary_vector"]
        self.client.command(f"INSERT INTO {self.table_name} (file_name, summary_vector) VALUES ('{file_name}', {summary_vector})")

    def delete_table(self) -> None:
        self.client.command(f"DROP TABLE IF EXISTS {self.table_name}")

    def delete_record(self) -> None:
        """Не реализована"""
        pass

    def hnsw_search(self, vector: list) -> clickhouse_connect.driver.client.QuerySummary:
        return self.client.query(f"select file_name from summarization "
                                 f"ORDER BY cosineDistance (summary_vector, {vector}) ASC LIMIT 3").result_rows

    def info(self) -> clickhouse_connect.driver.client.QueryResult:
        return self.client.query(f"SELECT * FROM {self.table_name}")


client = clickhouse_connect.get_client(host="localhost", username="default", password="123")

if __name__ == "__main__":
    # import numpy as np
    # import faiss
    #
    # dim = 512  # рассмотрим произвольные векторы размерности 512
    # nb = 10000  # количество векторов в индексе
    # nq = 5  # количество векторов в выборке для поиска
    # np.random.seed(228)
    # vectors = np.random.random((nb, dim)).astype('float32')
    # query = np.random.random((nq, dim)).astype('float32')
    #
    # index = faiss.IndexFlatL2(dim)
    # print(index.ntotal)  # пока индекс пустой
    # index.add(vectors)
    # print(index.ntotal)  # теперь в нем 10 000 векторов
    #
    # topn = 7
    # D, I = index.search(query, topn)  # Возвращает результат: Distances, Indices
    # print(I)
    # print(D)
    exit()
