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


    def insert_record(self, parametrs: dict) -> None:
        """
        Принимает аргумент словарь с ключами 'file_name' и 'summary_vector'
        """
        file_name = parametrs["file_name"]
        summary_vector = parametrs["summary_vector"]
        self.client.command(f"INSERT INTO {self.table_name} (file_name, summary_vector) VALUES ('{file_name}', {summary_vector})")

    def delete_table(self) -> None:
        self.client.command(f"DROP TABLE IF EXISTS {self.table_name}")

    def delete_record(self) -> None:
        """Не реализована"""
        pass

    def hnsw_search(self, vector: list) -> clickhouse_connect.driver.client.QuerySummary:
        return self.client.command(f"SELECT file_name FROM {self.table_name}"
                                   f" ORDER BY cosineDistance (summary_vector, {vector}) LIMIT 4")

    def info(self) -> clickhouse_connect.driver.client.QueryResult:
        return self.client.query(f"SELECT * FROM {self.table_name}")


client = clickhouse_connect.get_client(host="localhost", username="default", password="123")

if __name__ == "__main__":
    from numpy import zeros

    test_vector = zeros([1024])
    handler = SummarizationHandler(client)
    result = handler.hnsw_search(test_vector.tolist())

    print(result)
