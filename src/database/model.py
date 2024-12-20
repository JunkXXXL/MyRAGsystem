from sqlalchemy import create_engine, Column, MetaData

from clickhouse_sqlalchemy import (
    Table, make_session, get_declarative_base, types, engines
)
#https://clickhouse-sqlalchemy.readthedocs.io/en/latest/connection.html#connection

uri = 'clickhouse+native://default:123@localhost:9000'

engine = create_engine(uri)
metadata = MetaData(bind=engine)
Base = get_declarative_base(metadata=metadata)


class New_table(Base):
    id = Column(types.UInt32, primary_key=True)
    summary = Column(types.String)

    __table_args__ = (
        engines.Memory(),)


    def __repr__(self):
        return str(self.id)


def create_tables():
    session = make_session(engine)

    New_table.__table__.drop()
    try:
        New_table.__table__.create()
    except:
        pass

    return session


# def insert_record(summary: str):
#     rates = [
#         {'id': 0, 'summary': summary}
#     ]
#     session.execute(model.New_table.__table__.insert(), {"id": 3, "summary": summary})
#
# def info():
#     info = session.query(model.New_table.__table__).all()
#     return info


if __name__ == "__main__":
    create_tables()