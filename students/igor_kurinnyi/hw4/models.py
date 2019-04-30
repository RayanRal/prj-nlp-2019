from __future__ import annotations

from peewee import SqliteDatabase
from peewee import Model
from peewee import CharField, DateField, IntegerField, TextField
from peewee import ForeignKeyField

import config
from utils import jaccard_text_similarity


database = SqliteDatabase('books.db')


class BaseModel(Model):
    class Meta:
        database = database


class Author(BaseModel):
    name = CharField()
    birthday = DateField()

    def __str__(self):
        return f'{self.name} : {self.birthday}'


class WikiText(BaseModel):
    person = ForeignKeyField(Author, backref='wikipage')
    link = CharField()
    wiki_id = IntegerField()
    text = TextField()

    def __str__(self):
        return f'{self.person}\n{self.text}'


class FactSource(BaseModel):
    stype = CharField()

    def __str__(self):
        return f'{self.stype}'


class Book(BaseModel):
    author = ForeignKeyField(Author, backref='books')
    fact_source = ForeignKeyField(FactSource, backref='facts')
    title = CharField()
    year = IntegerField(null=True)

    def similarity(self, other: Book) -> float:
        s1 = config.YEAR_WEIGHT if self.year == other.year else 0
        s2 = config.TITLE_WEIGHT * jaccard_text_similarity(self.title.lower(), other.title.lower(), n=config.TITLE_NGRAMS)
        return s1 + s2

    def __str__(self):
        return f'{self.title} ({self.year})'


def setup_db():
    database.connect()
    database.drop_tables([Author, Book, FactSource, WikiText])
    database.create_tables([Author, Book, FactSource, WikiText])
    FactSource.create(stype='wikitext')
    FactSource.create(stype='dbpedia')
    database.commit()
    database.close()


if __name__ == '__main__':
    # setup_db()
    pass
