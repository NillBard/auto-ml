from db.session import engine
import models


def initdb():
    models.base.Base.metadata.create_all(engine)
