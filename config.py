import os
import logging
from pathlib import Path
from pydantic import BaseModel
from pydantic import BaseSettings
from collections import namedtuple


def get_project_root() -> Path:
    """"""
    # Path(__file__).parent.parent
    return Path(__file__).parent


# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = get_project_root()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', )

logger = logging.getLogger()
logger.setLevel(logging.INFO)


ROW = namedtuple("ROW", "SysID, ID, Cluster, ParentModuleID, ParentID, ParentPubList, "
                        "ChildBlockModuleID, ChildBlockID, ModuleID, Topic, Subtopic, DocName, ShortAnswerText")


class Etalon(BaseModel):
    """Схема данных для эталона (без ЭталонИД)"""
    TemplateId: int
    Text: str
    LemmText: str
    SysId: int
    ModuleId: int
    Pubs: list[int]


class TextsDeleteSample(BaseModel):
    """Схема данных для удаления данных по тексту из Индекса"""
    Index: str
    Texts: list[str]
    FieldName: str
    Score: float


class Settings(BaseSettings):
    """Base settings object to inherit from."""

    class Config:
        # print(os.path.join(ROOT_DIR, ".env"))
        env_file = os.path.join(PROJECT_ROOT_DIR, ".env")
        env_file_encoding = "utf-8"
