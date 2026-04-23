from backend.db.data_models import (
    DAOAssets,
    DAOJobEvents,
    DAOJobs,
    DAOPages,
    DAOPagesAssetsRel,
    DAOPhotobookBookmarks,
    DAOPhotobooks,
    DAOPhotobookSettings,
    DAOUsers,
)

from .assets import DALAssets
from .base import (
    AsyncPostgreSQLDAL,
    FilterOp,
    InvalidFilterFieldError,
    OrderDirection,
    safe_commit,
    safe_transaction,
)
from .job_events import DALJobEvents
from .schemas import (
    DAOAssetsCreate,
    DAOAssetsUpdate,
    DAOJobEventsCreate,
    DAOJobsCreate,
    DAOJobsUpdate,
    DAOPagesAssetsRelCreate,
    DAOPagesAssetsRelUpdate,
    DAOPagesCreate,
    DAOPagesUpdate,
    DAOPhotobookBookmarksCreate,
    DAOPhotobookBookmarksUpdate,
    DAOPhotobooksCreate,
    DAOPhotobookSettingsCreate,
    DAOPhotobookSettingsUpdate,
    DAOPhotobooksUpdate,
    DAOUsersCreate,
    DAOUsersUpdate,
)


class DALJobs(AsyncPostgreSQLDAL[DAOJobs, DAOJobsCreate, DAOJobsUpdate]):
    model = DAOJobs


class DALPages(AsyncPostgreSQLDAL[DAOPages, DAOPagesCreate, DAOPagesUpdate]):
    model = DAOPages


class DALPagesAssetsRel(
    AsyncPostgreSQLDAL[
        DAOPagesAssetsRel, DAOPagesAssetsRelCreate, DAOPagesAssetsRelUpdate
    ]
):
    model = DAOPagesAssetsRel


class DALPhotobooks(
    AsyncPostgreSQLDAL[DAOPhotobooks, DAOPhotobooksCreate, DAOPhotobooksUpdate]
):
    model = DAOPhotobooks


class DALUsers(AsyncPostgreSQLDAL[DAOUsers, DAOUsersCreate, DAOUsersUpdate]):
    model = DAOUsers


class DALPhotobookBookmarks(
    AsyncPostgreSQLDAL[
        DAOPhotobookBookmarks, DAOPhotobookBookmarksCreate, DAOPhotobookBookmarksUpdate
    ]
):
    model = DAOPhotobookBookmarks


class DALPhotobookSettings(
    AsyncPostgreSQLDAL[
        DAOPhotobookSettings, DAOPhotobookSettingsCreate, DAOPhotobookSettingsUpdate
    ]
):
    model = DAOPhotobookSettings


__all__ = [
    # DALs
    "DALAssets",
    "DALJobs",
    "DALJobEvents",
    "DALPages",
    "DALPagesAssetsRel",
    "DALPhotobooks",
    "DALPhotobookBookmarks",
    "DAOPhotobookSettings",
    # DAL base
    "AsyncPostgreSQLDAL",
    "FilterOp",
    "InvalidFilterFieldError",
    "OrderDirection",
    # ORM objects
    "DAOAssets",
    "DAOJobs",
    "DAOJobEvents",
    "DAOPages",
    "DAOPagesAssetsRel",
    "DAOPhotobooks",
    "DAOUsers",
    "DAOPhotobookBookmarks",
    "DAOPhotobookSettings",
    # Schemas
    "DAOAssetsCreate",
    "DAOAssetsUpdate",
    "DAOJobsCreate",
    "DAOJobsUpdate",
    "DAOJobEventsCreate",
    # "DAOJobEventsUpdate",   # Updating job events is not allowed, as it's an append-only log trail
    "DAOPagesCreate",
    "DAOPagesUpdate",
    "DAOPagesAssetsRelCreate",
    "DAOPagesAssetsRelUpdate",
    "DAOPhotobooksCreate",
    "DAOPhotobooksUpdate",
    "DAOUsersCreate",
    "DAOUsersUpdate",
    "DAOPhotobookBookmarksUpdate",
    "DAOPhotobookBookmarksCreate",
    "DAOPhotobookSettingsUpdate",
    "DAOPhotobookSettingsCreate",
    # Utils
    "safe_commit",
    "safe_transaction",
]
