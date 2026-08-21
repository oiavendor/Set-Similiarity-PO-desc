import datetime
import dateutil.relativedelta
import typing

def Today() -> datetime.datetime:
    return datetime.datetime.today()

def DateOffset(toOffset: datetime.datetime, offsetUnit: typing.Literal["days", "months", "years"], offsetAmount: int) -> datetime.datetime:
    return toOffset + dateutil.relativedelta.relativedelta(**{offsetUnit: offsetAmount})