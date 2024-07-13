import locale
import pytz

from datetime import datetime


def get_time_string(tz = 'Europe/Moscow'):
    locale.setlocale(locale.LC_ALL, ('ru_RU', 'UTF-8'))
    dt = datetime.now(pytz.timezone(tz))
    return dt.strftime(" ### Текущее время: %A, %d %B, %Yг. %H:%M")
