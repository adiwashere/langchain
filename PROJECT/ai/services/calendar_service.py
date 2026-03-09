import datetime
from googleapiclient.discovery import build

def create_event(calendar_service, title, date, time, duration):

    start_dt = datetime.datetime.fromisoformat(f"{date}T{time}:00")
    end_dt = start_dt + datetime.timedelta(minutes=int(duration))

    event = {
        "summary": title,
        "start": {
            "dateTime": start_dt.isoformat(),
            "timeZone": "Asia/Kolkata"
        },
        "end": {
            "dateTime": end_dt.isoformat(),
            "timeZone": "Asia/Kolkata"
        },
    }

    calendar_service.events().insert(
        calendarId="primary",
        body=event
    ).execute()