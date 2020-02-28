# ydl1.py
from __future__ import unicode_literals
import youtube_dl
import os

local = os.path.expanduser(r"~\Downloads")
os.chdir(local)

ydl_opts = {'ignoreerrors': True,
            'format':'m4a'}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/playlist?list=PLqIq1bkXU1VsGtE3bSbu-5FxmbacaGdu7'])