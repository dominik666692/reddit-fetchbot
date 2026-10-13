"""Reddit link -> the post's media (images, GIFs, videos).

Read-only. For a post link that a user shares, this makes one request to the
official Reddit Data API (OAuth, app-only token) to read that single post,
then downloads its media from Reddit's media servers (i.redd.it / v.redd.it).
It never posts, comments, votes or browses subreddits.

Handles: post links (/comments/, /gallery/, redd.it short links), app share
links (/r/x/s/AbC123), direct i.redd.it / preview.redd.it files, v.redd.it
videos, galleries, crossposts, and images embedded in text posts.
"""

from __future__ import annotations

import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

import aiohttp

from media_common import (
    MIME_EXT,
    Link,
    clean_url,
    DashSource,
    FileSource,
    Http,
    MediaError,
    MediaItem,
    ext_of,
    slug,
)

log = logging.getLogger("reddit")

WWW = "https://www.reddit.com"
OAUTH = "https://oauth.reddit.com"
TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
# Reddit asks for "<platform>:<app id>:<version> (by /u/<username>)"; set
# reddit_user_agent in config.toml to include your username.
API_UA = "linux:chat-link-preview:1.0"

_REDDIT_HOST = re.compile(r"^(?:[\w-]+\.)?reddit\.com$|^redd\.it$|^(?:i|v|preview)\.redd\.it$", re.I)
_POST_ID = re.compile(r"/(?:comments|gallery)/([a-z0-9]{4,12})(?:[/?#]|$)", re.I)
_SHARE = re.compile(r"^/(?:r|u|user)/[^/]+/s/[A-Za-z0-9]+/?$")
_IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "gif"}
_VIDEO_EXTS = {"mp4", "webm", "mov"}


def classify(url: str) -> Link | None:
    """Link kinds: post (id), share (url), file (url), vreddit (v.redd.it id)."""
    url = clean_url(url)
    try:
        p = urlparse(url)
    except ValueError:
        return None
    host = (p.hostname or "").lower()
    if not _REDDIT_HOST.match(host):
        return None
    path = p.path or "/"
    if host in ("i.redd.it", "preview.redd.it"):
        return Link("reddit", "file", url, url) if len(path) > 1 else None
    if host == "v.redd.it":
        vid = path.strip("/").split("/")[0]
        return Link("reddit", "vreddit", vid, url) if vid else None
    if host == "redd.it":
        pid = path.strip("/").split("/")[0]
        return Link("reddit", "post", pid.lower(), url) if re.fullmatch(r"[a-z0-9]{4,12}", pid, re.I) else None
    if _SHARE.match(path):
        return Link("reddit", "share", url, url)
    m = _POST_ID.search(path)
    if m:
        return Link("reddit", "post", m.group(1).lower(), url)
    return None  # subreddit / user / other pages: not a post


# --- talking to Reddit (official Data API) -----------------------------------


class RedditClient:
    def __init__(self, http: Http, cfg: dict[str, Any]):
        self.http = http
        self.client_id: str = cfg.get("reddit_client_id", "") or ""
        self.client_secret: str = cfg.get("reddit_client_secret", "") or ""
        self.api_ua: str = cfg.get("reddit_user_agent", "") or API_UA
        self._token: str | None = None
        self._token_exp = 0.0

    async def _oauth_token(self) -> str:
        """App-only OAuth token (client credentials), cached until it expires."""
        if self._token and time.time() < self._token_exp - 60:
            return self._token
        if self.client_secret:
            data = {"grant_type": "client_credentials"}
        else:
            data = {"grant_type": "https://oauth.reddit.com/grants/installed_client",
                    "device_id": "DO_NOT_TRACK_THIS_DEVICE"}
        auth = aiohttp.BasicAuth(self.client_id, self.client_secret)
        async with self.http.session.post(
            self.http.rewrite(TOKEN_URL), data=data, auth=auth, headers={"User-Agent": self.api_ua}
        ) as r:
            body = await r.json(content_type=None)
            if r.status != 200 or "access_token" not in body:
                raise MediaError(f"Reddit API login failed (HTTP {r.status}): {body}")
        self._token = body["access_token"]
        self._token_exp = time.time() + float(body.get("expires_in", 3600))
        return self._token

    async def get_post(self, post_id: str) -> dict[str, Any]:
        """One API request: GET /comments/{id} (just the post, no comment tree)."""
        if not self.client_id:
            raise MediaError(
                "Reddit post links need Reddit API credentials: set reddit_client_id / "
                "reddit_client_secret in config.toml (Reddit closed anonymous access)."
            )
        url = f"{OAUTH}/comments/{post_id}?raw_json=1&limit=1&depth=0"
        for attempt in (1, 2):
            token = await self._oauth_token()
            headers = {"Authorization": f"bearer {token}", "User-Agent": self.api_ua}
            async with self.http.session.get(self.http.rewrite(url), headers=headers) as r:
                text = await r.text()
                status = r.status
            if status != 401 or attempt == 2:
                break
            self._token = None  # expired/revoked token: log in again and retry once
        try:
            data = json.loads(text)
        except ValueError:
            raise MediaError(f"Reddit API returned HTTP {status} and no JSON")
        if isinstance(data, dict) and (data.get("error") or status != 200):
            raise MediaError(f"Reddit API refused (HTTP {status}): {data.get('reason') or data.get('message') or data.get('error')}")
        try:
            return data[0]["data"]["children"][0]["data"]
        except (KeyError, IndexError, TypeError):
            raise MediaError("unexpected Reddit response (post deleted?)")

    async def resolve_share(self, url: str) -> str:
        """App share links (/r/sub/s/AbC) are plain redirects to the real post."""
        current = url
        for _ in range(5):
            async with self.http.session.get(self.http.rewrite(current), allow_redirects=False,
                                             headers={"User-Agent": self.api_ua}) as r:
                loc = r.headers.get("Location")
                if r.status in (301, 302, 303, 307, 308) and loc:
                    current = urljoin(current, loc)
                    link = classify(current)
                    if link and link.kind == "post":
                        return link.value
                    continue
                raise MediaError(f"share link didn't redirect (HTTP {r.status})")
        raise MediaError("share link redirected too many times")

    # --- the public entry point --------------------------------------------

    async def extract(self, link: Link) -> list[MediaItem]:
        if link.kind == "file":
            return [file_item(link.url, name=f"reddit_{slug(link.url.split('/')[-1].split('.')[0])}")]
        if link.kind == "vreddit":
            vid = link.value
            return [MediaItem(
                sources=[DashSource(
                    mpd_url=f"https://v.redd.it/{vid}/DASHPlaylist.mpd",
                    hls_url=f"https://v.redd.it/{vid}/HLSPlaylist.m3u8",
                )],
                name=f"reddit_{vid}",
            )]
        post_id = link.value if link.kind == "post" else await self.resolve_share(link.url)
        post = await self.get_post(post_id)
        items = items_from_post(post)
        if not items:
            log.info("post %s has no media I can fetch (url: %s)", post_id, post.get("url"))
        return items


# --- post JSON -> items ---------------------------------------------------------


def _to_i_redd_it(url: str) -> str:
    """preview.redd.it/abc.jpg?width=..&s=.. -> i.redd.it/abc.jpg (the original)."""
    return url.split("?")[0].replace("://preview.redd.it/", "://i.redd.it/", 1)


def file_item(url: str, name: str, title: str | None = None,
              w: int | None = None, h: int | None = None,
              fallbacks: list[FileSource] | None = None) -> MediaItem:
    ext = ext_of(url)
    if ext == "gifv":  # imgur's fake gif = mp4
        url, ext = url[: url.lower().rfind(".gifv")] + ".mp4", "mp4"
    sources: list[FileSource] = []
    if "preview.redd.it" in url:
        sources.append(FileSource(_to_i_redd_it(url)))
    sources.append(FileSource(url, as_video=ext in _VIDEO_EXTS))
    sources += fallbacks or []
    return MediaItem(sources=sources, name=name, title=title, width=w, height=h,
                     gif_like=ext == "gif")


def _int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _from_reddit_video(rv: dict[str, Any], name: str, title: str | None) -> MediaItem:
    return MediaItem(
        sources=[DashSource(
            mpd_url=rv.get("dash_url"),
            hls_url=rv.get("hls_url"),
            fallback_url=rv.get("fallback_url"),
        )],
        name=name,
        title=title,
        width=_int(rv.get("width")),
        height=_int(rv.get("height")),
        duration=rv.get("duration"),
        gif_like=bool(rv.get("is_gif")),
    )


def _from_metadata(mid: str, meta: dict[str, Any], name: str, title: str | None) -> MediaItem | None:
    if meta.get("status") != "valid":
        return None
    kind = meta.get("e")
    s = meta.get("s") or {}
    w, h = _int(s.get("x")), _int(s.get("y"))
    if kind == "Image":
        ext = MIME_EXT.get(meta.get("m", ""), "") or ext_of(s.get("u", "")) or "jpg"
        srcs = [FileSource(f"https://i.redd.it/{mid}.{ext}")]
        if s.get("u"):
            srcs.append(FileSource(s["u"]))  # signed preview copy as fallback
        return MediaItem(sources=srcs, name=name, title=title, width=w, height=h)
    if kind == "AnimatedImage":
        srcs: list[Any] = []
        if s.get("gif"):
            srcs += [FileSource(_to_i_redd_it(s["gif"])), FileSource(s["gif"])]
        if s.get("mp4"):
            srcs.append(FileSource(s["mp4"], as_video=True))
        if not srcs:
            return None
        return MediaItem(sources=srcs, name=name, title=title, width=w, height=h, gif_like=True)
    if kind == "RedditVideo" and (meta.get("dashUrl") or meta.get("hlsUrl")):
        return MediaItem(
            sources=[DashSource(mpd_url=meta.get("dashUrl"), hls_url=meta.get("hlsUrl"))],
            name=name, title=title, width=w, height=h,
            gif_like=bool(meta.get("isGif")),
        )
    return None


def items_from_post(post: dict[str, Any]) -> list[MediaItem]:
    title = post.get("title")
    base = f"{slug(title, 50) or 'reddit'}_{post.get('id', '')}".strip("_")
    cross = post.get("crosspost_parent_list") or []
    media = cross[-1] if cross else post

    # 1) gallery
    gallery = (media.get("gallery_data") or {}).get("items") or []
    meta = media.get("media_metadata") or {}
    if gallery and meta:
        out = []
        for n, g in enumerate(gallery, 1):
            mid = g.get("media_id")
            item = _from_metadata(mid, meta.get(mid) or {}, f"{base}_{n}", title) if mid else None
            if item:
                out.append(item)
        return out

    # 2) video hosted on Reddit
    rv = ((media.get("secure_media") or {}).get("reddit_video")
          or (media.get("media") or {}).get("reddit_video"))
    if rv:
        return [_from_reddit_video(rv, base, title)]

    url: str = media.get("url_overridden_by_dest") or media.get("url") or ""
    host = (urlparse(url).hostname or "").lower()
    ext = ext_of(url)
    preview = (media.get("preview") or {})
    pimg = (preview.get("images") or [{}])[0]
    src = pimg.get("source") or {}
    w, h = _int(src.get("width")), _int(src.get("height"))

    # 3) direct picture / gif / video file (i.redd.it, i.imgur.com, ...)
    if host in ("i.redd.it", "preview.redd.it") or ext in _IMAGE_EXTS | _VIDEO_EXTS | {"gifv"}:
        fallbacks: list[FileSource] = []
        variants = pimg.get("variants") or {}
        if ext == "gif" and (variants.get("mp4") or {}).get("source", {}).get("url"):
            # if the real gif is too big, Reddit's mp4 version usually isn't
            fallbacks.append(FileSource(variants["mp4"]["source"]["url"], as_video=True))
        elif src.get("url") and ext in _IMAGE_EXTS:
            fallbacks.append(FileSource(src["url"]))
        return [file_item(url, base, title, w, h, fallbacks)]

    # 4) external gif Reddit transcoded for its player (imgur, giphy, ...)
    rvp = preview.get("reddit_video_preview")
    if rvp:
        return [_from_reddit_video(rvp, base, title)]

    # 5) pictures / gifs / videos embedded in a text post
    if meta:
        out = []
        for n, (mid, m) in enumerate(meta.items(), 1):
            item = _from_metadata(mid, m, f"{base}_{n}", title)
            if item:
                out.append(item)
        return out

    return []  # text post or an external link we don't handle (yet)


# --- DASH manifest ---------------------------------------------------------------


@dataclass
class Rep:
    url: str
    bandwidth: int
    height: int
    width: int


def _iso_duration(s: str | None) -> float | None:
    if not s:
        return None
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:([\d.]+)S)?", s.strip())
    if not m:
        return None
    d, h, mi, sec = (float(x) if x else 0.0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + sec


def parse_mpd(xml_text: str, mpd_url: str) -> tuple[list[Rep], list[Rep], float | None]:
    """-> (video reps best-first, audio reps best-first, duration seconds)."""
    root = ET.fromstring(xml_text)

    def base(el: ET.Element, cur: str) -> str:
        b = el.find("{*}BaseURL")
        return urljoin(cur, b.text.strip()) if b is not None and b.text else cur

    videos: list[Rep] = []
    audios: list[Rep] = []
    root_base = base(root, mpd_url)
    for period in root.findall("{*}Period"):
        pbase = base(period, root_base)
        for aset in period.findall("{*}AdaptationSet"):
            abase = base(aset, pbase)
            a_mime = (aset.get("mimeType") or aset.get("contentType") or "").lower()
            for rep in aset.findall("{*}Representation"):
                if rep.find("{*}BaseURL") is None and aset.find("{*}BaseURL") is None:
                    continue  # segmented representation: leave that to the HLS path
                mime = (rep.get("mimeType") or a_mime).lower()
                r = Rep(
                    url=base(rep, abase),
                    bandwidth=_int(rep.get("bandwidth")) or 0,
                    height=_int(rep.get("height")) or 0,
                    width=_int(rep.get("width")) or 0,
                )
                is_audio = mime.startswith("audio") or (
                    not r.height and rep.find("{*}AudioChannelConfiguration") is not None
                )
                (audios if is_audio else videos).append(r)
    videos.sort(key=lambda r: (r.height, r.bandwidth), reverse=True)
    audios.sort(key=lambda r: r.bandwidth, reverse=True)
    return videos, audios, _iso_duration(root.get("mediaPresentationDuration"))
