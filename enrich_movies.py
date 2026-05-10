"""
一次性脚本：从 OMDb API 拉取电影剧情摘要，丰富 movies.json。

用法：
    python enrich_movies.py                           # 处理全部 1682 部电影
    python enrich_movies.py --sample 5                # 先试 5 部验证
    python enrich_movies.py --overwrite               # 强制覆盖已有 overview_en

OMDb 免费 API Key 注册: http://www.omdbapi.com/apikey.aspx
免费 tier: 1000 次/天。1682 部电影可分两天跑完，或升级付费。

输出：覆盖 data/processed/movies.json 的 overview_en 字段。
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

OMDB_URL = "http://www.omdbapi.com/"
WIKI_URL = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "MovieRecAgent/1.0 (educational project)"}
# 当日 OMDb 限额耗尽后自动降级 Wikipedia
_omdb_exhausted = False


def load_movies(path: str | Path) -> List[Dict[str, Any]]:
    records = []
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def save_movies(path: str | Path, movies: List[Dict[str, Any]]):
    Path(path).write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in movies) + "\n",
        encoding="utf-8",
    )


def clean_title(title: str) -> str:
    """去掉年份后缀，并处理 ', The' 后缀: 'Matrix, The (1999)' → 'The Matrix'"""
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()
    # 转换 ", The" / ", A" / ", An" 后缀
    title = re.sub(r"^(.*),\s*(The|A|An)$", r"\2 \1", title, flags=re.IGNORECASE)
    return title


def get_wikipedia_plot(title: str, year: Optional[int]) -> Optional[str]:
    """从 Wikipedia 获取剧情摘要（免费，不限量）"""
    search_queries = [f'intitle:"{title}"']
    if year:
        search_queries.append(f'intitle:"{title}" {year} film')

    # 尝试多种搜索词
    pages = []
    for sq in search_queries[:2]:
        try:
            resp = requests.get(WIKI_URL, params={
                "action": "query", "format": "json",
                "list": "search", "srsearch": sq, "srlimit": 5,
            }, headers=WIKI_HEADERS, timeout=15)
            if resp.status_code == 200:
                pages = resp.json().get("query", {}).get("search", [])
                if pages:
                    break
        except Exception:
            continue

    if not pages:
        return None

    # 选最佳页面：排除专辑/歌曲/原声带，优先选含 "(film)" 或年份的
    best = None
    for p in pages:
        ptitle = p["title"]
        plower = ptitle.lower()
        if any(x in plower for x in ["album", "song:", "soundtrack"]):
            continue
        if year and str(year) in ptitle:
            best = ptitle
            break
        if "(film)" in plower or "film)" in plower:
            best = ptitle
            break
        if best is None:
            best = ptitle

    if not best:
        return None

    # 获取摘要
    try:
        resp = requests.get(WIKI_URL, params={
            "action": "query", "format": "json",
            "prop": "extracts", "exintro": 1, "explaintext": 1,
            "titles": best,
        }, headers=WIKI_HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        for pid, page in data.get("query", {}).get("pages", {}).items():
            if pid != "-1":
                extract = page.get("extract", "")
                if extract:
                    return extract[:500]
    except Exception:
        return None

    return None


def get_omdb_plot(api_key: str, title: str, year: Optional[int]) -> tuple:
    """通过 OMDb API 获取剧情摘要。返回 (plot_str_or_None, error_msg_or_None)"""
    params = {
        "apikey": api_key,
        "t": title,
        "plot": "short",
        "r": "json",
    }
    if year:
        params["y"] = str(year)

    try:
        resp = requests.get(OMDB_URL, params=params, timeout=15)
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        data = resp.json()
        if data.get("Response") == "True":
            return data.get("Plot", ""), None
        else:
            return None, data.get("Error", "Unknown OMDb error")
    except Exception as e:
        return None, str(e)


def enrich_movies(
    movies: List[Dict],
    api_key: str,
    sample: int = 0,
    overwrite: bool = False,
    output_path: str = None,
) -> List[Dict]:
    enriched = []
    total = len(movies) if not sample else min(sample, len(movies))
    global _omdb_exhausted
    found = 0
    skipped = 0
    overwritten = 0
    last_save = 0

    for i, movie in enumerate(movies):
        if sample and i >= sample:
            enriched.extend(movies[i:])
            break

        title_raw = movie.get("title", "")
        title = clean_title(title_raw)
        year = movie.get("release_year")
        year_int = int(year) if year else None

        # 如果已有 overview_en 且未要求覆盖，跳过
        existing = movie.get("overview_en", "")
        if existing and not overwrite:
            print(f"[{i+1}/{total}] {title} ({year_int or '?'}) ... SKIP (exists)")
            skipped += 1
            enriched.append(movie)
            continue

        print(f"[{i + 1}/{total}] {title} ({year_int or '?'}) ...", end=" ", flush=True)

        overview, err = None, None

        # 优先用 OMDb（质量更好），限额耗尽后降级 Wikipedia
        if not _omdb_exhausted and api_key:
            overview, err = get_omdb_plot(api_key, title, year_int)
            if err and ("limit" in err.lower() or "401" in str(err)):
                _omdb_exhausted = True
                print("(OMDb limit, fallback Wiki)", end=" ")
                overview, err = None, None  # 重置，走 Wikipedia

        # Wikipedia 回退
        if not overview and (not api_key or _omdb_exhausted):
            overview = get_wikipedia_plot(title, year_int)
            if overview:
                err = None

        # 还找不到的话试去掉前缀
        if not overview:
            alt_title = re.sub(r"^(The|A|An)\s+", "", title, flags=re.IGNORECASE)
            if alt_title != title:
                if not _omdb_exhausted and api_key:
                    overview, err = get_omdb_plot(api_key, alt_title, year_int)
                if not overview:
                    overview = get_wikipedia_plot(alt_title, year_int)

        if overview:
            tag = "OVERWRITE" if existing else "OK"
            print(f"{tag} ({len(overview)} chars)")
            found += 1
            if existing:
                overwritten += 1
        else:
            err_msg = f" ({err})" if err else ""
            print(f"NOT FOUND{err_msg}")
            skipped += 1

        enriched.append({
            **movie,
            "overview_en": overview or existing or "",
        })

        # OMDb 免费 tier 限速 1000/天，约 0.86s 间隔即可
        time.sleep(1.0)

        # 每 20 条增量保存，避免中断丢失进度
        if output_path and len(enriched) - last_save >= 20:
            save_movies(output_path, enriched + list(movies[len(enriched):]))
            last_save = len(enriched)

    if output_path:
        save_movies(output_path, enriched + list(movies[len(enriched):]))

    print(f"\nDone: {found} new/enriched, {overwritten} overwritten, {skipped} skipped, {total} total")
    return enriched


def main():
    parser = argparse.ArgumentParser(description="Enrich movies.json with OMDb plot summaries")
    parser.add_argument("--input", default="data/processed/movies.json")
    parser.add_argument("--output", default="data/processed/movies.json")
    parser.add_argument("--omdb-key", default="",
                        help="OMDb API key (or set OMDB_API_KEY env var)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Process only N movies for testing")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing overview_en fields")
    args = parser.parse_args()

    import os
    api_key = args.omdb_key or os.getenv("OMDB_API_KEY", "")
    if not api_key:
        print("请获取 OMDb API Key: http://www.omdbapi.com/apikey.aspx")
        print("然后: set OMDB_API_KEY=your_key")
        print("或: --omdb-key your_key")
        return

    movies = load_movies(args.input)
    # 统计当前状态
    has_plot = sum(1 for m in movies if m.get("overview_en"))
    print(f"Loaded {len(movies)} movies ({has_plot} with existing plot) from {args.input}")

    enriched = enrich_movies(movies, api_key, sample=args.sample, overwrite=args.overwrite,
                              output_path=args.output)

    print(f"Final save to {args.output}")


if __name__ == "__main__":
    main()
