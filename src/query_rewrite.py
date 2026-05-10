import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List


GENRE_ALIASES: Dict[str, str] = {
    "science fiction": "Sci-Fi",
    "sci fi": "Sci-Fi",
    "sci-fi": "Sci-Fi",
    "科幻": "Sci-Fi",
    "romance": "Romance",
    "爱情": "Romance",
    "喜剧": "Comedy",
    "comedy": "Comedy",
    "惊悚": "Thriller",
    "thriller": "Thriller",
    "恐怖": "Horror",
    "horror": "Horror",
    "动作": "Action",
    "action": "Action",
    "冒险": "Adventure",
    "adventure": "Adventure",
    "剧情": "Drama",
    "drama": "Drama",
    "犯罪": "Crime",
    "crime": "Crime",
    "悬疑": "Mystery",
    "mystery": "Mystery",
    "动画": "Animation",
    "animation": "Animation",
    "战争": "War",
    "war": "War",
    "纪录片": "Documentary",
    "documentary": "Documentary",
    "音乐": "Musical",
    "musical": "Musical",
    "西部": "Western",
    "western": "Western",
    "奇幻": "Fantasy",
    "fantasy": "Fantasy",
}

MOOD_ALIASES: Dict[str, str] = {
    "轻松": "relaxed",
    "治愈": "healing",
    "欢乐": "happy",
    "搞笑": "funny",
    "烧脑": "brainy",
    "紧张": "tense",
    "压抑": "dark",
    "悲伤": "sad",
    "温暖": "warm",
    "刺激": "exciting",
    "high energy": "exciting",
    "relaxed": "relaxed",
    "healing": "healing",
    "happy": "happy",
    "funny": "funny",
    "brainy": "brainy",
    "tense": "tense",
    "dark": "dark",
    "sad": "sad",
    "warm": "warm",
    "exciting": "exciting",
}

POPULARITY_WORDS = ["热门", "高分", "经典", "口碑", "热门电影", "best", "popular", "评分高", "高分片"]


@dataclass
class StructuredQuery:
    original_query: str
    rewritten_query: str
    intent: str
    route_hint: str
    genres: List[str]
    moods: List[str]
    years: List[int]
    reference_titles: List[str]
    popularity_request: bool
    keywords: List[str]
    confidence: float
    needs_clarification: bool
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QueryRewriteEngine:
    def _normalize_query(self, query: str) -> str:
        return (query or "").strip()

    def _extract_genres(self, query: str) -> List[str]:
        normalized_query = query.lower()
        genres: List[str] = []
        for alias, canonical in GENRE_ALIASES.items():
            if alias in normalized_query and canonical not in genres:
                genres.append(canonical)
        return genres

    def _extract_moods(self, query: str) -> List[str]:
        normalized_query = query.lower()
        moods: List[str] = []
        for alias, canonical in MOOD_ALIASES.items():
            if alias in normalized_query and canonical not in moods:
                moods.append(canonical)
        return moods

    def _extract_years(self, query: str) -> List[int]:
        years: List[int] = []
        for match in re.findall(r"(19\d{2}|20\d{2})", query):
            try:
                years.append(int(match))
            except ValueError:
                continue
        return sorted(set(years))

    def _extract_reference_titles(self, query: str) -> List[str]:
        candidates: List[str] = []
        quote_patterns = [r"《([^》]+)》", r"“([^”]+)”", r'"([^"]+)"', r"'([^']+)'"]
        for pattern in quote_patterns:
            for match in re.findall(pattern, query):
                title = str(match).strip()
                if title and title not in candidates:
                    candidates.append(title)
        return candidates

    def _extract_keywords(self, query: str) -> List[str]:
        tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", query.lower())
        keywords: List[str] = []
        for token in tokens:
            if len(token) < 2:
                continue
            if token not in keywords:
                keywords.append(token)
        return keywords[:12]

    def _is_popularity_request(self, query: str) -> bool:
        normalized_query = query.lower()
        return any(word in normalized_query for word in POPULARITY_WORDS)

    def rewrite(self, query: str) -> StructuredQuery:
        original_query = query or ""
        normalized_query = self._normalize_query(original_query)

        genres = self._extract_genres(normalized_query)
        moods = self._extract_moods(normalized_query)
        years = self._extract_years(normalized_query)
        reference_titles = self._extract_reference_titles(normalized_query)
        popularity_request = self._is_popularity_request(normalized_query)
        keywords = self._extract_keywords(normalized_query)

        explicit_slots = len(genres) + len(years) + len(reference_titles)
        if popularity_request:
            explicit_slots += 1

        if not normalized_query:
            intent = "cold_start"
            route_hint = "cold_start"
            needs_clarification = False
            confidence = 0.0
            rationale = "空查询，直接交给冷启动兜底。"
        elif explicit_slots > 0:
            intent = "constrained"
            route_hint = "filtered"
            needs_clarification = False
            confidence = min(1.0, 0.25 + 0.12 * explicit_slots)
            rationale = "query 中识别到显式约束，适合先过滤再排序。"
        elif moods:
            intent = "mood_based"
            route_hint = "personalized"
            needs_clarification = False
            confidence = 0.45
            rationale = "query 主要表达情绪/氛围偏好，适合做语义重写后进入个性化推荐。"
        else:
            intent = "open_ended"
            route_hint = "personalized"
            needs_clarification = False
            confidence = 0.25 if len(normalized_query) < 8 else 0.4
            rationale = "query 较模糊，保留给个性化推荐层结合用户记忆处理。"

        segments = ["电影需求"]
        if genres:
            segments.append("类型: " + ", ".join(genres))
        if moods:
            segments.append("情绪: " + ", ".join(moods))
        if years:
            segments.append("年份: " + ", ".join(str(year) for year in years))
        if reference_titles:
            segments.append("参考电影: " + ", ".join(reference_titles))
        if popularity_request:
            segments.append("偏好: 高分/热门")
        if not genres and not moods and not years and not reference_titles and not popularity_request:
            segments.append(normalized_query)

        rewritten_query = " | ".join(segments)
        return StructuredQuery(
            original_query=original_query,
            rewritten_query=rewritten_query,
            intent=intent,
            route_hint=route_hint,
            genres=genres,
            moods=moods,
            years=years,
            reference_titles=reference_titles,
            popularity_request=popularity_request,
            keywords=keywords,
            confidence=confidence,
            needs_clarification=needs_clarification,
            rationale=rationale,
        )