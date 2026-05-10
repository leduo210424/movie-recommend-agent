import pandas as pd
import os

DATA_DIR = "E:/movie-recommend-agent/data/raw/ml-100k" # 你的数据路径


def load_ratings():
    """加载评分数据 u.data"""
    ratings = pd.read_csv(
        os.path.join(DATA_DIR, "u.data"),
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1"
    )
    return ratings


def load_movies():
    """加载电影数据 u.item"""
    genre_cols = [
        "unknown", "Action", "Adventure", "Animation", "Children's",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
        "Sci-Fi", "Thriller", "War", "Western"
    ]

    columns = [
        "movie_id", "title", "release_date", "video_release_date",
        "imdb_url"
    ] + genre_cols

    movies = pd.read_csv(
        os.path.join(DATA_DIR, "u.item"),
        sep="|",
        names=columns,
        encoding="latin-1"
    )

    # 提取年份
    movies["release_year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)

    # 转换 genres
    def extract_genres(row):
        return [g for g in genre_cols if row[g] == 1]

    movies["genres"] = movies.apply(extract_genres, axis=1).apply(tuple)

    return movies[["movie_id", "title", "release_year", "genres"]]


def load_users():
    """加载用户数据 u.user"""
    users = pd.read_csv(
        os.path.join(DATA_DIR, "u.user"),
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
        encoding="latin-1"
    )
    return users


def merge_data():
    """合并所有数据"""
    ratings = load_ratings()
    movies = load_movies()
    users = load_users()

    # 合并
    df = ratings.merge(movies, on="movie_id", how="left")
    df = df.merge(users, on="user_id", how="left")

    # 构造 user_profile 字段（为后续 Agent 做准备）
    df["user_profile"] = df.apply(lambda x: {
        "age": x["age"],
        "gender": x["gender"],
        "occupation": x["occupation"]
    }, axis=1)

    return df


def save_outputs(df):
    """保存多种格式"""
    os.makedirs("data/processed", exist_ok=True)

    # 1️⃣ 主数据（推荐用）
    df.to_json("data/processed/full_data.json", orient="records", lines=True)

    # 2️⃣ 用户历史（给推荐系统用）
    user_history = df.groupby("user_id").apply(
        lambda x: x[["movie_id", "rating", "timestamp"]].to_dict("records")
    ).reset_index(name="history")

    user_history.to_json("data/processed/user_history.json", orient="records", lines=True)

    # 3️⃣ 电影表（给RAG用）
    movies = df[["movie_id", "title", "genres", "release_year"]].drop_duplicates()
    movies.to_json("data/processed/movies.json", orient="records", lines=True)

    print("✅ 数据处理完成，输出在 data/processed/ 目录")


if __name__ == "__main__":
    df = merge_data()
    print("数据样例：")
    print(df.head())

    save_outputs(df)