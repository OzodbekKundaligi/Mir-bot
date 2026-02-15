import argparse
import os
import sqlite3
from typing import Any

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument


def ensure_indexes(db: Any) -> None:
    db.admins.create_index("tg_id", unique=True)
    db.users.create_index("tg_id", unique=True)
    db.required_channels.create_index("channel_ref", unique=True)
    db.required_channels.create_index([("is_active", ASCENDING), ("created_at", DESCENDING)])
    db.movies.create_index("code", unique=True)
    db.movies.create_index([("created_at", DESCENDING)])
    db.serials.create_index("code", unique=True)
    db.serials.create_index([("created_at", DESCENDING)])
    db.serial_episodes.create_index(
        [("serial_id", ASCENDING), ("episode_number", ASCENDING)],
        unique=True,
    )
    db.requests_log.create_index([("created_at", DESCENDING)])


def has_table(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return bool(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate data from SQLite (kino_bot.db) to MongoDB."
    )
    parser.add_argument("--sqlite-path", default="kino_bot.db", help="Path to SQLite database file.")
    parser.add_argument("--mongo-uri", default="", help="MongoDB URI. If empty, uses MONGODB_URI from .env.")
    parser.add_argument("--mongo-db", default="", help="MongoDB database name. If empty, uses MONGODB_DB from .env.")
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop target MongoDB collections before migration.",
    )
    args = parser.parse_args()

    load_dotenv()
    mongo_uri = (args.mongo_uri or os.getenv("MONGODB_URI", "")).strip()
    mongo_db_name = (args.mongo_db or os.getenv("MONGODB_DB", "kino_bot")).strip() or "kino_bot"

    if not mongo_uri:
        raise RuntimeError("MONGODB_URI topilmadi. --mongo-uri yoki .env orqali bering.")

    conn = sqlite3.connect(args.sqlite_path)
    conn.row_factory = sqlite3.Row

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    db = client[mongo_db_name]

    collections = [
        "admins",
        "users",
        "required_channels",
        "movies",
        "serials",
        "serial_episodes",
        "requests_log",
    ]
    if args.drop_existing:
        for name in collections:
            db[name].drop()

    ensure_indexes(db)

    migrated_counts: dict[str, int] = {}

    if has_table(conn, "admins"):
        rows = conn.execute("SELECT tg_id, added_at FROM admins").fetchall()
        for row in rows:
            db.admins.update_one(
                {"tg_id": int(row["tg_id"])},
                {"$set": {"tg_id": int(row["tg_id"]), "added_at": row["added_at"]}},
                upsert=True,
            )
        migrated_counts["admins"] = len(rows)

    if has_table(conn, "users"):
        rows = conn.execute("SELECT tg_id, full_name, joined_at FROM users").fetchall()
        for row in rows:
            db.users.update_one(
                {"tg_id": int(row["tg_id"])},
                {
                    "$set": {"full_name": row["full_name"]},
                    "$setOnInsert": {"joined_at": row["joined_at"]},
                },
                upsert=True,
            )
        migrated_counts["users"] = len(rows)

    if has_table(conn, "required_channels"):
        rows = conn.execute(
            "SELECT channel_ref, title, join_link, is_active, created_at FROM required_channels"
        ).fetchall()
        for row in rows:
            db.required_channels.update_one(
                {"channel_ref": row["channel_ref"]},
                {
                    "$set": {
                        "channel_ref": row["channel_ref"],
                        "title": row["title"],
                        "join_link": row["join_link"],
                        "is_active": bool(row["is_active"]),
                        "created_at": row["created_at"],
                    }
                },
                upsert=True,
            )
        migrated_counts["required_channels"] = len(rows)

    if has_table(conn, "movies"):
        rows = conn.execute(
            "SELECT code, title, description, media_type, file_id, created_at FROM movies"
        ).fetchall()
        for row in rows:
            db.movies.update_one(
                {"code": row["code"]},
                {
                    "$set": {
                        "code": row["code"],
                        "title": row["title"],
                        "description": row["description"] or "",
                        "media_type": row["media_type"],
                        "file_id": row["file_id"],
                        "created_at": row["created_at"],
                    }
                },
                upsert=True,
            )
        migrated_counts["movies"] = len(rows)

    serial_id_map: dict[int, str] = {}
    if has_table(conn, "serials"):
        rows = conn.execute(
            "SELECT id, code, title, description, created_at FROM serials"
        ).fetchall()
        for row in rows:
            serial_doc = db.serials.find_one_and_update(
                {"code": row["code"]},
                {
                    "$set": {
                        "code": row["code"],
                        "title": row["title"],
                        "description": row["description"] or "",
                        "created_at": row["created_at"],
                    }
                },
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
            serial_id_map[int(row["id"])] = str(serial_doc["_id"])
        migrated_counts["serials"] = len(rows)

    if has_table(conn, "serial_episodes"):
        rows = conn.execute(
            "SELECT serial_id, episode_number, media_type, file_id, created_at FROM serial_episodes"
        ).fetchall()
        inserted = 0
        skipped = 0
        for row in rows:
            mapped_serial_id = serial_id_map.get(int(row["serial_id"]))
            if not mapped_serial_id:
                skipped += 1
                continue
            db.serial_episodes.update_one(
                {
                    "serial_id": mapped_serial_id,
                    "episode_number": int(row["episode_number"]),
                },
                {
                    "$set": {
                        "serial_id": mapped_serial_id,
                        "episode_number": int(row["episode_number"]),
                        "media_type": row["media_type"],
                        "file_id": row["file_id"],
                        "created_at": row["created_at"],
                    }
                },
                upsert=True,
            )
            inserted += 1
        migrated_counts["serial_episodes"] = inserted
        migrated_counts["serial_episodes_skipped"] = skipped

    if has_table(conn, "requests_log"):
        rows = conn.execute(
            "SELECT user_tg_id, movie_code, result, created_at FROM requests_log"
        ).fetchall()
        docs = [
            {
                "user_tg_id": int(row["user_tg_id"]),
                "movie_code": row["movie_code"],
                "result": row["result"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
        if docs:
            db.requests_log.insert_many(docs)
        migrated_counts["requests_log"] = len(docs)

    print("Migration completed.")
    for key in sorted(migrated_counts.keys()):
        print(f"- {key}: {migrated_counts[key]}")


if __name__ == "__main__":
    main()
