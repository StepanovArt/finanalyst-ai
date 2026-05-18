"""
Create Qdrant collection with hybrid search schema.

Usage:
    # Start Qdrant first:
    docker-compose up qdrant -d

    # Then run:
    uv run --extra data python scripts/setup_qdrant.py
    uv run --extra data python scripts/setup_qdrant.py --recreate
"""

import argparse

from app.rag.vector_store import QDRANT_URL, collection_info, create_collection, get_client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate collection")
    parser.add_argument("--url", default=QDRANT_URL, help="Qdrant URL")
    args = parser.parse_args()

    print(f"Connecting to Qdrant at {args.url}...")
    client = get_client(args.url)

    create_collection(client, recreate=args.recreate)

    info = collection_info(client)
    print(f"\nCollection info:")
    print(f"  status:        {info['status']}")
    print(f"  points_count:  {info['points_count']}")
    print(f"\nQdrant UI: {args.url}/dashboard")


if __name__ == "__main__":
    main()
