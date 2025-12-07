#!/usr/bin/env python3
"""
CLI tool to sync features from Neuronpedia S3 to local database with resume capability.

Usage:
    python sync-features-cli.py --model-id qwen3-4b --source-id 0-transcoder-hp
    python sync-features-cli.py --model-id qwen3-4b --source-id 0-transcoder-hp --feature-range 0-35
    python sync-features-cli.py --model-id qwen3-4b --source-id 0-transcoder-hp --skip-explanations
    python sync-features-cli.py --model-id qwen3-4b --source-id 0-transcoder-hp --resume
"""

import os
import sys
import json
import gzip
import argparse
import psycopg2
from typing import Optional, List, Tuple
from urllib.request import urlopen
from datetime import datetime, timezone
import pickle
from pathlib import Path

DATASET_BASE_URL = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/v1"
STATE_FILE = ".sync_state.pkl"
CACHE_DIR = ".sync_cache"


def load_env_file(env_file_path: str):
    """Load environment variables from .env file."""
    if not os.path.exists(env_file_path):
        return

    with open(env_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # Skip if value references another env var
                if not value.startswith('${'):
                    os.environ[key] = value


def find_and_load_env():
    """Find and load .env.localhost from project root."""
    # Try to find project root by looking for .env.localhost
    current_dir = Path.cwd()

    # Search up to 5 levels up
    for _ in range(5):
        env_file = current_dir / '.env.localhost'
        if env_file.exists():
            print(f"📁 Loading environment from: {env_file}")
            load_env_file(str(env_file))
            return True

        parent = current_dir.parent
        if parent == current_dir:  # Reached root
            break
        current_dir = parent

    return False


class SyncState:
    """Tracks sync progress for resume capability."""

    def __init__(self, model_id: str, source_id: str):
        self.model_id = model_id
        self.source_id = source_id
        self.metadata_done = False
        self.features_files_done: List[str] = []
        self.activations_files_done: List[str] = []
        self.explanations_files_done: List[str] = []

    @staticmethod
    def load(model_id: str, source_id: str) -> 'SyncState':
        """Load state from file or create new."""
        state_file = f"{STATE_FILE}.{model_id}.{source_id}"
        if os.path.exists(state_file):
            with open(state_file, 'rb') as f:
                return pickle.load(f)
        return SyncState(model_id, source_id)

    def save(self):
        """Save state to file."""
        state_file = f"{STATE_FILE}.{self.model_id}.{self.source_id}"
        with open(state_file, 'wb') as f:
            pickle.dump(self, f)

    def clear(self):
        """Clear state file."""
        state_file = f"{STATE_FILE}.{self.model_id}.{self.source_id}"
        if os.path.exists(state_file):
            os.remove(state_file)


def get_db_connection():
    """Get PostgreSQL database connection from environment."""
    # Try DATABASE_URL first
    database_url = os.getenv('DATABASE_URL')

    # Fall back to POSTGRES_URL_NON_POOLING from .env.localhost
    if not database_url:
        database_url = os.getenv('POSTGRES_URL_NON_POOLING')

    # Fall back to POSTGRES_PRISMA_URL
    if not database_url:
        database_url = os.getenv('POSTGRES_PRISMA_URL')

    if not database_url:
        raise ValueError(
            "No database URL found. Please set DATABASE_URL, POSTGRES_URL_NON_POOLING, "
            "or POSTGRES_PRISMA_URL in your environment or .env.localhost file."
        )

    # Convert Docker hostname to localhost if running outside Docker
    if 'postgres:5432' in database_url and not os.getenv('IS_DOCKER_COMPOSE'):
        database_url = database_url.replace('postgres:5432', 'localhost:5432')
        print(f"🔄 Converted Docker hostname to localhost")

    print(f"🔌 Connecting to database: {database_url.split('@')[1] if '@' in database_url else database_url}")
    return psycopg2.connect(database_url)


def download_and_decompress(url: str) -> str:
    """Download and decompress gzipped file from URL with local caching."""
    # Create cache directory if it doesn't exist
    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)

    # Generate cache file path from URL
    # Extract meaningful parts: model-id/source-id/type/filename
    url_parts = url.replace(DATASET_BASE_URL + '/', '').split('/')
    cache_subdir = cache_dir / '/'.join(url_parts[:-1])
    cache_subdir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_subdir / url_parts[-1]

    # Check if cached file exists
    if cache_file.exists():
        print(f"  Using cached: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                compressed_data = f.read()
                if url.endswith('.gz'):
                    return gzip.decompress(compressed_data).decode('utf-8')
                return compressed_data.decode('utf-8')
        except Exception as e:
            print(f"  Cache read error, re-downloading: {e}")
            # If cache is corrupted, continue to download

    # Download file
    print(f"  Downloading: {url}")
    try:
        with urlopen(url) as response:
            compressed_data = response.read()

            # Save to cache
            with open(cache_file, 'wb') as f:
                f.write(compressed_data)

            if url.endswith('.gz'):
                return gzip.decompress(compressed_data).decode('utf-8')
            return compressed_data.decode('utf-8')
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        raise


def list_files_in_s3_path(base_url: str, path: str, extension: str) -> List[str]:
    """List files in S3 path by trying sequential batch files."""
    url = f"{base_url}/{path}"
    print(f"  Discovering batch files...")

    # For S3, we need to construct expected file names
    # Typically batch-0.jsonl.gz, batch-1.jsonl.gz, etc.
    files = []
    i = 0
    consecutive_failures = 0
    max_consecutive_failures = 3

    while consecutive_failures < max_consecutive_failures:
        file_url = f"{url}/batch-{i}{extension}"
        print(f"    Checking batch-{i}...", end=' ', flush=True)
        try:
            # Try to access the file to see if it exists
            with urlopen(file_url) as response:
                if response.status == 200:
                    print("✓")
                    files.append(file_url)
                    consecutive_failures = 0
                else:
                    print("✗")
                    consecutive_failures += 1
        except Exception:
            print("✗")
            consecutive_failures += 1

        i += 1

    print(f"  Found {len(files)} batch files\n")
    return files


def import_jsonl_batch(conn, table_name: str, jsonl_string: str, feature_range: Optional[Tuple[int, int]] = None):
    """Import JSONL data into database table."""
    lines = [line for line in jsonl_string.strip().split('\n') if line.strip()]

    if not lines:
        return 0

    imported_count = 0
    cursor = conn.cursor()

    for line in lines:
        try:
            data = json.loads(line)

            # Filter by feature range if specified
            if feature_range and 'index' in data:
                feature_idx = int(data['index']) if isinstance(data['index'], str) else data['index']
                if not (feature_range[0] <= feature_idx <= feature_range[1]):
                    continue

            # Insert based on table type
            if table_name == 'Neuron':
                # Convert index to string for database compatibility
                index_str = str(data['index'])

                # Check if exists
                cursor.execute(
                    'SELECT 1 FROM "Neuron" WHERE "modelId" = %s AND layer = %s AND index = %s',
                    (data['modelId'], data['layer'], index_str)
                )
                if cursor.fetchone():
                    continue

                # Insert feature
                # Convert vector to PostgreSQL array format
                vector = data.get('vector', [])

                # Handle null createdAt - use current timestamp if null
                created_at = data.get('createdAt')
                if created_at is None:
                    created_at = datetime.now(timezone.utc).isoformat()

                cursor.execute('''
                    INSERT INTO "Neuron" (
                        "modelId", layer, index, "creatorId", "createdAt", "hasVector",
                        vector, "hookName", "maxActApprox", "vectorDefaultSteerStrength"
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                ''', (
                    data['modelId'], data['layer'], index_str, data.get('creatorId'),
                    created_at, data.get('hasVector', False),
                    vector, data.get('hookName'),
                    data.get('maxActApprox'), data.get('vectorDefaultSteerStrength')
                ))
                imported_count += 1

            elif table_name == 'Activation':
                # Convert index to string for database compatibility
                index_str = str(data['index'])

                # Check if exists
                cursor.execute(
                    'SELECT 1 FROM "Activation" WHERE id = %s',
                    (data['id'],)
                )
                if cursor.fetchone():
                    continue

                # Filter by feature range
                if feature_range:
                    feature_idx = data.get('index')
                    if feature_idx is not None:
                        feature_idx = int(feature_idx) if isinstance(feature_idx, str) else feature_idx
                        if not (feature_range[0] <= feature_idx <= feature_range[1]):
                            continue

                # Check if referenced neuron exists
                cursor.execute(
                    'SELECT 1 FROM "Neuron" WHERE "modelId" = %s AND layer = %s AND index = %s',
                    (data['modelId'], data['layer'], index_str)
                )
                if not cursor.fetchone():
                    # Skip this activation as the neuron doesn't exist
                    continue

                # Insert activation
                # Convert arrays to PostgreSQL format
                tokens = data.get('tokens', [])
                values = data.get('values', [])

                # Handle null createdAt - use current timestamp if null
                created_at = data.get('createdAt')
                if created_at is None:
                    created_at = datetime.now(timezone.utc).isoformat()

                cursor.execute('''
                    INSERT INTO "Activation" (
                        id, tokens, "modelId", layer, index, "maxValue", "maxValueTokenIndex",
                        "minValue", values, "creatorId", "createdAt"
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                ''', (
                    data['id'], tokens, data['modelId'],
                    data['layer'], index_str, data.get('maxValue'), data.get('maxValueTokenIndex'),
                    data.get('minValue'), values,
                    data.get('creatorId'), created_at
                ))
                imported_count += 1

            elif table_name == 'Explanation':
                # Convert index to string for database compatibility
                index_str = str(data.get('index'))

                # Similar logic for explanations
                if feature_range:
                    feature_idx = data.get('index')
                    if feature_idx is not None:
                        feature_idx = int(feature_idx) if isinstance(feature_idx, str) else feature_idx
                        if not (feature_range[0] <= feature_idx <= feature_range[1]):
                            continue

                # Handle null createdAt - use current timestamp if null
                created_at = data.get('createdAt')
                if created_at is None:
                    created_at = datetime.now(timezone.utc).isoformat()

                cursor.execute('''
                    INSERT INTO "Explanation" (
                        "modelId", layer, index, text, "creatorId", "createdAt"
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                ''', (
                    data.get('modelId'), data.get('layer'), index_str,
                    data.get('text'), data.get('creatorId'), created_at
                ))
                imported_count += 1
                # Commit after each successful insert to avoid losing data on errors
                conn.commit()

        except Exception as e:
            print(f"    Error importing line: {e}")
            # Rollback only the failed transaction
            conn.rollback()
            continue

    cursor.close()
    return imported_count


def import_metadata(conn, base_path: str, model_id: str):
    """Import metadata (release, model, sourceset, source)."""
    print("Importing metadata...")
    cursor = conn.cursor()

    # Import release
    try:
        release_data = download_and_decompress(f"{base_path}/release.jsonl")
        for line in release_data.strip().split('\n'):
            if line.strip():
                data = json.loads(line)
                # Convert urls to PostgreSQL array format if it's a list
                urls = data.get('urls', [])
                if isinstance(urls, list):
                    urls_array = '{' + ','.join(f'"{url}"' for url in urls) + '}'
                else:
                    urls_array = '{}'

                cursor.execute(
                    'INSERT INTO "SourceRelease" (name, description, "descriptionShort", urls, "creatorName", "creatorId", "createdAt") '
                    'VALUES (%s, %s, %s, %s::text[], %s, %s, %s) ON CONFLICT DO NOTHING',
                    (data['name'], data.get('description'), data.get('descriptionShort'),
                     urls_array, data.get('creatorName'),
                     data.get('creatorId'), data.get('createdAt'))
                )
        conn.commit()
    except Exception as e:
        print(f"  Warning: Could not import release: {e}")
        conn.rollback()

    # Import model
    try:
        model_data = download_and_decompress(f"{base_path}/model.jsonl")
        for line in model_data.strip().split('\n'):
            if line.strip():
                data = json.loads(line)
                cursor.execute(
                    'INSERT INTO "Model" (id, instruct, "displayName", "displayNameShort", "creatorId", "createdAt", '
                    'owner, layers, "neuronsPerLayer", website, visibility, dimension, "inferenceEnabled", "tlensId", '
                    '"defaultSourceSetName", "defaultSourceId", "updatedAt") '
                    'VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING',
                    (data['id'], data.get('instruct', False), data.get('displayName'),
                     data.get('displayNameShort'), data.get('creatorId'), data.get('createdAt'),
                     data.get('owner'), data.get('layers'), data.get('neuronsPerLayer'),
                     data.get('website'), data.get('visibility', 'PRIVATE'), data.get('dimension'),
                     data.get('inferenceEnabled', False), data.get('tlensId'),
                     data.get('defaultSourceSetName'), data.get('defaultSourceId'), data.get('updatedAt'))
                )
        conn.commit()
    except Exception as e:
        print(f"  Warning: Could not import model: {e}")
        conn.rollback()

    # Import sourceset
    try:
        sourceset_data = download_and_decompress(f"{base_path}/sourceset.jsonl")
        for line in sourceset_data.strip().split('\n'):
            if line.strip():
                data = json.loads(line)
                # Convert urls to PostgreSQL array format if it's a list
                urls = data.get('urls', [])
                if isinstance(urls, list):
                    urls_array = '{' + ','.join(f'"{url}"' for url in urls) + '}'
                else:
                    urls_array = '{}'

                cursor.execute(
                    'INSERT INTO "SourceSet" ("modelId", name, description, visibility, "releaseName", "creatorId", "createdAt", '
                    '"creatorName", type, urls, "creatorEmail") '
                    'VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::text[], %s) ON CONFLICT DO NOTHING',
                    (data['modelId'], data['name'], data.get('description'),
                     data.get('visibility', 'PUBLIC'), data.get('releaseName'),
                     data.get('creatorId'), data.get('createdAt'),
                     data.get('creatorName'), data.get('type', ''), urls_array, data.get('creatorEmail'))
                )
        conn.commit()
    except Exception as e:
        print(f"  Warning: Could not import sourceset: {e}")
        conn.rollback()

    # Import source
    try:
        source_data = download_and_decompress(f"{base_path}/source.jsonl")
        for line in source_data.strip().split('\n'):
            if line.strip():
                data = json.loads(line)
                cursor.execute(
                    'INSERT INTO "Source" ("modelId", id, "setName", dataset, visibility, "hfRepoId", "creatorId") '
                    'VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING',
                    (data['modelId'], data['id'], data['setName'], data.get('dataset'),
                     data.get('visibility', 'PUBLIC'), data.get('hfRepoId'), data.get('creatorId'))
                )
        conn.commit()
    except Exception as e:
        print(f"  Warning: Could not import source: {e}")
        conn.rollback()

    cursor.close()
    print("  Metadata imported.")


def sync_features(model_id: str, source_id: str, skip_explanations: bool = False,
                  resume: bool = False, feature_range: Optional[Tuple[int, int]] = None):
    """Sync features from S3 to local database with resume capability."""

    # Load or create state
    state = SyncState.load(model_id, source_id) if resume else SyncState(model_id, source_id)

    base_path = f"{DATASET_BASE_URL}/{model_id}/{source_id}"

    try:
        conn = get_db_connection()

        # Import metadata
        if not state.metadata_done:
            import_metadata(conn, base_path, model_id)
            state.metadata_done = True
            state.save()

        # Import features
        print("\nImporting features...")
        feature_files = list_files_in_s3_path(DATASET_BASE_URL, f"{model_id}/{source_id}/features", ".jsonl.gz")
        for i, file_url in enumerate(feature_files):
            file_name = file_url.split('/')[-1]
            if file_name in state.features_files_done:
                print(f"  [{i+1}/{len(feature_files)}] {file_name} (skipped - already done)")
                continue

            print(f"  [{i+1}/{len(feature_files)}] {file_name}")
            data = download_and_decompress(file_url)
            count = import_jsonl_batch(conn, 'Neuron', data, feature_range)
            print(f"    Imported {count} features")

            state.features_files_done.append(file_name)
            state.save()

        # Import activations
        print("\nImporting activations...")
        activation_files = list_files_in_s3_path(DATASET_BASE_URL, f"{model_id}/{source_id}/activations", ".jsonl.gz")
        for i, file_url in enumerate(activation_files):
            file_name = file_url.split('/')[-1]
            if file_name in state.activations_files_done:
                print(f"  [{i+1}/{len(activation_files)}] {file_name} (skipped - already done)")
                continue

            print(f"  [{i+1}/{len(activation_files)}] {file_name}")
            data = download_and_decompress(file_url)
            count = import_jsonl_batch(conn, 'Activation', data, feature_range)
            print(f"    Imported {count} activations")

            # Mark as done even if 0 imported (already exists in DB)
            state.activations_files_done.append(file_name)
            state.save()

        # Import explanations
        if not skip_explanations:
            print("\nImporting explanations...")
            explanation_files = list_files_in_s3_path(DATASET_BASE_URL, f"{model_id}/{source_id}/explanations", ".jsonl.gz")
            for i, file_url in enumerate(explanation_files):
                file_name = file_url.split('/')[-1]
                if file_name in state.explanations_files_done:
                    print(f"  [{i+1}/{len(explanation_files)}] {file_name} (skipped - already done)")
                    continue

                print(f"  [{i+1}/{len(explanation_files)}] {file_name}")
                data = download_and_decompress(file_url)
                count = import_jsonl_batch(conn, 'Explanation', data, feature_range)
                print(f"    Imported {count} explanations")

                state.explanations_files_done.append(file_name)
                state.save()

        conn.close()

        # Clear state on successful completion
        state.clear()
        print("\n✅ Sync completed successfully!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted! Progress has been saved. Run with --resume to continue.")
        state.save()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("Progress has been saved. Run with --resume to continue.")
        state.save()
        sys.exit(1)


def main():
    # Load environment before anything else
    if not find_and_load_env():
        print("⚠️  Warning: Could not find .env.localhost file")
        print("   Continuing with system environment variables...\n")

    parser = argparse.ArgumentParser(
        description='Sync features from Neuronpedia S3 to local database with resume capability.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sync single layer, all features
  python sync-features-cli.py --model-id qwen3-4b --source-id 0-transcoder-hp

  # Sync single layer, specific feature range
  python sync-features-cli.py --model-id qwen3-4b --source-id 0-transcoder-hp --feature-range 0-35

  # Sync all layers 0-25, all features, skip explanations (RECOMMENDED)
  python sync-features-cli.py --model-id qwen3-4b --layer-range 0-25 --skip-explanations

  # Sync all layers 0-25, specific feature range
  python sync-features-cli.py --model-id qwen3-4b --layer-range 0-25 --feature-range 0-100

  # Resume interrupted sync
  python sync-features-cli.py --model-id qwen3-4b --source-id 0-transcoder-hp --resume
        """
    )

    parser.add_argument('--model-id', required=True, help='Model ID (e.g., qwen3-4b)')
    parser.add_argument('--source-id', help='Source ID (e.g., 0-transcoder-hp). Can also use --layer-range instead.', default=None)
    parser.add_argument('--layer-range', help='Layer range to sync (e.g., 0-35). Will sync all layers from start to end.', default=None)
    parser.add_argument('--source-prefix', help='Source prefix when using --layer-range (e.g., transcoder-hp)', default='transcoder-hp')
    parser.add_argument('--feature-range', help='Feature range to sync (e.g., 0-35)', default=None)
    parser.add_argument('--skip-explanations', action='store_true', help='Skip downloading explanations')
    parser.add_argument('--resume', action='store_true', help='Resume from previous interrupted sync')

    args = parser.parse_args()

    # Parse feature range
    feature_range = None
    if args.feature_range:
        try:
            start, end = map(int, args.feature_range.split('-'))
            feature_range = (start, end)
        except Exception:
            print(f"Error: Invalid feature range format. Use: 0-35")
            sys.exit(1)

    # Determine which layers to sync
    layers_to_sync = []
    if args.layer_range:
        try:
            start_layer, end_layer = map(int, args.layer_range.split('-'))
            layers_to_sync = [f"{i}-{args.source_prefix}" for i in range(start_layer, end_layer + 1)]
        except Exception:
            print(f"Error: Invalid layer range format. Use: 0-35")
            sys.exit(1)
    elif args.source_id:
        layers_to_sync = [args.source_id]
    else:
        print("Error: Must specify either --source-id or --layer-range")
        sys.exit(1)

    # Sync each layer
    for idx, source_id in enumerate(layers_to_sync):
        print(f"\n{'='*60}")
        print(f"🔄 Syncing layer {idx+1}/{len(layers_to_sync)}: {args.model_id}/{source_id}")
        print(f"{'='*60}")
        if feature_range:
            print(f"   Feature range: {feature_range[0]}-{feature_range[1]}")
        if args.skip_explanations:
            print(f"   Skipping explanations")
        if args.resume:
            print(f"   Resuming from saved state")
        print()

        try:
            sync_features(
                args.model_id,
                source_id,
                skip_explanations=args.skip_explanations,
                resume=args.resume,
                feature_range=feature_range
            )
            print(f"✅ Layer {source_id} completed!")
        except Exception as e:
            print(f"❌ Error syncing layer {source_id}: {e}")
            print(f"Continuing to next layer...")
            continue

    print(f"\n{'='*60}")
    print(f"🎉 All layers synced successfully!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
