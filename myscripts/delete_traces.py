#!/usr/bin/env python3
"""
Script to delete all traces from a locally hosted Langfuse instance.

This script connects to a Langfuse instance running on localhost:3000 and deletes
all traces. It requires the Langfuse Python SDK and appropriate credentials.

Usage:
    python src/rai_finetune/delete_langfuse_traces.py

Environment variables required:
    - LANGFUSE_PUBLIC_KEY: Your Langfuse public key
    - LANGFUSE_SECRET_KEY: Your Langfuse secret key
    - LANGFUSE_HOST: (optional) Langfuse host URL (defaults to http://localhost:3000)
"""

import os
import sys
from typing import Optional

import requests
from langfuse import Langfuse


def get_langfuse_client() -> Optional[tuple[Langfuse, str]]:
    """Initialize and return a Langfuse client and host URL."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

    if not public_key or not secret_key:
        print(
            "Error: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables are required."
        )
        print("Please set them before running this script.")
        return None

    try:
        langfuse = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        return langfuse, host
    except Exception as e:
        print(f"Error initializing Langfuse client: {e}")
        return None


def check_langfuse_connection(langfuse: Langfuse, host: str) -> bool:
    """Check if we can connect to the Langfuse instance."""
    try:
        # Try to get a list of traces to verify connection
        langfuse.get_traces(limit=1)
        print(f"✓ Successfully connected to Langfuse at {host}")
        return True
    except Exception as e:
        print(f"✗ Failed to connect to Langfuse: {e}")
        return False


def get_trace_count(langfuse: Langfuse) -> int:
    """Get the total number of traces in the Langfuse instance."""
    try:
        # Get all traces to count them
        traces = langfuse.get_traces(limit=100)  # Adjust limit as needed
        return len(traces.data)
    except Exception as e:
        print(f"Error getting trace count: {e}")
        return 0


def delete_all_traces(
    langfuse: Langfuse, host: str, public_key: str, secret_key: str
) -> bool:
    """
    Delete all traces from the Langfuse instance.

    Note: This uses the Langfuse API directly since the Python SDK doesn't
    provide a direct method to delete all traces at once.
    """
    try:
        # Get all trace IDs
        traces = langfuse.get_traces(limit=100)  # Adjust limit as needed
        trace_ids = [trace.id for trace in traces.data]

        if not trace_ids:
            print("No traces found to delete.")
            return True

        print(f"Found {len(trace_ids)} traces to delete...")

        # Delete each trace
        deleted_count = 0
        for trace_id in trace_ids:
            try:
                # Use the API directly to delete the trace with Basic Auth
                # username: Langfuse Public Key, password: Langfuse Secret Key
                response = requests.delete(
                    f"{host}/api/public/traces/{trace_id}",
                    auth=(public_key, secret_key),
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code == 200:
                    deleted_count += 1
                    print(f"✓ Deleted trace {trace_id}")
                else:
                    print(
                        f"✗ Failed to delete trace {trace_id}: {response.status_code}"
                    )

            except Exception as e:
                print(f"✗ Error deleting trace {trace_id}: {e}")

        print(f"\nSuccessfully deleted {deleted_count} out of {len(trace_ids)} traces.")
        return deleted_count == len(trace_ids)

    except Exception as e:
        print(f"Error during trace deletion: {e}")
        return False


def main():
    """Main function to delete all traces from Langfuse."""
    print("Langfuse Trace Deletion Script")
    print("=" * 40)

    # Initialize Langfuse client
    result = get_langfuse_client()
    if not result:
        sys.exit(1)

    langfuse, host = result
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    # Check connection
    if not check_langfuse_connection(langfuse, host):
        sys.exit(1)

    # Get trace count
    trace_count = get_trace_count(langfuse)
    print(f"Current trace count: {trace_count}")

    if trace_count == 0:
        print("No traces to delete.")
        return

    # Confirm deletion
    print(
        f"\n⚠️  WARNING: This will delete ALL {trace_count} traces from your Langfuse instance!"
    )
    print("This action cannot be undone.")

    confirm = input("\nAre you sure you want to continue? (yes/no): ").lower().strip()
    if confirm not in ["yes", "y"]:
        print("Operation cancelled.")
        return

    # Delete traces
    print("\nDeleting traces...")
    success = delete_all_traces(langfuse, host, public_key, secret_key)

    if success:
        print("\n✅ All traces have been successfully deleted!")
    else:
        print(
            "\n❌ Some traces may not have been deleted. Please check the output above."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
