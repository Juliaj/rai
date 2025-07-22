# Copyright (C) 2025 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pytest


@pytest.fixture(autouse=True)
def disable_tracing():
    """Disable tracing during tests to avoid external dependencies."""
    # Store original values
    original_langfuse_public = os.environ.get("LANGFUSE_PUBLIC_KEY")
    original_langfuse_secret = os.environ.get("LANGFUSE_SECRET_KEY")
    
    # Temporarily unset Langfuse keys during tests
    if "LANGFUSE_PUBLIC_KEY" in os.environ:
        del os.environ["LANGFUSE_PUBLIC_KEY"]
    if "LANGFUSE_SECRET_KEY" in os.environ:
        del os.environ["LANGFUSE_SECRET_KEY"]
    
    yield
    
    # Restore original values after test
    if original_langfuse_public is not None:
        os.environ["LANGFUSE_PUBLIC_KEY"] = original_langfuse_public
    if original_langfuse_secret is not None:
        os.environ["LANGFUSE_SECRET_KEY"] = original_langfuse_secret
