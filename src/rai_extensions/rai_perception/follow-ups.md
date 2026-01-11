# After refactoring is done

## Generic Detection Tools Abstraction

problem

`gdino_tools.py` is tightly coupled to GroundingDINO model and service interface:

-   Hardcoded `RAIGroundingDino` service type
-   Model-specific parameters (`box_threshold`, `text_threshold`) as class fields
-   Direct parsing of `RAIGroundingDino.Response` structure

This prevents easy migration to other models (e.g., YOLO) without code changes.

## Proposed Solution: Service-Level Abstraction (Option A)

### Abstraction Layers Needed

1. **Generic Service Interface** (`rai_interfaces/`)

    - Create `RAIDetection.srv` with generic fields:
        - Request: `source_img`, `object_names[]`, `model_params` (dict/JSON)
        - Response: `RAIDetectionArray` (already exists)

2. **Service Adapter Layer** (`services/detection_service.py`)

    - Convert generic `model_params` → model-specific parameters
    - GroundingDINO: extract `box_threshold`, `text_threshold`
    - YOLO: extract `confidence_threshold`, `nms_threshold`, etc.
    - Update service to use `RAIDetection` interface

3. **Tool Updates** (`tools/gdino_tools.py`)
    - Replace `RAIGroundingDino` → `RAIDetection` service type
    - Replace hardcoded parameter fields → `model_params` dict
    - Parse `RAIDetectionArray` directly (already returned by service)

### Benefits

-   Tools become model-agnostic
-   New models only require registry entry + parameter mapping in service
-   Minimal tool code changes for model switching

### Scope

-   New service interface definition
-   Service adapter logic (~50-100 lines)
-   Tool refactoring (~100-150 lines)
-   Model parameter registry extension
