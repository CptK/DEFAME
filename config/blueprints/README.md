# DEFAME Verification Blueprints

This directory contains verification blueprints for the DEFAME fact-checking system. Blueprints define verification strategies tailored to different types of claims.

## Available Tools

**Note**: Blueprints currently use only the fully functional tools in DEFAME:
- ✅ **search** - Web search (Google, Wikipedia, news, etc.)
- ✅ **geolocate** - Determine location from images (StreetCLIP model)
- ✅ **detect_objects** - Detect objects in images (DETR model)
- ✅ **detect_manipulation** - Detect image manipulation (TruFor model)

**Not currently used** (stub implementations, not functional):
- ❌ **credibility_check** - Source credibility assessment (not implemented)
- ❌ **face_recognition** - Identify faces (not implemented)
- ❌ **ocr** - Extract text from images (not implemented, code commented out)

## Available Blueprints

### 1. Simple Factual (`simple_factual.yaml`)
- **Use for**: Straightforward factual claims without images or complex context
- **Strategy**: Basic web search with iterative verification
- **Iterations**: 2
- **Tools**: search
- **Example**: "The capital of France is Paris"

### 2. Complex Multimodal (`complex_multimodal.yaml`)
- **Use for**: Claims involving both text and visual content
- **Strategy**: Image analysis (objects, manipulation, location), then text search
- **Iterations**: 3
- **Tools**: detect_objects, detect_manipulation, geolocate, search
- **Example**: "This photo shows flooding in City X in 2023"

### 3. Temporal Verification (`temporal_verification.yaml`)
- **Use for**: Time-sensitive events or historical facts
- **Strategy**: Multiple searches to establish timeline from diverse sources
- **Iterations**: 3-4
- **Tools**: search
- **Example**: "Event X happened before Event Y"

### 4. Source Comparison (`source_comparison.yaml`)
- **Use for**: Controversial claims requiring multiple independent sources
- **Strategy**: Multiple parallel searches, compare and synthesize findings
- **Iterations**: 3
- **Tools**: search (multiple per iteration)
- **Example**: "Reports claim that Policy X has Effect Y"

### 5. Visual Misinformation (`visual_misinformation.yaml`)
- **Use for**: Claims where image/video authenticity is the primary concern
- **Strategy**: Detect manipulation, analyze objects and location, then search for context
- **Iterations**: 3
- **Tools**: detect_manipulation, detect_objects, geolocate, search
- **Example**: "This video shows a deepfake of Person X"

### 6. Numerical Claim (`numerical_claim.yaml`)
- **Use for**: Claims involving statistics or quantitative data
- **Strategy**: Multiple searches to find authoritative sources with exact data
- **Iterations**: 3
- **Tools**: search
- **Example**: "Unemployment rate is X%"

### 7. Quote Verification (`quote_verification.yaml`)
- **Use for**: Claims about statements allegedly made by public figures
- **Strategy**: Multiple searches to find primary source or official statement
- **Iterations**: 3
- **Tools**: search
- **Example**: "Person X said 'controversial statement'"

### 8. Deep Investigation (`deep_investigation.yaml`)
- **Use for**: Complex, multi-faceted claims requiring thorough investigation
- **Strategy**: Multiple searches per iteration with synthesis at each stage
- **Iterations**: 4-5
- **Tools**: search (multiple per iteration)
- **Example**: Complex conspiracy claims or multi-part allegations

## Blueprint Structure

Each blueprint YAML file contains:
- `name`: Unique identifier
- `description`: When to use this blueprint
- `claim_characteristics`: Types of claims this handles
- `iterations`: Sequence of actions per iteration
  - `actions`: List of actions to execute
  - `synthesis`: Whether to add explicit synthesis stage
- `stopping_criteria`: When to stop verification
  - `max_iterations`: Maximum iterations allowed
  - `early_stop_conditions`: Conditions for early stopping

## Blueprint Selection

The appropriate blueprint should be selected based on claim characteristics:
- Presence of images/videos
- Complexity of the claim
- Temporal aspects
- Source of the claim
- Type of content (quote, number, event, etc.)

See `schema.yaml` for detailed format specification.

## Adding New Blueprints

1. Copy an existing blueprint as a template
2. Modify the action sequence based on your verification strategy
3. Update claim_characteristics to describe when to use it
4. Test on relevant claims
5. Document rationale for the strategy choices
