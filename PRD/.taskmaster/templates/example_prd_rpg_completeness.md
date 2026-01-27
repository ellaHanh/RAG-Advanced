<rpg-method>
# Repository Planning Graph (RPG) Method - PRD Template with Completeness Checklist

This template extends the standard RPG methodology with a **Completeness Checklist** to ensure PRDs don't miss critical bootstrap, scaffolding, and developer experience requirements.

## Lesson Learned
PRDs that focus only on *domain capabilities* tend to miss *bootstrap/scaffolding/glue code* that is essential for the system to actually run. Task Master and similar tools will not generate tasks for "obvious" infrastructure unless explicitly specified.

## Core Principles

1. **Dual-Semantics**: Think functional (capabilities) AND structural (code organization) separately, then map them
2. **Explicit Dependencies**: Never assume - always state what depends on what
3. **Topological Order**: Build foundation first, then layers on top
4. **Progressive Refinement**: Start broad, refine iteratively
5. **Bootstrap First** *(NEW)*: Include application entrypoints, config, and DX as first-class features

## Completeness Checklist (Review Before Finalizing PRD)

Before considering your PRD complete, verify these items are explicitly defined:

### Application Bootstrap
- [ ] Main application entrypoint file (e.g., `api/main.py`, `src/index.ts`)
- [ ] App instance creation and configuration
- [ ] Router/route registration
- [ ] Lifespan/startup/shutdown hooks
- [ ] CLI entrypoint matching `pyproject.toml`/`package.json` scripts
- [ ] Health check endpoint (`/health` or `/healthz`)
- [ ] API documentation endpoint (`/docs`, `/swagger`)

### Configuration & Environment
- [ ] Environment template file (`.env.example`)
- [ ] Settings/config module with validation
- [ ] Required vs optional environment variables documented
- [ ] Secrets management approach

### Database & Infrastructure Init
- [ ] Connection pool initialization in app lifespan
- [ ] Schema migration or setup scripts
- [ ] Cache/Redis initialization
- [ ] External service connections

### Developer Experience
- [ ] "Smoke test" acceptance criteria (server starts, health returns 200)
- [ ] Local development setup instructions
- [ ] Test fixtures and mock data
- [ ] Pre-commit hooks configuration

### Deployment Contracts
- [ ] Dockerfile CMD matches actual code entrypoint
- [ ] docker-compose.yml service dependencies
- [ ] Container health checks
- [ ] Volume mounts for persistent data

## Review Questions (Ask Before Submitting PRD)

1. "Can I actually START the application with what's specified?"
2. "Is every file referenced in pyproject.toml/Dockerfile defined as a task?"
3. "Is there a health check and docs endpoint?"
4. "Is there an .env.example or config documentation?"
5. "Can a new developer run this in 5 minutes?"
</rpg-method>

---

<overview>
<instruction>
Start with the problem, not the solution. Be specific about:
- What pain point exists?
- Who experiences it?
- Why existing solutions don't work?
- What success looks like (measurable outcomes)?

Keep this section focused - don't jump into implementation details yet.
</instruction>

## Problem Statement
[Describe the core problem. Be concrete about user pain points.]

## Target Users
[Define personas, their workflows, and what they're trying to achieve.]

## Success Metrics
[Quantifiable outcomes. Examples: "80% task completion via autopilot", "< 5% manual intervention rate"]

</overview>

---

<functional-decomposition>
<instruction>
Now think about CAPABILITIES (what the system DOES), not code structure yet.

Step 1: Identify high-level capability domains
- Think: "What major things does this system do?"
- Examples: Data Management, Core Processing, Presentation Layer
- **NEW**: Always include "Application Bootstrap & Developer Experience" as a capability

Step 2: For each capability, enumerate specific features
- Use explore-exploit strategy:
  * Exploit: What features are REQUIRED for core value?
  * Explore: What features make this domain COMPLETE?

Step 3: For each feature, define:
- Description: What it does in one sentence
- Inputs: What data/context it needs
- Outputs: What it produces/returns
- Behavior: Key logic or transformations

<example type="good">
Capability: Data Validation
  Feature: Schema validation
    - Description: Validate JSON payloads against defined schemas
    - Inputs: JSON object, schema definition
    - Outputs: Validation result (pass/fail) + error details
    - Behavior: Iterate fields, check types, enforce constraints

  Feature: Business rule validation
    - Description: Apply domain-specific validation rules
    - Inputs: Validated data object, rule set
    - Outputs: Boolean + list of violated rules
    - Behavior: Execute rules sequentially, short-circuit on failure
</example>

<example type="bad">
Capability: validation.js
  (Problem: This is a FILE, not a CAPABILITY. Mixing structure into functional thinking.)

Capability: Validation
  Feature: Make sure data is good
  (Problem: Too vague. No inputs/outputs. Not actionable.)
</example>
</instruction>

## Capability Tree

### Capability: [Domain Capability Name]
[Brief description of what this capability domain covers]

#### Feature: [Name]
- **Description**: [One sentence]
- **Inputs**: [What it needs]
- **Outputs**: [What it produces]
- **Behavior**: [Key logic]

#### Feature: [Name]
- **Description**:
- **Inputs**:
- **Outputs**:
- **Behavior**:

### Capability: [Another Domain Capability]
...

---

<bootstrap-capability>
<instruction>
**CRITICAL**: This section is often missed in PRDs, causing deployment failures.

Always include this capability to ensure the application can actually start and run.
Without this, you'll have all the features but no way to wire them together.

This capability covers:
1. Application entrypoint (the file that creates and configures the app)
2. Configuration management (loading env vars, validating config)
3. Health/observability endpoints
4. Developer experience tooling
</instruction>

### Capability: Application Bootstrap & Developer Experience
Provides the runtime foundation that wires all other capabilities together and ensures the system is runnable, observable, and developer-friendly.

#### Feature: Application Entrypoint
- **Description**: Main application file that creates and configures the app instance
- **Inputs**: Environment variables, configuration files
- **Outputs**: Configured application instance ready to serve requests
- **Behavior**: 
  - Create app instance (e.g., FastAPI, Express)
  - Register all route modules
  - Configure middleware (CORS, logging, error handling)
  - Initialize lifespan hooks for startup/shutdown
  - Expose `run()` function for CLI entrypoint

<example type="good">
Feature: Application Entrypoint
  - Description: FastAPI application with proper initialization
  - Files: api/main.py
  - Inputs: Environment variables (DATABASE_URL, REDIS_URL, API_KEY)
  - Outputs: 
    - `app` - FastAPI instance
    - `run()` - Function called by `pyproject.toml` script
  - Behavior:
    - Create FastAPI app with metadata
    - Register routers: strategies, evaluation, benchmarks
    - Add lifespan manager for DB pool and pricing provider init
    - Configure CORS for allowed origins
    - Expose run() that calls uvicorn.run(app)
</example>

#### Feature: Configuration Management
- **Description**: Load and validate configuration from environment
- **Inputs**: Environment variables, .env file, config files
- **Outputs**: Validated settings object
- **Behavior**:
  - Load from environment with fallbacks
  - Validate required variables exist
  - Fail fast on missing critical config
  - Expose typed settings for other modules

<example type="good">
Feature: Configuration Management
  - Files: config/settings.py
  - Inputs: Environment variables
  - Outputs: Settings pydantic model
  - Behavior:
    - Use pydantic-settings BaseSettings
    - Validate DATABASE_URL format
    - Validate API keys are present
    - Provide defaults for optional settings
</example>

#### Feature: Health Check Endpoint
- **Description**: Endpoint for infrastructure health monitoring
- **Inputs**: HTTP GET request
- **Outputs**: JSON response with health status
- **Behavior**:
  - Return 200 OK when service is healthy
  - Include version, uptime, dependency status
  - Used by load balancers and Kubernetes probes

<example type="good">
Feature: Health Check Endpoint
  - Endpoint: GET /health
  - Response: {"status": "healthy", "version": "1.0.0", "uptime": 3600}
  - Behavior:
    - Check database connection is alive
    - Check Redis connection is alive  
    - Return 503 if any dependency is down
</example>

#### Feature: API Documentation
- **Description**: Auto-generated API documentation from code
- **Inputs**: Route definitions with type hints and docstrings
- **Outputs**: Interactive API documentation UI
- **Behavior**:
  - Expose OpenAPI spec at /openapi.json
  - Serve Swagger UI at /docs
  - Serve ReDoc at /redoc

#### Feature: Environment Template
- **Description**: Template file documenting all environment variables
- **Inputs**: None (static file)
- **Outputs**: .env.example file
- **Behavior**:
  - List all required variables with descriptions
  - Provide example values (non-sensitive)
  - Document which are required vs optional

<example type="good">
.env.example contents:
```
# Required - Database
DATABASE_URL=postgresql://user:pass@localhost:5432/mydb

# Required - Redis
REDIS_URL=redis://localhost:6379

# Required - API Keys
OPENAI_API_KEY=sk-your-key-here

# Optional - Defaults shown
LOG_LEVEL=INFO
API_RATE_LIMIT=100
```
</example>

#### Feature: Lifespan Initialization
- **Description**: Startup and shutdown logic for the application
- **Inputs**: App configuration
- **Outputs**: Initialized resources (DB pool, caches, etc.)
- **Behavior**:
  - On startup: Create DB connection pool, init pricing provider, warm caches
  - On shutdown: Close DB pool gracefully, flush logs

</bootstrap-capability>

</functional-decomposition>

---

<structural-decomposition>
<instruction>
NOW think about code organization. Map capabilities to actual file/folder structure.

Rules:
1. Each capability maps to a module (folder or file)
2. Features within a capability map to functions/classes
3. Use clear module boundaries - each module has ONE responsibility
4. Define what each module exports (public interface)
5. **NEW**: Ensure bootstrap entrypoint files are explicitly listed

The goal: Create a clear mapping between "what it does" (functional) and "where it lives" (structural).

<example type="good">
Capability: Data Validation
  → Maps to: src/validation/
    ├── schema-validator.js      (Schema validation feature)
    ├── rule-validator.js         (Business rule validation feature)
    └── index.js                  (Public exports)

Capability: Application Bootstrap
  → Maps to: 
    ├── api/main.py              (Application entrypoint)
    ├── config/settings.py       (Configuration management)
    └── .env.example             (Environment template)

Exports:
  - validateSchema(data, schema)
  - validateRules(data, rules)
  - app (FastAPI instance)
  - run() (CLI entrypoint)
</example>

<example type="bad">
Capability: Data Validation
  → Maps to: src/utils.js
  (Problem: "utils" is not a clear module boundary. Where do I find validation logic?)

Capability: Application Bootstrap
  → Not mapped anywhere
  (Problem: CRITICAL - no entrypoint means app can't start!)
</example>
</instruction>

## Repository Structure

```
project-root/
├── api/                        # Maps to: REST API capability
│   ├── main.py                 # **CRITICAL** Application entrypoint
│   ├── routes/
│   │   ├── [domain].py
│   │   └── health.py           # Health check endpoint
│   └── middleware/
├── config/
│   ├── settings.py             # Configuration management
│   └── [config-files].json
├── [domain-module]/            # Maps to: [Domain Capability]
│   ├── [feature].py
│   └── __init__.py
├── tests/
├── docs/
├── .env.example                # **CRITICAL** Environment template
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml              # Must reference api.main:run
```

## Module Definitions

### Module: api
- **Maps to capability**: Application Bootstrap & Developer Experience, REST API
- **Responsibility**: HTTP interface, application initialization, health checks
- **File structure**:
  ```
  api/
  ├── main.py              # **CRITICAL** - Creates app, registers routes, defines run()
  ├── routes/
  │   ├── health.py        # Health check endpoint
  │   └── [domain].py      # Domain-specific routes
  └── middleware/
  ```
- **Exports**:
  - `app` - FastAPI application instance
  - `run()` - Function to start server (used by pyproject.toml scripts)

### Module: config
- **Maps to capability**: Application Bootstrap (Configuration Management)
- **Responsibility**: Load, validate, and expose application configuration
- **File structure**:
  ```
  config/
  ├── settings.py          # Pydantic settings model
  └── [domain].json        # Domain-specific config files
  ```
- **Exports**:
  - `Settings` - Configuration class
  - `get_settings()` - Dependency injection helper

### Module: [Domain Name]
- **Maps to capability**: [Domain Capability]
- **Responsibility**: [Single clear purpose]
- **Exports**:
  - `functionName()` - [what it does]
  - `ClassName` - [what it does]

</structural-decomposition>

---

<dependency-graph>
<instruction>
This is THE CRITICAL SECTION for Task Master parsing.

Define explicit dependencies between modules. This creates the topological order for task execution.

Rules:
1. List modules in dependency order (foundation first)
2. For each module, state what it depends on
3. Foundation modules should have NO dependencies
4. Every non-foundation module should depend on at least one other module
5. Think: "What must EXIST before I can build this module?"
6. **NEW**: Application entrypoint (main.py) depends on ALL route modules and config

<example type="good">
Foundation Layer (no dependencies):
  - error-handling: No dependencies
  - config-manager: No dependencies
  - base-types: No dependencies

Data Layer:
  - schema-validator: Depends on [base-types, error-handling]
  - data-ingestion: Depends on [schema-validator, config-manager]

Core Layer:
  - algorithm-engine: Depends on [base-types, error-handling]
  - pipeline-orchestrator: Depends on [algorithm-engine, data-ingestion]

Bootstrap Layer (LAST - depends on everything it wires together):
  - api/main.py: Depends on [config-manager, all route modules, error-handling]
</example>

<example type="bad">
- validation: Depends on API
- API: Depends on validation
(Problem: Circular dependency. This will cause build/runtime issues.)

- api/main.py: No dependencies listed
(Problem: main.py MUST depend on all routes it registers, or tasks will be out of order)
</example>
</instruction>

## Dependency Chain

### Foundation Layer (Phase 0)
No dependencies - these are built first.

- **[Module Name]**: [What it provides]
- **config/settings.py**: Application configuration loading and validation

### [Domain Layer] (Phase 1)
- **[Module Name]**: Depends on [[module-from-phase-0], [module-from-phase-0]]

### [Another Layer] (Phase 2)
- **[Module Name]**: Depends on [[module-from-phase-1], [module-from-foundation]]

### Bootstrap Layer (Final Phase)
Application entrypoint - depends on all modules it integrates.

- **api/main.py**: Depends on [config/settings, api/routes/*, all domain modules it initializes]
- **.env.example**: Depends on [config/settings - must document all variables it uses]

</dependency-graph>

---

<implementation-roadmap>
<instruction>
Turn the dependency graph into concrete development phases.

Each phase should:
1. Have clear entry criteria (what must exist before starting)
2. Contain tasks that can be parallelized (no inter-dependencies within phase)
3. Have clear exit criteria (how do we know phase is complete?)
4. Build toward something USABLE (not just infrastructure)
5. **NEW**: Final phase should include bootstrap/entrypoint tasks with "smoke test" exit criteria

Phase ordering follows topological sort of dependency graph.

<example type="good">
Phase 0: Foundation
  Entry: Clean repository
  Tasks:
    - Implement error handling utilities
    - Create base type definitions
    - Setup configuration system
  Exit: Other modules can import foundation without errors

Phase 1: Data Layer
  Entry: Phase 0 complete
  Tasks:
    - Implement schema validator (uses: base types, error handling)
    - Build data ingestion pipeline (uses: validator, config)
  Exit: End-to-end data flow from input to validated output

Phase N: Bootstrap & Integration (FINAL)
  Entry: All domain phases complete
  Tasks:
    - Create api/main.py with route registration and lifespan
    - Create .env.example documenting all variables
    - Configure Dockerfile and docker-compose.yml
  Exit: 
    - `uvicorn api.main:app` starts without errors
    - `curl localhost:8000/health` returns 200
    - `docker-compose up` runs successfully
</example>

<example type="bad">
Phase 1: Build Everything
  Tasks:
    - API
    - Database
    - UI
    - Tests
  (Problem: No clear focus. Too broad. Dependencies not considered.)

Phase N: Done
  Exit: "It works"
  (Problem: No concrete, testable exit criteria. What does "works" mean?)
</example>
</instruction>

## Development Phases

### Phase 0: [Foundation Name]
**Goal**: [What foundational capability this establishes]

**Entry Criteria**: [What must be true before starting]

**Tasks**:
- [ ] [Task name] (depends on: [none or list])
  - Acceptance criteria: [How we know it's done]
  - Test strategy: [What tests prove it works]

- [ ] Create config/settings.py with environment validation (depends on: none)
  - Acceptance criteria: Settings class loads from env, fails on missing required vars
  - Test strategy: Test with valid/invalid env configurations

**Exit Criteria**: [Observable outcome that proves phase complete]

**Delivers**: [What can users/developers do after this phase?]

---

### Phase 1: [Layer Name]
**Goal**:

**Entry Criteria**: Phase 0 complete

**Tasks**:
- [ ] [Task name] (depends on: [[tasks-from-phase-0]])
- [ ] [Task name] (depends on: [[tasks-from-phase-0]])

**Exit Criteria**:

**Delivers**:

---

### Phase N: Bootstrap & Integration (FINAL)
**Goal**: Wire all components together into a runnable application

**Entry Criteria**: All domain phases complete

**Tasks**:
- [ ] Create api/main.py with FastAPI app (depends on: [all route modules, config])
  - Acceptance criteria: 
    - App instance created with metadata
    - All routes registered
    - Lifespan manager initializes DB pool and other resources
    - run() function exposed for CLI
  - Test strategy: Import main, verify app has all expected routes

- [ ] Create .env.example (depends on: [config/settings])
  - Acceptance criteria: All required env vars documented with examples
  - Test strategy: New developer can copy and configure

- [ ] Verify pyproject.toml scripts work (depends on: [api/main.py])
  - Acceptance criteria: `rag-advanced` or equivalent command starts server
  - Test strategy: Run script, verify server starts

- [ ] Create health check endpoint (depends on: [api/main.py])
  - Acceptance criteria: GET /health returns 200 with status JSON
  - Test strategy: curl localhost:8000/health

- [ ] Verify Docker setup (depends on: [api/main.py, .env.example])
  - Acceptance criteria: `docker-compose up` starts all services
  - Test strategy: Run docker-compose, verify health endpoints

**Exit Criteria** (Smoke Test):
- [ ] `uvicorn api.main:app --reload` starts without errors
- [ ] `curl http://localhost:8000/health` returns 200 OK
- [ ] `http://localhost:8000/docs` renders OpenAPI documentation
- [ ] `docker-compose up` runs successfully with all services healthy

**Delivers**: 
- Runnable application that can be deployed
- New developers can onboard in < 10 minutes
- CI/CD can verify deployability

</implementation-roadmap>

---

<test-strategy>
<instruction>
Define how testing will be integrated throughout development (TDD approach).

Specify:
1. Test pyramid ratios (unit vs integration vs e2e)
2. Coverage requirements
3. Critical test scenarios
4. Test generation guidelines for Surgical Test Generator
5. **NEW**: Include bootstrap smoke tests as critical scenarios

<example type="good">
Critical Test Scenarios for Data Validation module:
  - Happy path: Valid data passes all checks
  - Edge cases: Empty strings, null values, boundary numbers
  - Error cases: Invalid types, missing required fields
  - Integration: Validator works with ingestion pipeline

Critical Test Scenarios for Application Bootstrap:
  - Smoke test: App starts, health returns 200, docs load
  - Config validation: Missing required env vars fail fast
  - Lifespan: Resources initialized on startup, cleaned on shutdown
</example>
</instruction>

## Test Pyramid

```
        /\
       /E2E\       ← [X]% (End-to-end, slow, comprehensive)
      /------\
     /Integration\ ← [Y]% (Module interactions)
    /------------\
   /  Unit Tests  \ ← [Z]% (Fast, isolated, deterministic)
  /----------------\
```

## Coverage Requirements
- Line coverage: [X]% minimum
- Branch coverage: [X]% minimum
- Function coverage: [X]% minimum
- Statement coverage: [X]% minimum

## Critical Test Scenarios

### Application Bootstrap (api/main.py)
**Smoke test (CRITICAL)**:
- Scenario: Start application and verify basic functionality
- Steps:
  1. Start server: `uvicorn api.main:app`
  2. Check health: `GET /health` → 200 OK
  3. Check docs: `GET /docs` → 200 OK with Swagger UI
- Expected: All checks pass

**Configuration validation**:
- Scenario: Missing required environment variable
- Expected: Application fails to start with clear error message

**Lifespan hooks**:
- Scenario: Startup and shutdown lifecycle
- Expected: 
  - Startup: DB pool created, pricing provider initialized
  - Shutdown: DB pool closed gracefully

### [Module/Feature Name]
**Happy path**:
- [Scenario description]
- Expected: [What should happen]

**Edge cases**:
- [Scenario description]
- Expected: [What should happen]

**Error cases**:
- [Scenario description]
- Expected: [How system handles failure]

**Integration points**:
- [What interactions to test]
- Expected: [End-to-end behavior]

## Test Generation Guidelines
[Specific instructions for Surgical Test Generator about what to focus on, what patterns to follow, project-specific test conventions]

</test-strategy>

---

<architecture>
<instruction>
Describe technical architecture, data models, and key design decisions.

Keep this section AFTER functional/structural decomposition - implementation details come after understanding structure.
</instruction>

## System Components
[Major architectural pieces and their responsibilities]

## Data Models
[Core data structures, schemas, database design]

## Technology Stack
[Languages, frameworks, key libraries]

**Decision: [Technology/Pattern]**
- **Rationale**: [Why chosen]
- **Trade-offs**: [What we're giving up]
- **Alternatives considered**: [What else we looked at]

</architecture>

---

<risks>
<instruction>
Identify risks that could derail development and how to mitigate them.

Categories:
- Technical risks (complexity, unknowns)
- Dependency risks (blocking issues)
- Scope risks (creep, underestimation)
- **NEW**: Bootstrap risks (deployment failures)
</instruction>

## Technical Risks
**Risk**: [Description]
- **Impact**: [High/Medium/Low - effect on project]
- **Likelihood**: [High/Medium/Low]
- **Mitigation**: [How to address]
- **Fallback**: [Plan B if mitigation fails]

## Dependency Risks
[External dependencies, blocking issues]

## Scope Risks
[Scope creep, underestimation, unclear requirements]

## Bootstrap/Deployment Risks
**Risk**: Application entrypoint not defined or misconfigured
- **Impact**: High - Application cannot start in production
- **Likelihood**: Medium (commonly missed in PRDs)
- **Mitigation**: Include explicit "Application Bootstrap" capability in PRD
- **Fallback**: Verify Dockerfile CMD matches actual code before deployment

**Risk**: Missing environment variables in production
- **Impact**: High - Application crashes on startup
- **Likelihood**: High (env vars often undocumented)
- **Mitigation**: Create .env.example, validate config on startup
- **Fallback**: Health check fails, preventing bad deploy

</risks>

---

<appendix>
## References
[Papers, documentation, similar systems]

## Glossary
[Domain-specific terms]

## Open Questions
[Things to resolve during development]

## Completeness Verification Checklist

Before finalizing this PRD, verify:

### Bootstrap & Entrypoints
- [ ] Application entrypoint file defined (e.g., api/main.py)
- [ ] CLI script matches pyproject.toml/package.json
- [ ] Health check endpoint specified
- [ ] API docs endpoint specified

### Configuration
- [ ] .env.example file defined as a deliverable
- [ ] All environment variables documented
- [ ] Config validation on startup specified

### Deployment
- [ ] Dockerfile CMD matches actual entrypoint
- [ ] docker-compose.yml includes all services
- [ ] Health checks configured for all services

### Developer Experience
- [ ] Smoke test exit criteria defined
- [ ] Quick start instructions achievable
- [ ] Test fixtures and mock data available

</appendix>

---

<task-master-integration>
# How Task Master Uses This PRD

When you run `task-master parse-prd <file>.txt`, the parser:

1. **Extracts capabilities** → Main tasks
   - Each `### Capability:` becomes a top-level task
   - **Including** "Application Bootstrap & Developer Experience"

2. **Extracts features** → Subtasks
   - Each `#### Feature:` becomes a subtask under its capability
   - Bootstrap features become concrete tasks (create main.py, create .env.example)

3. **Parses dependencies** → Task dependencies
   - `Depends on: [X, Y]` sets task.dependencies = ["X", "Y"]
   - Bootstrap tasks depend on all modules they integrate

4. **Orders by phases** → Task priorities
   - Phase 0 tasks = highest priority
   - Bootstrap phase = lowest priority (built last, after all dependencies)

5. **Uses test strategy** → Test generation context
   - Feeds test scenarios to Surgical Test Generator during implementation
   - **Smoke tests** verify the application actually runs

**Result**: A dependency-aware task graph that can be executed in topological order, **including the critical bootstrap tasks that wire everything together**.

## Why RPG + Completeness Structure Matters

Traditional flat PRDs lead to:
- ❌ Unclear task dependencies
- ❌ Arbitrary task ordering
- ❌ Circular dependencies discovered late
- ❌ Poorly scoped tasks
- ❌ **Missing entrypoint/bootstrap code** (app can't start!)

RPG + Completeness PRDs provide:
- ✅ Explicit dependency chains
- ✅ Topological execution order
- ✅ Clear module boundaries
- ✅ Validated task graph before implementation
- ✅ **Runnable application** as final deliverable

## Tips for Best Results

1. **Spend time on dependency graph** - This is the most valuable section for Task Master
2. **Keep features atomic** - Each feature should be independently testable
3. **Progressive refinement** - Start broad, use `task-master expand` to break down complex tasks
4. **Use research mode** - `task-master parse-prd --research` leverages AI for better task generation
5. **Always include Bootstrap capability** - Don't assume "obvious" scaffolding will be generated
6. **Define smoke test exit criteria** - Concrete "the app runs" verification
</task-master-integration>
