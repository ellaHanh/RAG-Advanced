-- =============================================================================
-- RAG-Advanced Evaluation Schema Extension
-- Extends base schema with evaluation and API management tables
-- =============================================================================

-- -----------------------------------------------------------------------------
-- API Keys Table
-- Stores hashed API keys for authentication
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    key_hash TEXT UNIQUE NOT NULL,  -- SHA256 hash of the API key
    name TEXT,  -- Optional friendly name
    rate_limit INTEGER DEFAULT 100,  -- Requests per minute
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for API key lookups
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_is_active ON api_keys(is_active) WHERE is_active = true;

-- -----------------------------------------------------------------------------
-- Evaluation Runs Table
-- Tracks benchmark and evaluation runs
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    api_key_id UUID REFERENCES api_keys(id) ON DELETE SET NULL,
    run_type TEXT NOT NULL CHECK (run_type IN ('benchmark', 'comparison', 'metrics')),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    config JSONB NOT NULL DEFAULT '{}',
    results JSONB,
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for evaluation run queries
CREATE INDEX IF NOT EXISTS idx_evaluation_runs_api_key_id ON evaluation_runs(api_key_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_runs_status ON evaluation_runs(status);
CREATE INDEX IF NOT EXISTS idx_evaluation_runs_created_at ON evaluation_runs(created_at DESC);

-- -----------------------------------------------------------------------------
-- Strategy Metrics Table
-- Stores per-strategy metrics from evaluations
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS strategy_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evaluation_run_id UUID NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
    strategy_name TEXT NOT NULL,
    query_id TEXT,
    -- IR Metrics
    precision_at_k JSONB,  -- {"3": 0.67, "5": 0.60, "10": 0.50}
    recall_at_k JSONB,
    mrr FLOAT,
    ndcg_at_k JSONB,
    -- Performance Metrics
    latency_ms INTEGER,
    cost_usd FLOAT,
    token_counts JSONB,  -- {"embedding": 10, "llm_input": 50, "llm_output": 20}
    -- Results
    retrieved_doc_ids TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for strategy metrics queries
CREATE INDEX IF NOT EXISTS idx_strategy_metrics_evaluation_run_id ON strategy_metrics(evaluation_run_id);
CREATE INDEX IF NOT EXISTS idx_strategy_metrics_strategy_name ON strategy_metrics(strategy_name);

-- -----------------------------------------------------------------------------
-- Ground Truth Datasets Table
-- Stores test datasets with ground truth for evaluation
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ground_truth_datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    version TEXT DEFAULT '1.0',
    data JSONB NOT NULL,  -- Full dataset including queries and relevance
    query_count INTEGER NOT NULL,
    created_by UUID REFERENCES api_keys(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for dataset lookups
CREATE INDEX IF NOT EXISTS idx_ground_truth_datasets_name ON ground_truth_datasets(name);

-- -----------------------------------------------------------------------------
-- Benchmark Results Table
-- Aggregated benchmark results for reporting
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS benchmark_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evaluation_run_id UUID NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
    strategy_name TEXT NOT NULL,
    -- Aggregated Metrics
    avg_precision_at_k JSONB,
    avg_recall_at_k JSONB,
    avg_mrr FLOAT,
    avg_ndcg_at_k JSONB,
    -- Performance Percentiles
    latency_p50_ms INTEGER,
    latency_p95_ms INTEGER,
    latency_p99_ms INTEGER,
    -- Cost Aggregates
    total_cost_usd FLOAT,
    avg_cost_per_query_usd FLOAT,
    -- Statistics
    query_count INTEGER,
    iteration_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for benchmark result queries
CREATE INDEX IF NOT EXISTS idx_benchmark_results_evaluation_run_id ON benchmark_results(evaluation_run_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_results_strategy_name ON benchmark_results(strategy_name);

-- -----------------------------------------------------------------------------
-- API Key Verification Function
-- Atomic verification and usage update
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION verify_and_update_api_key(key_hash_input TEXT)
RETURNS TABLE (
    id UUID,
    user_id TEXT,
    rate_limit INTEGER,
    is_valid BOOLEAN
)
LANGUAGE plpgsql
AS $$
DECLARE
    key_record api_keys%ROWTYPE;
BEGIN
    -- Atomic update and return
    UPDATE api_keys
    SET last_used_at = NOW()
    WHERE key_hash = key_hash_input
      AND is_active = true
      AND (expires_at IS NULL OR expires_at > NOW())
    RETURNING * INTO key_record;
    
    IF key_record.id IS NOT NULL THEN
        RETURN QUERY SELECT 
            key_record.id,
            key_record.user_id,
            key_record.rate_limit,
            true AS is_valid;
    ELSE
        RETURN QUERY SELECT 
            NULL::UUID,
            NULL::TEXT,
            NULL::INTEGER,
            false AS is_valid;
    END IF;
END;
$$;

-- -----------------------------------------------------------------------------
-- Update Timestamp Triggers
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TRIGGER update_api_keys_updated_at
    BEFORE UPDATE ON api_keys
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE OR REPLACE TRIGGER update_ground_truth_datasets_updated_at
    BEFORE UPDATE ON ground_truth_datasets
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- -----------------------------------------------------------------------------
-- Comments for Documentation
-- -----------------------------------------------------------------------------
COMMENT ON TABLE api_keys IS 'API keys for authentication with SHA256 hashes';
COMMENT ON TABLE evaluation_runs IS 'Tracks benchmark and evaluation job runs';
COMMENT ON TABLE strategy_metrics IS 'Per-query, per-strategy metrics from evaluations';
COMMENT ON TABLE ground_truth_datasets IS 'Test datasets with relevance judgments';
COMMENT ON TABLE benchmark_results IS 'Aggregated benchmark results';
COMMENT ON FUNCTION verify_and_update_api_key IS 'Atomic API key verification with usage update';
