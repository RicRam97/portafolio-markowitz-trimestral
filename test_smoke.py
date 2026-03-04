"""Quick smoke test: import the API module and verify it loads."""
try:
    from api import app
    print("[OK] api.py imported successfully")
    
    # Check endpoints exist
    routes = [r.path for r in app.routes]
    assert "/health" in routes, "Missing /health endpoint"
    assert "/api/optimize" in routes, "Missing /api/optimize endpoint"
    assert "/api/tickers" in routes, "Missing /api/tickers endpoint"
    assert "/api/stats" in routes, "Missing /api/stats endpoint"
    assert "/api/dreams_test" in routes, "Missing /api/dreams_test endpoint"
    print("[OK] All expected endpoints registered")
    
    from auth import verify_supabase_token, get_current_user, get_optional_user
    print("[OK] auth.py imported successfully")
    
    from config import CORS_ORIGINS, ENVIRONMENT, RATE_LIMIT_AUTH, RATE_LIMIT_ANON
    print(f"[OK] config.py loaded - ENV={ENVIRONMENT}, CORS origins={len(CORS_ORIGINS)}")
    print(f"     Rate limits: auth={RATE_LIMIT_AUTH}, anon={RATE_LIMIT_ANON}")
    
    print("\n=== ALL SMOKE TESTS PASSED ===")
except Exception as e:
    print(f"[FAIL]: {e}")
    import traceback
    traceback.print_exc()
