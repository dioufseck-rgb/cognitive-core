#!/bin/bash
set -e

echo "══════════════════════════════════════════════"
echo " Cognitive Core — Starting"
echo " Entry mode: ${CC_ENTRY:-both}"
echo " Provider:   ${LLM_PROVIDER:-azure_foundry}"
echo "══════════════════════════════════════════════"

case "${CC_ENTRY}" in
  api)
    echo "→ API server on port 8000"
    exec uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 1
    ;;
  foundry)
    echo "→ Foundry adapter on port 8088"
    echo "  Workflow: ${WORKFLOW:-dynamic}"
    echo "  Domain:   ${DOMAIN:-dynamic}"
    exec uvicorn api.foundry_adapter:app --host 0.0.0.0 --port 8088 --workers 1
    ;;
  both|*)
    echo "→ API server on port 8000"
    echo "→ Foundry adapter on port 8088"
    echo "  Workflow: ${WORKFLOW:-dynamic}"
    echo "  Domain:   ${DOMAIN:-dynamic}"
    
    # Start API in background
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 1 &
    API_PID=$!
    
    # Start Foundry adapter in foreground
    uvicorn api.foundry_adapter:app --host 0.0.0.0 --port 8088 --workers 1 &
    FOUNDRY_PID=$!
    
    # Wait for either to exit
    wait -n $API_PID $FOUNDRY_PID
    ;;
esac
