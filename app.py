from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import asyncio

# The pipeline might take a bit to load, we lazy-load or initialize on startup
# For the MVP we will mock the pipeline processing if the models are not downloaded yet
# to allow the WebSocket connection to establish immediately.
try:
    from pipeline import VortexPipeline
    # We will initialize this on startup event
    vortex_engine = None
except ImportError:
    vortex_engine = None

app = FastAPI(title="VORTEX AI Voice Fraud Detection MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global vortex_engine
    if vortex_engine is None:
        try:
            print("Loading VORTEX Pipeline models...")
            vortex_engine = VortexPipeline()
            print("VORTEX Engine Ready.")
        except Exception as e:
            print(f"Failed to load VORTEX Engine: {e}")

@app.get("/")
async def get():
    with open("ui.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to VORTEX stream.")
    
    try:
        while True:
            # Receive audio chunk bytes from the client WebRTC/MediaRecorder
            data = await websocket.receive_bytes()
            
            # Process the chunk in real-time
            if vortex_engine is not None:
                # The engine expects 16kHz 16-bit PCM bytes. 
                # This could block the event loop slightly if the model is heavy,
                # but for MVP this simple continuous loop works.
                try:
                    telemetry = vortex_engine.process_chunk(data)
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    telemetry = {"error": str(e)}
            else:
                # Mock telemetry if models are missing
                telemetry = {
                    "verdict": "MOCK_REAL",
                    "acoustic_fake_probability": 0.1234,
                    "transcript": "Mock text... models not loaded.",
                    "intent_confidence": 0.05,
                    "intent_breakdown": {"urgency": 0.0, "authority impersonation": 0.0, "financial demand": 0.05}
                }
                
            # Send live telemetry back to UI
            await websocket.send_text(json.dumps(telemetry))
            
    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
