from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajusta esto en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from services.api_visual_dnn import dnnvisrouter
app.include_router(dnnvisrouter)

from services.model_layers_visualizer import layervisrouter
app.include_router(layervisrouter)

# Montar carpetas estáticas
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/demos", StaticFiles(directory="demos"), name="demos")

# Configurar plantillas
templates = Jinja2Templates(directory="templates")

def get_demos():
    demos_dir = "demos"
    return [name for name in os.listdir(demos_dir) if os.path.isdir(os.path.join(demos_dir, name))]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    demos = get_demos()
    return templates.TemplateResponse("index.html", {"request": request, "demos": demos})

@app.get("/demo/{demo_name}", response_class=HTMLResponse)
async def show_demo(request: Request, demo_name: str):
    demos = get_demos()
    demo_path = os.path.join("demos", demo_name, "index.html")
    if not os.path.exists(demo_path):
        return HTMLResponse(content="Demo no encontrada", status_code=404)
    with open(demo_path, 'r', encoding='utf-8') as f:
        demo_content = f.read()
    return templates.TemplateResponse(
        "demo_template.html",
        {
            "request": request,
            "demos": demos,
            "demo_content": demo_content,
            "demo_name": demo_name
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
