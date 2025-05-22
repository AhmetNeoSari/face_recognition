from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from dataclasses import dataclass, field
from pathlib import Path
import uvicorn
from typing import Any


@dataclass
class PersonEntryAPI:
    host: str = "0.0.0.0"
    port: int = 5285
    storage_dir: Path = Path("data")
    app: FastAPI = field(default_factory=FastAPI, init=False)

    def __post_init__(self):
        self.setup_routes()

    def setup_routes(self):
        @self.app.post("/Transaction/CreateTransaction")
        async def user_detected(request: Request):
            try:
                data = await request.json()
                event_id = data.get("activityId")
                name = data.get("name")
                print("PersonEntryAPI: ", name)

                if not event_id or not name:
                    raise HTTPException(status_code=400, detail="event_id ve name zorunludur.")

                # Dosyaya kaydet
                entry_path = self.storage_dir / f"{event_id}_log.txt"
                with open(entry_path, "a", encoding="utf-8") as f:
                    f.write(f"{name} went in\n")

                return JSONResponse(content={"message": "Kayıt alındı", "event_id": event_id, "name": name})

            except Exception as e:
                self.logger.error(f"İstek işlenemedi: {str(e)}")
                raise HTTPException(status_code=500, detail="Sunucu hatası")

    def run(self):
        uvicorn.run(self.app, host=self.host, port=self.port)


# CLI'den çalıştırmak için
if __name__ == "__main__":
    service = PersonEntryAPI()
    service.run()
