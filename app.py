from pathlib import Path
import os

from dotenv import load_dotenv
from flask import Flask, render_template, redirect, request, url_for

load_dotenv()

app = Flask(__name__) # Flask 애플리케이션(서버) 객체 생성 

# __file__ : 현재 파일의 경로 
# Path(__file__) : 그 문자열 경로를 pathlib.Path 객체로 변환
# .resolve() : 현재 파일의 절대 경로를 반환 
# .parent : 현재 파일의 부모 디렉토리 경로를 반환
BASE_DIR = Path(__file__).resolve().parent   # 프로젝트의 루트 폴더
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/") # / 경로로 GET 요청이 오면 아래 함수 실행 (라우팅)
def home():
    files = sorted([p.name for p in UPLOAD_DIR.glob("*.txt")])
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))
    return render_template("index.html", files=files, has_api_key=has_api_key)

@app.post("/upload")
def upload_file():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return redirect(url_for("home"))
    safe_name = Path(file.filename).name
    if not safe_name.lower().endswith(".txt"):
        return redirect(url_for("home"))
    save_path = UPLOAD_DIR / safe_name
    file.save(save_path)
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)