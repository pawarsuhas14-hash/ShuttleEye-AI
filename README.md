# ShuttleEye AI Backend v0.3

## Deploy on Render
1. Create a GitHub account/repository.
2. Upload the `backend` folder contents.
3. In Render, create a new Web Service from the repository.
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Copy the resulting HTTPS URL.

The PWA sends videos as multipart/form-data using FastAPI UploadFile, which is suitable for large uploaded files. 
